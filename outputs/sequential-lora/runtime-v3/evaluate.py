"""Run the locked XNLI task adapter through lm-evaluation-harness."""
import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable

from common import CFG, LABELS, LANGUAGES, MODEL, ROOT, rows, sha, write_json, write_rows


def verify_score(lm, sample):
    doc = sample["doc"]
    args = [(doc["prompt"], CFG["label_delimiter"] + label) for label in LABELS]
    device = lm.device
    with torch.no_grad():
        values = []
        for context, answer in args:
            prefix, response = lm._encode_pair(context, answer)
            direct = lm.tokenizer.encode(context + answer, add_special_tokens=False)
            if direct != prefix + response:
                raise ValueError("Harness tokenization mismatch")
            input_ids = torch.tensor([direct[:-1]], device=device)
            # Match the Harness model forward, including default KV-cache behavior.
            # Compute the continuation mask and float32 log probabilities independently.
            logits = lm.model(input_ids).logits
            output = logits[0, -len(response):]
            log_probs = output.float().log_softmax(dim=-1)
            target = torch.tensor(response, device=device).unsqueeze(-1)
            values.append(log_probs.gather(-1, target).sum().item())
    observed = [float(item[0]) for item in sample["filtered_resps"]]
    max_error = max(abs(a - b) for a, b in zip(observed, values, strict=True))
    if max_error > 0.005 or int(np.argmax(values)) != int(np.argmax(observed)):
        raise ValueError(f"Independent real-model score comparison failed: {observed} vs {values}")
    return dict(id=doc["id"], direct=values, harness=observed, max_abs_error=max_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("smoke", "validation", "test"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--batch", type=int, default=CFG["eval_batch"])
    parser.add_argument("--stage", type=Path, help="Completed training stage directory, omit for Base")
    args = parser.parse_args()
    if args.device == "cpu" and args.split != "smoke":
        raise ValueError("CPU entry point is sized for the official validation smoke sample")
    if args.device != "cpu" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError("Evaluation requires exactly one visible CUDA GPU")
    manifest_file = args.data / "MANIFEST.json"
    manifest = json.loads(manifest_file.read_text())
    if manifest["protocol_sha256"] != sha(ROOT / "protocol.json"):
        raise ValueError("The evaluation protocol changed")
    repo = Path(CFG["harness_repo"])
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    harness_path = __import__("lm_eval").__file__
    if commit != CFG["harness_commit"] or repo.resolve() not in Path(harness_path).resolve().parents:
        raise ValueError(f"Unexpected Harness source: {harness_path}, {commit}")
    for lang in LANGUAGES:
        for name in (f"tasks/{args.split}/{lang}.yaml", f"{args.split}_{lang}.jsonl"):
            if sha(args.data / name) != manifest["files"][name]:
                raise ValueError(f"Evaluation input changed: {name}")
    for name, digest in manifest["model"].items():
        if sha(MODEL / name) != digest:
            raise ValueError(f"Base model changed: {name}")
    state = None
    if args.stage:
        state = json.loads((args.stage / "DONE.json").read_text())
        if state["manifest_sha256"] != sha(manifest_file) or (state["pilot"] and args.split != "smoke"):
            raise ValueError("Evaluation checkpoint does not match data identity or split")
        for name, digest in state["artifact_sha256"].items():
            if name.startswith("adapter/") and sha(args.stage / name) != digest:
                raise ValueError(f"Adapter changed: {name}")
    args.output.mkdir(parents=True)
    tasks = [f"seqlora_xnli_{args.split}_{lang}" for lang in LANGUAGES]
    manager = TaskManager(include_path=str(args.data / "tasks" / args.split), include_defaults=False)
    dtype = "float32" if args.device == "cpu" else "bfloat16"
    lm = HFLM(
        pretrained=str(MODEL), backend="causal", device=args.device, dtype=dtype,
        softmax_dtype="float32", max_length=CFG["max_length"], add_bos_token=False,
        batch_size=args.batch, peft=str(args.stage / "adapter") if args.stage else None,
        attn_implementation="sdpa",
    )
    result = simple_evaluate(
        model=lm, tasks=tasks, num_fewshot=0, batch_size=args.batch, log_samples=True,
        task_manager=manager, bootstrap_iters=0, random_seed=CFG["seed"],
        numpy_random_seed=CFG["seed"], torch_random_seed=CFG["seed"], fewshot_random_seed=CFG["seed"],
    )
    summaries, comparisons = {}, {}
    for lang, task in zip(LANGUAGES, tasks, strict=True):
        samples = sorted(result["samples"][task], key=lambda row: row["doc_id"])
        expected = rows(args.data / f"{args.split}_{lang}.jsonl")
        if len(samples) != len(expected):
            raise ValueError(f"Unexpected scorer denominator for {lang}")
        details, losses, class_correct, class_counts = [], [], Counter(), Counter()
        for sample, doc in zip(samples, expected, strict=True):
            if sample["doc"] != doc:
                raise ValueError(f"Harness row mismatch for {lang}/{doc['id']}")
            observed = [float(item[0]) for item in sample["filtered_resps"]]
            if not np.isfinite(observed).all():
                raise ValueError(f"Non-finite candidate score for {lang}/{doc['id']}")
            prediction = int(np.argmax(observed))
            correct = int(prediction == doc["label"])
            if float(sample["acc"]) != correct:
                raise ValueError("Harness ACC disagrees with argmax")
            gold = doc["label"]
            class_correct[LABELS[gold]] += correct
            class_counts[LABELS[gold]] += 1
            losses.append(-observed[gold])
            details.append(dict(id=doc["id"], premise_group=doc["premise_group"], label=gold,
                                prediction=prediction, correct=correct, candidate_loglikelihood=observed))
        if args.split == "smoke":
            comparisons[lang] = [verify_score(lm, next(row for row in samples if row["doc"]["label"] == k))
                                 for k in range(len(LABELS))]
        accuracy = sum(row["correct"] for row in details) / len(details)
        if not np.isclose(result["results"][task]["acc,none"], accuracy, atol=1e-12, rtol=0):
            raise ValueError(f"Harness aggregate failed the row-level audit for {lang}")
        predictions = Counter(LABELS[row["prediction"]] for row in details)
        summaries[lang] = dict(
            n=len(details), accuracy=accuracy, correct=sum(class_correct.values()), class_denominators=class_counts,
            class_correct=class_correct, predicted_labels={label: predictions[label] for label in LABELS},
            mean_gold_sequence_nll=float(np.mean(losses)),
        )
        write_rows(args.output / f"predictions_{lang}.jsonl", details)
        serializable = json.loads(json.dumps(samples, default=handle_non_serializable))
        write_rows(args.output / f"harness_samples_{lang}.jsonl", serializable)
    write_json(args.output / "DONE.json", dict(
        split=args.split, source_commit=commit, device=args.device, dtype=dtype, batch=args.batch,
        manifest_sha256=sha(manifest_file), training_state=state,
        accuracy_macro=float(np.mean([value["accuracy"] for value in summaries.values()])),
        languages=summaries, real_score_cross_checks=comparisons,
    ))
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
