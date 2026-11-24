"""Evaluate frozen MASSIVE validation/test records with the verified upstream scorer."""
import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import CFG, DATA, EVAL, MODEL, PREVIOUS, UPSTREAM, atomic_json, current_model_verified, load, manifest, read, sha, write_rows


def inherit_scorer():
    src = PREVIOUS / "eval.py"
    spec = spec_from_file_location("massive_screening_scorer", src)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.grade


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--locales", nargs="+", choices=EVAL, default=EVAL)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    receipt = manifest()
    current_model_verified(receipt)
    manifest_sha = sha(DATA / "MANIFEST.json")
    upstream = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    assert upstream == receipt["upstream_commit"]
    grade = inherit_scorer()
    counts = {"dev": 2033, "test": 2974}
    cohorts = {}
    for locale in args.locales:
        examples = load(f"{args.split}_{locale}.jsonl", receipt)
        assert len(examples) == counts[args.split]
        if args.smoke:
            examples = examples[:12]
            _, check = grade(examples, [{"output": row["answer"]} for row in examples], receipt)
            assert all(check[k] == 1 for k in ("intent_acc", "slot_micro_f1", "exact", "strict_exact"))
        cohorts[locale] = examples

    from vllm_compat import install
    install()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    files = []
    for entry in args.checkpoints:
        label, path = entry.split("=", 1)
        assert label and all(part and part not in (".", "..") for part in label.split("/"))
        request, cert = None, None
        if path != "base":
            checkpoint = Path(path)
            cert = read(checkpoint / "DONE.json")
            assert cert["manifest_sha256"] == manifest_sha and cert["smoke"] == args.smoke
            for name, digest in cert["files"].items():
                if name.startswith("adapter/"):
                    assert sha(checkpoint / name) == digest
            request = LoRARequest(label, len(files) + 1, str(checkpoint / "adapter"))
        files.append((label, request, cert))

    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    assert gpu.startswith("GPU-") and "," not in gpu
    gen = CFG["generation"]
    engine = LLM(model=MODEL, tokenizer=MODEL, dtype="bfloat16", tensor_parallel_size=1,
                 max_model_len=gen["max_model_len"], max_num_seqs=gen["max_num_seqs"],
                 max_num_batched_tokens=gen["max_num_batched_tokens"], gpu_memory_utilization=.78,
                 seed=CFG["seed"], generation_config="vllm", enable_prefix_caching=True, trust_remote_code=False,
                 enforce_eager=True, enable_lora=any(record[1] for record in files), max_lora_rank=16)
    params = SamplingParams(temperature=gen["temperature"], top_p=gen["top_p"], max_tokens=gen["max_new_tokens"],
                            seed=CFG["seed"], stop_token_ids=gen["stop_token_ids"], skip_special_tokens=True)
    for label, adapter, cert in files:
        output = args.output / label / args.split
        output.mkdir(parents=True, exist_ok=True)
        for locale, cohort in cohorts.items():
            progress = output / f"{locale}.DONE.json"
            prediction = output / f"{locale}.jsonl"
            if progress.exists():
                prior = read(progress)
                assert prior["checkpoint"] == cert and prior["manifest_sha256"] == manifest_sha
                assert prior["prediction_sha256"] == sha(prediction)
                continue
            n = len(cohort)
            start = time.monotonic()
            raw = []
            for off in range(0, n, gen["submit_batch"]):
                batch = [{"prompt_token_ids": row["tokens"][:row["answer_start"]]} for row in cohort[off:off + gen["submit_batch"]]]
                decoded = engine.generate(batch, params, lora_request=adapter, use_tqdm=False)
                for item, result in zip(batch, decoded, strict=True):
                    assert item["prompt_token_ids"] == list(result.prompt_token_ids)
                    out = result.outputs[0]
                    raw.append(dict(output=out.text, generated_tokens=len(out.token_ids), finish=out.finish_reason))
                if len(raw) == n or len(raw) % 512 == 0:
                    print(json.dumps(dict(checkpoint=label, split=args.split, locale=locale, completed=len(raw), n=n,
                                          seconds=time.monotonic() - start)), flush=True)
            graded, metrics = grade(cohort, raw, receipt)
            write_rows(prediction, graded)
            result = dict(checkpoint=cert, name=label, locale=locale, n=n, smoke=args.smoke, split=args.split, gpu=gpu,
                          manifest_sha256=manifest_sha, inputs_sha256=receipt["files"][f"{args.split}_{locale}.jsonl"],
                          prediction_sha256=sha(prediction), length_stop=sum(x["finish"] == "length" for x in raw),
                          generation=gen, evaluator_sha256=sha(__file__), upstream_commit=upstream,
                          seconds=time.monotonic() - start, **metrics)
            atomic_json(progress, result)
            print(json.dumps({k: result[k] for k in ("name", "locale", "n", "intent_acc", "slot_micro_f1", "exact", "parse_failure")}), flush=True)


if __name__ == "__main__":
    main()
