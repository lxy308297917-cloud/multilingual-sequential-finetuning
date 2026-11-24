"""PEFT LoRA training on the frozen XNLI schedules."""
import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, set_seed

from common import CFG, LANGUAGES, MODEL, ROOT, rows, sha, tokenizer, write_json


def read_schedules(directory, arm):
    data = {lang: rows(directory / f"train_{lang}.jsonl") for lang in LANGUAGES}
    lookup = {row["id"]: row for group in data.values() for row in group}
    stage_size = CFG["train_examples_per_language"]
    if arm == "joint":
        order = json.loads((directory / "joint_order.json").read_text())
        return [(f"joint_{i // stage_size + 1}", [lookup[idx] for idx in order[i:i + stage_size]])
                for i in range(0, len(order), stage_size)]
    if arm == "replay":
        for lang in LANGUAGES[1:]:
            memory = json.loads((directory / f"replay_{lang}.json").read_text())
            allowed = {idx for buffer in memory["buffers"].values() for idx in buffer}
            for position, idx in memory["replacements"].items():
                index = int(position)
                current, old = data[lang][index], lookup[idx]
                if idx not in allowed or old["lang"] not in LANGUAGES[:LANGUAGES.index(lang)]:
                    raise ValueError(f"Invalid old-language row {idx}/{lang}")
                if (len(current["tokens"]), current["label"], current["answer_start"]) != (
                    len(old["tokens"]), old["label"], old["answer_start"]
                ):
                    raise ValueError(f"Replay budget mismatch {lang}/{index}")
                data[lang][index] = old
    return list(data.items())


def batch_loss(base, data, pad, device):
    tokens = [row["tokens"] for row in data]
    width = max(map(len, tokens))
    inputs = torch.tensor([ids + [pad] * (width - len(ids)) for ids in tokens], device=device)
    mask = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids in tokens], device=device)
    item_indices, target_positions = [], []
    for i, row in enumerate(data):
        positions = list(range(row["answer_start"], len(row["tokens"])))
        item_indices.extend([i] * len(positions))
        target_positions.extend(positions)
    item_indices = torch.tensor(item_indices, device=device)
    target_positions = torch.tensor(target_positions, device=device)
    hidden = base.model(input_ids=inputs, attention_mask=mask, use_cache=False).last_hidden_state
    scores = base.lm_head(hidden[item_indices, target_positions - 1]).float()
    targets = inputs[item_indices, target_positions]
    loss = F.cross_entropy(scores, targets, reduction="sum")
    return loss, len(data) * width


def load_resume(path, arm, manifest_digest, exact_stages):
    state = json.loads((path / "DONE.json").read_text())
    if state["manifest_sha256"] != manifest_digest or state["pilot"]:
        raise ValueError("Resume checkpoint has a different data identity or contains pilot weights")
    shared_en = state["arm"] == "seq" and arm == "replay" and state["stage_number"] == 1
    if state["arm"] != arm and not shared_en:
        raise ValueError("Invalid cross-arm continuation")
    if state["stage_number"] not in range(1, len(exact_stages) + 1):
        raise ValueError("Invalid resume stage number")
    if exact_stages[state["stage_number"] - 1][0] != state["stage"]:
        raise ValueError("Invalid resume stage name")
    for name, digest in state["artifact_sha256"].items():
        if sha(path / name) != digest:
            raise ValueError(f"Resume artifact changed: {name}")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--arm", choices=("seq", "replay", "joint"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--from-stage", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--micro-batch", type=int, default=CFG["micro_batch"])
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Training requires exactly one visible CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("Training requires BF16 support")
    if args.pilot and args.from_stage:
        raise ValueError("A pilot starts from the original Base")
    if args.micro_batch <= 0 or CFG["effective_batch"] % args.micro_batch:
        raise ValueError("micro-batch must be a positive divisor of effective batch")
    manifest_file = args.data / "MANIFEST.json"
    manifest_digest = sha(manifest_file)
    manifest = json.loads(manifest_file.read_text())
    if manifest["protocol_sha256"] != sha(ROOT / "protocol.json"):
        raise ValueError("Protocol differs from data manifest")
    for name, digest in manifest["files"].items():
        if sha(args.data / name) != digest:
            raise ValueError(f"Schedule changed: {name}")
    for name, digest in manifest["model"].items():
        if sha(MODEL / name) != digest:
            raise ValueError(f"Base model changed: {name}")
    schedule = read_schedules(args.data, args.arm)
    resume = load_resume(args.from_stage, args.arm, manifest_digest, schedule) if args.from_stage else None
    completed = resume["stage_number"] if resume else 0
    if completed >= len(schedule):
        raise ValueError("All stages in this arm have already completed")
    effective_batch = CFG["effective_batch"]
    if args.pilot:
        stages = [(i, name, dataset[:effective_batch * 8])
                  for i, (name, dataset) in enumerate(schedule[:2], 1)]
    else:
        stages = [(i, name, dataset) for i, (name, dataset) in enumerate(schedule, 1) if i > completed]
    args.output.mkdir(parents=True)
    set_seed(CFG["seed"])
    if resume and resume["micro_batch"] != args.micro_batch:
        raise ValueError("Use the same micro-batch when continuing a scientific run")
    device = torch.device("cuda:0")
    tok = tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = (PeftModel.from_pretrained(model, args.from_stage / "adapter", is_trainable=True)
             if resume else get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **CFG["lora"])))
    model.to(device).train()
    base = model.get_base_model()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("Expected FP32 trainable PEFT parameters")
    optimizer = torch.optim.AdamW(parameters, lr=CFG["learning_rate"], betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.0)
    step = resume["global_steps"] if resume else 0
    if resume:
        optimizer.load_state_dict(torch.load(args.from_stage / "optimizer.pt", map_location="cpu", weights_only=True))
    write_json(args.output / "RUN.json", dict(
        arm=args.arm, seed=CFG["seed"], pilot=args.pilot, base=str(MODEL.resolve()),
        GPU=os.environ.get("CUDA_VISIBLE_DEVICES"), GPU_name=torch.cuda.get_device_name(device),
        trainable_parameters=sum(p.numel() for p in parameters), micro_batch=args.micro_batch,
        effective_batch=effective_batch, continued_from=str(args.from_stage.resolve()) if resume else None,
        manifest_sha256=manifest_digest, protocol_sha256=sha(ROOT / "protocol.json"),
        training_code_sha256=sha(ROOT / "train.py"), torch=torch.__version__,
    ))
    for stage_number, stage, dataset in stages:
        if len(dataset) % effective_batch:
            raise ValueError(f"Stage {stage} does not divide into full effective batches")
        stage_dir = args.output / f"{stage_number:02d}_{stage}"
        stage_dir.mkdir()
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        language_input, language_supervised, language_examples = Counter(), Counter(), Counter()
        padded_tokens, running_loss = 0, 0.0
        effective_counts = Counter()
        with (stage_dir / "events.jsonl").open("x", buffering=1) as log:
            for offset in range(0, len(dataset), effective_batch):
                batch = dataset[offset:offset + effective_batch]
                supervised = sum(len(row["tokens"]) - row["answer_start"] for row in batch)
                optimizer.zero_grad(set_to_none=True)
                step += 1
                optimizer.param_groups[0]["lr"] = CFG["learning_rate"] * min(step / CFG["warmup_steps"], 1.0)
                step_loss = torch.zeros((), device=device)
                for pos in range(0, effective_batch, args.micro_batch):
                    sample = batch[pos:pos + args.micro_batch]
                    loss, padded = batch_loss(base, sample, tok.pad_token_id, device)
                    (loss / supervised).backward()
                    padded_tokens += padded
                    step_loss += loss.detach()
                    for row in sample:
                        lang = row["lang"]
                        language_examples[lang] += 1
                        language_input[lang] += len(row["tokens"])
                        language_supervised[lang] += len(row["tokens"]) - row["answer_start"]
                        effective_counts[lang, row["id"]] += 1
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                loss_avg = (step_loss / supervised).item()
                if not math.isfinite(loss_avg):
                    raise ValueError(f"Non-finite label loss at step {step}")
                running_loss += loss_avg
                stage_step = offset // effective_batch + 1
                if stage_step == 1 or stage_step % 25 == 0 or offset + effective_batch == len(dataset):
                    event = dict(stage=stage, step=step, stage_step=stage_step, grad_norm=norm.item(),
                                 last_loss=loss_avg, mean_loss=running_loss / stage_step,
                                 examples=offset + effective_batch, input_tokens=sum(language_input.values()),
                                 seconds=time.perf_counter() - start)
                    line = json.dumps(event)
                    log.write(line + "\n")
                    print(line, flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        adapter = stage_dir / "adapter"
        model.save_pretrained(adapter, safe_serialization=True)
        torch.save(optimizer.state_dict(), stage_dir / "optimizer.pt")
        artifacts = ("adapter/adapter_config.json", "adapter/adapter_model.safetensors", "optimizer.pt")
        state = dict(
            arm=args.arm, pilot=args.pilot, stage_number=stage_number, stage=stage, global_steps=step,
            manifest_sha256=manifest_digest, micro_batch=args.micro_batch, seconds=elapsed,
            input_tokens_per_second=sum(language_input.values()) / elapsed,
            input_tokens=dict(language_input), supervised_tokens=dict(language_supervised),
            examples=dict(language_examples), distinct_records=len(effective_counts),
            padded_tokens=padded_tokens, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            mean_step_loss=running_loss / (len(dataset) // effective_batch),
            artifact_sha256={name: sha(stage_dir / name) for name in artifacts},
        )
        write_json(stage_dir / "DONE.json", state)
        print(json.dumps(state), flush=True)
    write_json(args.output / "TRAINING_DONE.json", dict(arm=args.arm, pilot=args.pilot, steps=step,
                                                      manifest_sha256=manifest_digest))


if __name__ == "__main__":
    main()
