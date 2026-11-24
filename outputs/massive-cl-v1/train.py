"""One stage of BF16 Qwen3-4B SFT, saving immutable PEFT and optimizer states."""
import argparse
import ast
from collections import Counter
import json
import os
from pathlib import Path
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from common import CFG, DATA, LEGACY_TRAINER, MODEL, PROTOCOL, atomic_json, current_model_verified, load, manifest, read, sha


def loss_reference():
    tree = ast.parse(LEGACY_TRAINER.read_text(), filename=str(LEGACY_TRAINER))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "batch_loss"]
    assert len(body) == 1
    scope = {"torch": torch, "F": torch.nn.functional}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(LEGACY_TRAINER), "exec"), scope)
    return scope["batch_loss"]


def checkpoint(model, optimizer, root, step, identity):
    target = root / f"step-{step:03}"
    temporary = root / f".step-{step:03}.staging"
    temporary.mkdir()
    model.save_pretrained(temporary / "adapter", safe_serialization=True)
    torch.save(dict(optimizer=optimizer.state_dict(), cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state()), temporary / "training_state.pt")
    files = ("adapter/adapter_model.safetensors", "adapter/adapter_config.json", "training_state.pt")
    atomic_json(temporary / "DONE.json", dict(identity, stage_step=step,
                                             files={name: sha(temporary / name) for name in files}))
    temporary.rename(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule")
    parser.add_argument("output", type=Path)
    parser.add_argument("--continue-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trajectory", choices=("seq", "replay", "single", "joint"), required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    assert os.environ["CUDA_VISIBLE_DEVICES"].startswith("GPU-") and "," not in os.environ["CUDA_VISIBLE_DEVICES"]
    receipt = manifest()
    current_model_verified(receipt)
    rows = load(args.schedule, receipt)
    effective, micro = CFG["effective_batch"], CFG["micro_batch"]
    if args.smoke:
        rows = rows[:32]
    count, residual = divmod(len(rows), effective)
    assert count and residual == 0 and effective % micro == 0
    start_step = 0
    recovery_path = args.continue_from
    cert = None
    if recovery_path:
        cert = read(recovery_path / "DONE.json")
        can_continue = (cert["trajectory"] == args.trajectory or
                        (cert["trajectory"] == "seq" and args.trajectory == "replay" and
                         cert["schedule"] == "train_de-DE.jsonl" and args.schedule == "replay_ar-SA.jsonl"))
        assert can_continue and cert["smoke"] == args.smoke and cert["complete_stage"]
        assert cert["manifest_sha256"] == sha(DATA / "MANIFEST.json")
        assert cert["schedule"] != args.schedule
        for part, digest in cert["files"].items():
            assert sha(recovery_path / part) == digest, recovery_path / part
        start_step = cert["global_step"]
    eval_steps = [count]
    if args.trajectory in ("seq", "replay") and not args.smoke:
        assert count == CFG["progress_steps"][-1]
        eval_steps = CFG["progress_steps"]
    set_seed(CFG["seed"])
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    if recovery_path:
        model = PeftModel.from_pretrained(model, recovery_path / "adapter", is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **CFG["lora"]))
    model = model.to("cuda:0").train()
    base = model.get_base_model()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert parameters and all(p.dtype == torch.float32 for p in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=CFG["learning_rate"], eps=1e-8, betas=(.9, .999), weight_decay=0)
    if recovery_path:
        state = torch.load(recovery_path / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        assert len(optimizer.state) == len(parameters)
        assert {int(value["step"]) for value in optimizer.state.values()} == {start_step}
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        del state
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    selective_loss = loss_reference()
    args.output.mkdir(parents=True)
    run = dict(trajectory=args.trajectory, schedule=args.schedule, smoke=args.smoke,
               protocol_sha256=sha(PROTOCOL), manifest_sha256=sha(DATA / "MANIFEST.json"),
               schedule_sha256=receipt["files"][args.schedule], seed=CFG["seed"],
               trainer_sha256=sha(__file__), selective_loss_sha256=sha(LEGACY_TRAINER),
               continued_from=str(recovery_path) if recovery_path else None,
               continued_from_sha256=sha(recovery_path / "DONE.json") if recovery_path else None,
               start_step=start_step, gpu=os.environ["CUDA_VISIBLE_DEVICES"], gpu_name=torch.cuda.get_device_name(),
               torch_version=torch.__version__, learning_rate=CFG["learning_rate"], effective_batch=effective, micro_batch=micro)
    atomic_json(args.output / "RUN.json", run)
    exposure, inputs, answers = Counter(), Counter(), Counter()
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with (args.output / "events.jsonl").open("x", buffering=1) as stream:
        for index in range(count):
            step = index + 1
            global_step = start_step + step
            batch = rows[index * effective:step * effective]
            supervised = sum(len(r["tokens"]) - r["answer_start"] for r in batch)
            optimizer.zero_grad(set_to_none=True)
            lr = CFG["learning_rate"] * min(1., global_step / CFG["warmup_steps"])
            optimizer.param_groups[0]["lr"] = lr
            total = torch.zeros((), device="cuda:0")
            for pos in range(0, effective, micro):
                loss, _ = selective_loss(base, batch[pos:pos + micro], tokenizer.pad_token_id, "cuda:0")
                (loss / supervised).backward()
                total += loss.detach()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            for row in batch:
                locale = row["locale"]
                exposure[locale] += 1
                inputs[locale] += len(row["tokens"])
                answers[locale] += len(row["tokens"]) - row["answer_start"]
            event = dict(stage_step=step, global_step=global_step, lr=lr, loss=float(total / supervised),
                         grad_norm=float(norm), seconds=time.monotonic() - start)
            print(json.dumps(event), file=stream, flush=True)
            if step == 1 or step % 10 == 0 or step in eval_steps:
                print(json.dumps(event), flush=True)
            if step in eval_steps:
                torch.cuda.synchronize()
                checkpoint(model, optimizer, args.output, step,
                           dict(run, global_step=global_step, rows=dict(exposure), input_tokens=dict(inputs),
                                supervised_tokens=dict(answers), complete_stage=step == count,
                                seconds=time.monotonic() - start, peak_bytes=torch.cuda.max_memory_reserved()))
    atomic_json(args.output / "TRAINING_DONE.json", dict(run, global_step=start_step + count, rows=dict(exposure),
                                                        input_tokens=dict(inputs), supervised_tokens=dict(answers),
                                                        checkpoint=str(args.output / f"step-{count:03}")))


if __name__ == "__main__":
    main()
