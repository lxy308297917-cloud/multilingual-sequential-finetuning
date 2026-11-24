"""Run matched full-parameter SFT with four-rank FSDP and immutable checkpoints."""

import argparse
import ast
from collections import Counter
from datetime import timedelta
from functools import partial
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from common import MODEL, OUTPUTS, budgets, dataset, read, save, schedule, sha

WORLD = 4
BATCH = 16
LOSS = OUTPUTS / "sequential-lora/train.py"


def loss_function():
    tree = ast.parse(LOSS.read_text(encoding="utf-8"))
    definition, = (node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="batch_loss")
    scope = {"torch":torch,"F":torch.nn.functional}
    exec(compile(ast.Module(body=[definition],type_ignores=[]),str(LOSS),"exec"),scope)
    return scope["batch_loss"]


class AnswerLoss(nn.Module):
    def __init__(self, model, pad):
        super().__init__()
        self.lm = model
        self.pad = pad
        self.loss = loss_function()

    def forward(self, items, device):
        return self.loss(self.lm, items, self.pad, device)[0]


def load_inputs(plan, arm, control):
    data, receipt = dataset(plan)
    if arm.endswith("replay"):
        specs = plan["replay"] if "replay" in plan else plan["stages"]
    elif arm.endswith("seq"):
        specs = plan["sequential"]
    elif control=="joint":
        specs = [dict(name="joint",file="joint.jsonl",**plan["joint"])]
    else:
        specs = [dict(name=f"single_{control}",file=f"train_{control}.jsonl",**plan["single"][control])]
    outputs = []
    for stage in specs:
        rows=schedule(data,receipt,stage["file"])
        assert len(rows)%BATCH==0 and budgets(rows)=={k:stage[k] for k in budgets(rows)}
        assert all(0<r["answer_start"]<len(r["tokens"])<=plan["max_context"] for r in rows)
        outputs.append((stage,rows))
    return data, receipt, outputs


def parent_certificate(parent, expected_step):
    if not parent:
        assert expected_step == 0
        return None
    path = parent / "DONE.json"
    cert = read(path)
    assert cert["global_step"] == expected_step and not cert["smoke"], path
    return cert


def save_stage(model, optimizer, tokenizer, output, stage, settings, rank, resumable):
    temporary = output / f".{stage}.staging"
    target = output / stage
    if rank==0:
        assert not target.exists()
        temporary.mkdir()
    dist.barrier()
    if resumable:
        file = temporary / f"recovery-rank{rank:02d}.pt"
        torch.save(dict(optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),
                        gpu_rng=torch.cuda.get_rng_state()),file)
    with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
        state = model.state_dict()
    if rank==0:
        assert all(key.startswith("lm.") for key in state)
        state = {key[3:]:v for key,v in state.items()}
        assert sum(x.numel() for x in state.values())>=settings["parameters"]
        model.module.lm.save_pretrained(temporary/"model",state_dict=state,safe_serialization=True,max_shard_size="4GB")
        tokenizer.save_pretrained(temporary/"model")
    del state
    dist.barrier()
    if rank==0:
        weights={str(f.relative_to(temporary)):sha(f) for f in sorted((temporary/"model").glob("*")) if f.is_file()}
        recovery={f.name:sha(f) for f in sorted(temporary.glob("recovery-rank*.pt"))}
        save(temporary/"DONE.json",dict(settings,stage=stage,smoke=False,weights=weights,recovery=recovery))
        temporary.rename(target)
        print(json.dumps({"checkpoint":str(target),"global_step":settings["global_step"]}),flush=True)
    dist.barrier()


def restore(optimizer, parent, cert, rank, expected_step):
    if not cert:
        return
    name = f"recovery-rank{rank:02d}.pt"
    path = parent/name
    assert sha(path)==cert["recovery"][name], path
    state = torch.load(path,map_location="cpu",weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    assert {int(x["step"]) for x in optimizer.state.values()}=={expected_step}
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["gpu_rng"])


def run(options,plan,data,receipt,stages):
    rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"])==torch.cuda.device_count()==WORLD
    torch.cuda.set_device(rank)
    assert torch.cuda.is_bf16_supported()
    device=torch.device("cuda",rank)
    dist.init_process_group(backend="nccl",timeout=timedelta(minutes=40),device_id=device)
    try:
        cert=parent_certificate(options.parent,options.skip*plan["steps_per_stage"])
        if rank==0:
            if cert:
                assert cert["learning_rate"]==plan["learning_rate"] and cert["seed"]==plan["seed"]
                assert cert["rank_micro_batch"]==plan["micro_batch"] and cert["effective_batch"]==BATCH
                assert all(sha(options.parent/name)==digest for name,digest in cert["weights"].items())
            else:
                base=read(OUTPUTS/"massive-cl-v1/DATA_MANIFEST.json")["model"]
                assert all(sha(MODEL/name)==digest for name,digest in base.items())
            options.output.mkdir(parents=True)
        dist.barrier()
        set_seed(plan["seed"])
        original = AutoModelForCausalLM.from_pretrained(
            options.parent/"model" if cert else MODEL,local_files_only=True,dtype=torch.float32,attn_implementation="sdpa")
        parameters=sum(p.numel() for p in original.parameters())
        original.config.use_cache=False
        original.requires_grad_(True)
        original.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        amp=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.bfloat16)
        model=FSDP(AnswerLoss(original,tok.pad_token_id),device_id=device,mixed_precision=amp,limit_all_gathers=True,
                   auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls={Qwen3DecoderLayer})).train()
        assert all(p.requires_grad and p.dtype==torch.float32 for p in model.parameters())
        optimizer=torch.optim.AdamW(model.parameters(),lr=plan["learning_rate"],betas=(.9,.999),eps=1e-8,
                                   weight_decay=0,fused=True)
        seen=options.skip*plan["steps_per_stage"]
        restore(optimizer,options.parent,cert,rank,seen)
        run_info=dict(benchmark=plan["benchmark"],arm=options.arm,control=options.control,seed=plan["seed"],
                      learning_rate=plan["learning_rate"],warmup_steps=plan["warmup_steps"],parameters=parameters,
                      micro_batch=plan["micro_batch"],rank_micro_batch=plan["micro_batch"],effective_batch=BATCH,
                      gradient_accumulation=(BATCH//WORLD)//plan["micro_batch"],world_size=WORLD,
                      language_order=plan["order"],plan_sha256=sha(options.plan),manifest_sha256=sha(data/"MANIFEST.json"),
                      trainer_sha256=sha(__file__),loss_sha256=sha(LOSS),
                      initializer=str(options.parent) if cert else str(MODEL),
                      initializer_sha256=sha(options.parent/"DONE.json") if cert else None,
                      optimizer="AdamW fused FP32 masters/moments, BF16 compute, FP32 FSDP reduction",
                      gpu_uuids=os.environ["CUDA_VISIBLE_DEVICES"],torch_version=torch.__version__)
        if rank==0:
            save(options.output/"PROTOCOL.json",run_info)

        remaining=stages[options.skip:]
        if options.smoke:
            batches=[rows[i:i+BATCH] for _,rows in remaining for i in range(0,len(rows),BATCH)]
            stress=max(batches,key=lambda batch:(max(len(r["tokens"]) for r in batch),sum(len(r["tokens"]) for r in batch)))
            remaining=[(remaining[0][0],batches[0]+stress)]
        segment_summaries = []
        for i,(stage,rows) in enumerate(remaining):
            torch.cuda.reset_peak_memory_stats()
            started=time.monotonic()
            with (options.output/f"{stage['name']}.rank{rank:02d}.jsonl").open("x",buffering=1) as stream:
                for offset in range(0,len(rows),BATCH):
                    batch=rows[offset:offset+BATCH]
                    sup=sum(len(r["tokens"])-r["answer_start"] for r in batch)
                    local=batch[rank*4:(rank+1)*4]
                    optimizer.zero_grad(set_to_none=True)
                    total=torch.zeros((),device=device)
                    micro=plan["micro_batch"]
                    for pos in range(0,len(local),micro):
                        loss=model(local[pos:pos+micro],device)
                        (loss*(WORLD/sup)).backward()
                        total+=loss.detach()
                    norm=model.clip_grad_norm_(1.)
                    if not torch.isfinite(norm):
                        raise FloatingPointError(f"Nonfinite gradient at global step {seen+1}")
                    seen+=1
                    scale=plan["learning_rate"]*min(1,seen/plan["warmup_steps"])
                    optimizer.param_groups[0]["lr"]=scale
                    optimizer.step()
                    dist.all_reduce(total)
                    event=dict(stage=stage["name"],step=offset//BATCH+1,global_step=seen,lr=scale,
                               loss=float(total/sup),grad_norm=float(norm),seconds=time.monotonic()-started,
                               peak_bytes=torch.cuda.max_memory_reserved())
                    assert math.isfinite(event["loss"])
                    print(json.dumps(event),file=stream)
                    if rank==0 and (event["step"]==1 or event["step"]%10==0 or offset+BATCH==len(rows)):
                        print(json.dumps(event),flush=True)
            elapsed=time.monotonic()-started
            max_mem=torch.tensor(torch.cuda.max_memory_reserved(),device=device,dtype=torch.int64)
            dist.all_reduce(max_mem,op=dist.ReduceOp.MAX)
            counts=Counter(r["locale"] for r in rows)
            info=dict(run_info,**budgets(rows),global_step=seen,seconds=elapsed,peak_bytes=int(max_mem),
                      stage_seconds=elapsed,gpu_seconds_compute=elapsed*WORLD,example_counts=dict(counts),
                      schedule=stage["file"],schedule_sha256=receipt["files"][stage["file"]])
            segment_summaries.append(dict(stage=stage["name"],**info))
            if not options.smoke:
                save_stage(model,optimizer,tok,options.output,stage["name"],info,rank,i<len(remaining)-1 or options.keep_final_recovery)
        if rank==0:
            save(options.output/"DONE.json",dict(smoke=options.smoke,steps=seen,stages=segment_summaries,
                                                 protocol_sha256=sha(options.output/"PROTOCOL.json")))
    finally:
        dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan",type=Path)
    p.add_argument("arm",choices=("full-seq","full-replay","full-low-controls","full-controls"))
    p.add_argument("output",type=Path)
    p.add_argument("--control",default="joint")
    p.add_argument("--parent",type=Path)
    p.add_argument("--skip",type=int,default=0)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--keep-final-recovery",action="store_true")
    a=p.parse_args()
    plan=read(a.plan)
    data,receipt,stages=load_inputs(plan,a.arm,a.control)
    assert (BATCH//WORLD)%plan["micro_batch"]==0
    assert bool(a.parent)==(a.skip>0) and a.skip<len(stages)
    if a.preflight:
        with torch.device("meta"):
            skeleton=AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(MODEL,local_files_only=True))
        print(json.dumps(dict(plan=str(a.plan),arm=a.arm,control=a.control,skip=a.skip,
                              parameters=sum(p.numel() for p in skeleton.parameters()),
                              manifest_sha256=sha(data/"MANIFEST.json"),stages=[s for s,_ in stages]),indent=2))
        return
    run(a,plan,data,receipt,stages)


if __name__=="__main__":
    main()
