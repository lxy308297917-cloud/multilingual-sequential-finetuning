"""Freeze the existing MASSIVE schedules for the missing Full-FT controls."""

import os as _release_os
import sys as _release_sys

from pathlib import Path

from common import ACCOUNT, CKPT, MASSIVE, ROOT, WORKSPACE, budgets, dataset, read, save, schedule, sha, source

MASSIVE_DATA = WORKSPACE / "work/massive-cl-v1-data"
SWAP_DATA = WORKSPACE / "work/massive-cl-single-order-v1-data"
PREFIX = Path(_release_os.path.join(_release_os.environ.get("CL_EXTERNAL_ROOT", "local"), 'checkpoints/massive_full_sft_v1_20261121/train/01_de-DE'))
LOW_FIRST = Path(_release_os.path.join(_release_os.environ.get("CL_EXTERNAL_ROOT", "local"), 'checkpoints/massive_full_sft_low_lr_v1_20261122/train/01_de-DE'))


def verify_resume(path, step, learning_rate):
    cert = read(path / "DONE.json")
    assert cert["global_step"] == step and cert["learning_rate"] == learning_rate and cert["seed"] == 42
    assert cert["rank_micro_batch"] == 4 and cert["effective_batch"] == 16
    assert cert["manifest_sha256"] == sha(MASSIVE_DATA / "MANIFEST.json")
    assert len(cert["weights"]) >= 5 and len(cert["recovery"]) == 4
    assert all((path / name).is_file() for name in (*cert["weights"], *cert["recovery"]))
    return dict(path=str(path), certificate_sha256=sha(path / "DONE.json"), global_step=step)


def make():
    root = ROOT / "plans"
    root.mkdir(exist_ok=True)
    for order,folder,key in [(MASSIVE,MASSIVE_DATA,"original"),
                             ((*MASSIVE[:3], MASSIVE[4], MASSIVE[3]),SWAP_DATA,"swap")]:
        plan = dict(benchmark="massive", arm="full-replay", order=order, seed=42,
                    learning_rate=5e-5, warmup_steps=16, micro_batch=4, steps_per_stage=125,
                    max_context=1536, **source(folder))
        data, manifest = dataset(plan)
        stages, exposed = [], set()
        for i,lang in enumerate(order):
            current = schedule(data,manifest,f"train_{lang}.jsonl")
            file = f"replay_{lang}.jsonl" if i else f"train_{lang}.jsonl"
            replay = schedule(data,manifest,file)
            assert budgets(current) == budgets(replay)
            assert all(len(a["tokens"])==len(b["tokens"]) and a["answer_start"]==b["answer_start"]
                       for a,b in zip(current,replay,strict=True))
            assert all(row["locale"]==lang or row["id"] in exposed for row in replay)
            old=sum(len(row["tokens"]) for row in replay if row["locale"]!=lang)
            assert i==0 or .095<old/budgets(replay)["input_tokens"]<.105
            exposed.update(row["id"] for row in replay if row["locale"]==lang)
            stages.append(dict(name=f"{i+1:02}_{lang}",file=file,**budgets(replay),replay_input_tokens=old))
        plan.update(stages=stages, reused_first=verify_resume(PREFIX,125,5e-5))
        save(root/f"massive-full-replay-{key}.json",plan)
    rate=1e-5
    plan=dict(benchmark="massive", arm="full-low-controls",order=MASSIVE,seed=42,
              learning_rate=rate,warmup_steps=16,micro_batch=4,steps_per_stage=125,
              max_context=1536,reused_single_de=verify_resume(LOW_FIRST,125,rate),**source(MASSIVE_DATA))
    data,manifest=dataset(plan)
    plan["single"]={lang:budgets(schedule(data,manifest,f"train_{lang}.jsonl")) for lang in MASSIVE}
    joint=schedule(data,manifest,"joint.jsonl")
    assert budgets(joint)=={key:sum(r[key] for r in plan["single"].values()) for key in budgets(joint)}
    plan["joint"]=budgets(joint)
    save(root/"massive-full-low-controls.json", plan)


if __name__=="__main__":
    make()
