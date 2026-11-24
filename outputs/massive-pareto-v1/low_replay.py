"""Verify and execute the matched 1e-5 final-stage replay branch."""
from pathlib import Path
import os
import sys

out=Path(__file__).resolve().parent
sys.path.insert(0,str(out.parents[1]/"work/continual-v2"))
from common import MASSIVE, TRAIN, read, sha
from run_massive_full import distributed, evaluate

plan=out/"LOW_REPLAY_PLAN.json"
lock=read(plan)
ref=lock["branch_parent"]
parent=Path(ref["path"])
assert sha(parent/"DONE.json")==ref["certificate_sha256"]
cert=read(parent/"DONE.json")
assert cert["global_step"]==500 and cert["learning_rate"]==1e-5
assert cert["manifest_sha256"]==lock["manifest_sha256"]
for n,h in cert["recovery"].items():
    assert sha(parent/n)==h,n
assert len(cert["recovery"])==4
target=Path(sys.argv[1])
if len(sys.argv)==3:
    assert sys.argv[2]=="verify"
    from full_train import load_inputs
    _,_,stages=load_inputs(lock,"full-replay",None)
    assert stages[-1][0]["file"]=="replay_ur-PK.jsonl"
else:
    opts=["--parent",parent,"--skip","4"]
    distributed(plan,"full-replay",target/"smoke",[*opts,"--smoke"])
    assert read(target/"smoke/DONE.json")["steps"]==502
    distributed(plan,"full-replay",target/"train",opts)
    assert read(target/"train/DONE.json")["steps"]==625
    os.environ["CUDA_VISIBLE_DEVICES"]=os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
    evaluate(target/"train/05_ur-PK",out/"low-replay-eval","dev",MASSIVE)
    (out/"LOW_REPLAY_COMPLETE").touch()
