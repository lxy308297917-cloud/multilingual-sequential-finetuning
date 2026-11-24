"""Launch separately verifiable MASSIVE V2 full-parameter training and scores."""

import argparse
from pathlib import Path

from common import CKPT, EVAL, HISTORICAL, LOGS, MASSIVE, OUTPUTS, ROOT, TRAIN, read, save, sha
from gpu_runtime import execute


def distributed(plan,arm,dest,opts):
    execute([TRAIN,"-u","-m","torch.distributed.run","--standalone","--nproc-per-node=4","--max-restarts=0",
             ROOT/"full_train.py",plan,arm,dest,*opts],4,21600)


def replay_train(order, checkpoint=None, skip=0, recovery=None):
    plan=ROOT/f"plans/massive-full-replay-{order}.json"
    settings=read(plan)
    dest=CKPT/f"massive-full-replay-{order}"
    if recovery:
        dest=dest/recovery
    initial=settings["reused_first"]
    parent=Path(checkpoint or initial["path"])
    if not checkpoint:
        assert sha(parent/"DONE.json")==initial["certificate_sha256"]
    skip=skip or 1
    opts=["--parent",parent,"--skip",str(skip)]
    distributed(plan,"full-replay",dest/"smoke",[*opts,"--smoke"])
    smoke=read(dest/"smoke/DONE.json")
    assert smoke["steps"]==125*skip+2 and smoke["smoke"] and smoke["stages"][0]["peak_bytes"]>0
    distributed(plan,"full-replay",dest/"train",opts)
    result=read(dest/"train/DONE.json")
    assert result["steps"]==625 and not result["smoke"]


def full_controls():
    path=ROOT/"plans/massive-full-low-controls.json"
    plan=read(path)
    first=plan["reused_single_de"]
    assert sha(Path(first["path"])/"DONE.json")==first["certificate_sha256"]
    root=CKPT/"massive-full-low-controls"
    for control in ("joint",*MASSIVE[1:]):
        dest=root/control/"train"
        distributed(path,"full-low-controls",dest,["--control",control])
        result=read(dest/"DONE.json")
        assert result["steps"]==(625 if control=="joint" else 125) and not result["smoke"]
    save(root/"DONE.json",dict(plan_sha256=sha(path),reuse_de=first))


def evaluate(checkpoint,root,split,langs):
    cert=read(checkpoint/"DONE.json")
    assert cert["stage"]
    script=OUTPUTS/"massive-full-sft-v1/evaluate_full.py"
    execute([EVAL,"-u",script,checkpoint,root,split,*langs])
    for lang in langs:
        folder=root/cert["stage"]/split
        row=read(folder/f"{lang}.DONE.json")
        assert row["prediction_sha256"]==sha(folder/f"{lang}.jsonl") and row["full_sft"]==cert


def replay_test(order,output):
    labels=read(ROOT/f"plans/massive-full-replay-{order}.json")["order"]
    path=CKPT/f"massive-full-replay-{order}/train"
    assert read(path/"DONE.json")["steps"]==625
    for i,lang in enumerate(labels[1:],2):
        for split in ("dev","test"):
            evaluate(path/f"{i:02}_{lang}",output,split,labels)


def low_test(output):
    root=CKPT/"massive-full-low-controls"
    assert (root/"DONE.json").is_file()
    for control in ("joint",*MASSIVE[1:]):
        checkpoint=root/control/"train"/("joint" if control=="joint" else f"single_{control}")
        langs=MASSIVE if control=="joint" else (control,)
        for split in ("dev","test"):
            evaluate(checkpoint,output,split,langs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("job",choices=("replay-train","replay-test","low-train","low-test"))
    p.add_argument("log_key")
    p.add_argument("--order",choices=("original","swap"),default="original")
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--skip",type=int,default=0)
    p.add_argument("--recovery")
    args=p.parse_args()
    output=LOGS/args.log_key
    output.mkdir(parents=True)
    if args.job=="replay-train":
        replay_train(args.order,args.checkpoint,args.skip,args.recovery)
    elif args.job=="low-train":
        full_controls()
    elif args.job=="replay-test":
        replay_test(args.order,output)
    else:
        low_test(output)
    save(output/"DONE.json",dict(job=args.job,order=args.order,driver_sha256=sha(__file__)))


if __name__=="__main__":
    main()
