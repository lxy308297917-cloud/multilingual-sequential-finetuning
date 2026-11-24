"""Run frozen course baselines without changing historical files."""

import argparse
from pathlib import Path

from common import CKPT, EVAL, LOGS, ROOT, TRAIN, dataset, read, save, sha
from gpu_runtime import execute
from run_massive_full import distributed


def evaluate(plan,split,out,langs,model=None,full=False):
    args=["--full" if full else "--lora",model] if model else []
    execute([EVAL,"-u",ROOT/"eval_worker.py",plan,split,out,*langs,*args])


def train_lora(plan,file,out,arm,steps,parent=None):
    options=["--parent",parent] if parent else []
    execute([TRAIN,"-u",ROOT/"lora_worker.py",plan,file,out,"--arm",arm,*options])
    end=out/f"step-{steps:03}"
    cert=read(end/"DONE.json")
    assert cert["complete_stage"] and not cert.get("smoke") and cert["schedule"]==file
    assert cert["manifest_sha256"]==read(plan)["manifest_sha256"]
    if parent:
        prior=read(parent/"DONE.json")
        assert cert["start_step"]==prior["global_step"] and cert["global_step"]==steps+prior["global_step"]
        assert cert["continued_from_sha256"]==sha(parent/"DONE.json")
    return end


def screen(plan_path,output):
    plan=read(plan_path)
    order=plan["order"]
    evaluate(plan_path,"dev",output/"base/dev",order)
    for l in order:
        model=train_lora(plan_path,f"train_{l}.jsonl",CKPT/"ner-screen/single"/l,"single",64)
        evaluate(plan_path,"dev",output/f"single_{l}/dev",[l],model)


def lora_job(plan_path,arm,output):
    plan=read(plan_path)
    order=plan["order"]
    dest=CKPT/f"{plan['benchmark']}-course"/f"lora-{arm}"
    prefix=plan["matched_lora"][order[0]]
    if arm=="joint":
        last=train_lora(plan_path,"joint.jsonl",dest/"joint","joint",320)
        stages=[("joint",last)]
    else:
        first=Path(prefix["path"])
        assert sha(first/"DONE.json")==prefix["certificate_sha256"]
        specs=plan["sequential"] if arm=="seq" else plan["replay"]
        assert specs[0]["file"]==f"train_{order[0]}.jsonl"
        stages=[(specs[0]["name"],first)]
        for n,stage in enumerate(specs[1:],2):
            end=train_lora(plan_path,stage["file"],dest/stage["name"],arm,64,stages[-1][1])
            stages.append((stage["name"],end))
    save(output/"STAGES.json", {label:str(model) for label,model in stages})
    for label,model in stages:
        for split in ("dev","test"):
            evaluate(plan_path,split,output/label/split,order,model)


def full_course(plan_path,arm,control,dest,selected_parent=None,skip=0,smoke=True):
    opts=["--control",control]
    if selected_parent:
        opts.extend(["--parent",selected_parent,"--skip",str(skip)])
    if smoke:
        distributed(plan_path,arm,dest/"smoke",[*opts,"--smoke"])
        result=read(dest/"smoke/DONE.json")
        assert result["steps"]==64*skip+2 and result["smoke"]
    distributed(plan_path,arm,dest/"train",opts)
    result=read(dest/"train/DONE.json")
    required=64 if arm=="full-controls" and control!="joint" else 320
    assert result["steps"]==required and not result["smoke"]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("task",choices=("ner","mtop"))
    p.add_argument("job",choices=("screen","lora-seq","lora-replay","lora-joint"))
    p.add_argument("key")
    a=p.parse_args()
    out=LOGS/a.key
    out.mkdir(parents=True)
    plan=ROOT/"plans"/f"{a.task}-{'screen' if a.job=='screen' else 'course'}.json"
    if a.job=="screen":
        assert a.task=="ner"
        screen(plan,out)
    else:
        lora_job(plan,a.job.removeprefix("lora-"),out)
    save(out/"DONE.json",dict(benchmark=a.task,job=a.job,plan_sha256=sha(plan)))


if __name__=="__main__":
    main()
