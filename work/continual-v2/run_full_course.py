"""Execute the five-language full-parameter course conditions and paired scores."""

import argparse
from pathlib import Path

from common import CKPT, LOGS, ROOT, read, save, sha
from run_course import evaluate, full_course


def run(task,job,out):
    path=ROOT/"plans"/f"{task}-course.json"
    plan=read(path)
    order=plan["order"]
    dest=CKPT/f"{task}-course"
    seq=dest/"full-seq"
    replay=dest/"full-replay"
    controls=dest/"full-controls"
    first=seq/"train"/plan["sequential"][0]["name"]
    if job=="train-seq":
        full_course(path,"full-seq","joint",seq)
    elif job=="train-replay":
        cert=read(first/"DONE.json")
        assert cert["arm"]=="full-seq" and cert["global_step"]==64
        assert cert["schedule_sha256"]==read(Path(plan["data"])/"MANIFEST.json")["files"][f"train_{order[0]}.jsonl"]
        full_course(path,"full-replay","joint",replay,selected_parent=first,skip=1)
    elif job=="train-controls":
        assert read(first/"DONE.json")["global_step"]==64
        for control in ("joint",*order[1:]):
            full_course(path,"full-controls",control,controls/control,smoke=False)
    elif job=="eval-singles":
        assert task=="ner"
        next_lang=next(l for l in order if l in plan["candidate_single_scores"])
        fitted=Path(plan["matched_lora"][next_lang]["path"])
        evaluate(path,"test",out/"base/test", [next_lang])
        evaluate(path,"test",out/f"single_{next_lang}/test",[next_lang],fitted)
    elif job=="eval-controls":
        for control in ("joint",*order):
            if control==order[0]:
                ckpt=first
            else:
                tag="joint" if control=="joint" else f"single_{control}"
                ckpt=controls/control/"train"/tag
            langs=order if control=="joint" else (control,)
            for split in ("dev","test"):
                evaluate(path,split,out/("joint" if control=="joint" else f"single_{control}")/split,langs,ckpt,full=True)
    else:
        is_seq=job=="eval-seq"
        fitted=seq if is_seq else replay
        assert read(fitted/"train/DONE.json")["steps"]==320
        for i,stage in enumerate(plan["sequential"]):
            if i==0 and not is_seq:
                continue
            name=stage["name"]
            for split in ("dev","test"):
                evaluate(path,split,out/name/split,order,fitted/"train"/name,full=True)
    save(out/"DONE.json",dict(benchmark=task,job=job,plan_sha256=sha(path)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("task",choices=("ner","mtop"))
    p.add_argument("job",choices=("train-seq","train-replay","train-controls","eval-seq","eval-replay","eval-controls","eval-singles"))
    p.add_argument("key")
    a=p.parse_args()
    target=LOGS/a.key
    target.mkdir(parents=True)
    run(a.task,a.job,target)
