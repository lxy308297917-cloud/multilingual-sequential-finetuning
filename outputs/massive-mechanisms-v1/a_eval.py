"""Run the frozen MASSIVE scorer on the independent training-side cohorts."""
from pathlib import Path
import json
import sys

root=Path(__file__).resolve().parent
original=root.parent/"massive-cl-v1/evaluate.py"
sys.path.insert(0,str(original.parent))
from common import read_rows
out=Path(sys.argv[1]).resolve()
partition=sys.argv[2]
assert partition in ("dose","mechanism")
count={"dose":128,"mechanism":512}[partition]
lock=json.loads((out/"A_LOCK.json").read_text())
code=original.read_text()
assert code.count('counts = {"dev": 2033, "test": 2974}')==1
code=code.replace('counts = {"dev": 2033, "test": 2974}',f'counts = {{"dev": {count}}}')
code=code.replace('if __name__ == "__main__":\n    main()', '')
scope={"__name__":"cohort_evaluator","__file__":str(original)}
exec(compile(code,str(original),"exec"),scope)
load=scope["load"]

def cohort(name,receipt):
    locale=name.removeprefix("dev_").removesuffix(".jsonl")
    path=out/f"{partition}_{locale}.jsonl"
    rows=read_rows(path)
    assert [r["id"] for r in rows]==lock["cohorts"][f"{partition}/{locale}"]
    receipt["files"][name]=scope["sha"](path)
    return rows

scope["load"]=cohort
arms=sorted((out/"checkpoints").iterdir())
if partition=="mechanism":
    scores={}
    for arm in arms:
        measures=[scope["read"](out/"dose"/arm.name/"dev"/f"{loc}.DONE.json")["slot_micro_f1"] for loc in scope["EVAL"][:5]]
        scores[arm.name]=dict(old=sum(measures[:-1])/4,ur=measures[-1])
    p1=scope["read"](out/"dose/P1/dev/ur-PK.DONE.json")["slot_micro_f1"]
    picked={}
    for method in ("uniform","functional","random","old_loss","format"):
        candidates=[k for k in scores if k.startswith(method+"_") and scores[k]["ur"]>=p1-.01]
        picked[method]=max(candidates,key=lambda k:(scores[k]["old"],-float(k.split("_")[-1]))) if candidates else None
    (out/"A_SELECTION.json").write_text(json.dumps(dict(baseline_ur=p1,selected=picked,scores=scores),indent=2))
    chosen={a for a in picked.values() if a}
    if picked["functional"]:
        dose=picked["functional"].rsplit("_",1)[1]
        chosen|={f"{m}_{dose}" for m in picked}
    arms=[a for a in arms if a.name in chosen]
sys.argv=[str(original),str(out/partition),*(f"{p.name}={p}" for p in arms),
          *(f"P{i}={p}" for i,p in enumerate(lock["parents"])),"--split","dev","--locales",*scope["EVAL"][:5]]
scope["main"]()
