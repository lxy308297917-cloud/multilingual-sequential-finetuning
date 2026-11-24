"""Summarize matched MASSIVE experiment predictions and bootstrap intervals."""

import os as _release_os
import sys as _release_sys
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"massive-cl-v1"))
from common import CHECKPOINTS, LANGUAGES, read, read_rows, sha

FIELDS=("slot_f1","intent_accuracy","exact_match","json_valid")
OLD=LANGUAGES[:-1]
UR=LANGUAGES[-1]
SW=OLD[-1]


def load(path):
    receipt=read(path.with_suffix(".DONE.json"))
    assert sha(path)==receipt["prediction_sha256"]
    rows={r["id"]:r for r in read_rows(path)}
    assert len(rows)==receipt["n"]
    return rows


def point(sums,n):
    tp,fp,fn,*rest=np.moveaxis(sums,-1,0)
    denom=2*tp+fp+fn
    f1=np.divide(200*tp,denom,out=np.zeros_like(denom,dtype=float),where=denom>0)
    return np.stack((f1,*(100*r/n for r in rest)),axis=-1)


def matrix(rows,ids):
    return np.array([[r["tp"],r["fp"],r["fn"],r["intent_correct"],r["exact"],r["error"]!="invalid_json"]
                     for r in (rows[i] for i in ids)])


def contrast(group,treatment,control,langs,shared,valid=False):
    rng=np.random.default_rng(42)
    diffs=[]
    means=[]
    common_draw=None
    ordered=sorted(set.intersection(*(set(group[a][l]) for a in (treatment,control) for l in langs))) if shared else None
    for lang in langs:
        a,b=(group[k][lang] for k in (treatment,control))
        assert set(a)==set(b)
        ids=ordered if shared else sorted(a)
        if valid:
            ids=[i for i in ids if a[i]["error"] is None and b[i]["error"] is None]
        assert ids
        nums=[matrix(s,ids) for s in (a,b)]
        if common_draw is None or not shared or valid:
            draws=rng.integers(0,len(ids),(5000,len(ids)),dtype=np.int32)
            if shared and not valid:
                common_draw=draws
        else:
            draws=common_draw
        vals=[np.concatenate([point(x[index].sum(axis=1),len(ids)) for index in np.array_split(draws,50)]) for x in nums]
        diffs.append(vals[0]-vals[1])
        means.append(point(nums[0].sum(axis=0),len(ids))-point(nums[1].sum(axis=0),len(ids)))
    ci=np.quantile(np.mean(diffs,axis=0),[.025,.975],axis=0)
    return {f:dict(delta=float(v),lo=float(lo),hi=float(hi)) for f,v,lo,hi in zip(FIELDS,np.mean(means,axis=0),*ci)}


def report(mode,out):
    shared=mode=="B"
    root=out/"evaluation" if shared else out/"mechanism"
    baseline=Path(_release_os.path.join(_release_os.environ.get("CL_EXTERNAL_ROOT", "local"), 'logs/massive_cl_v1_20261120/seq/04_sw-KE/step-125'))
    arms=sorted(p for p in root.iterdir() if p.is_dir())
    arms+=([baseline] if shared else [])
    identities={("P0" if p==baseline else p.name):p for p in arms}
    group={a:{l:load(p/"dev"/f"{l}.jsonl") for l in LANGUAGES} for a,p in identities.items()}
    result={"scores":{},"changes":{},"contrasts":{},"valid_subset":{}}
    for arm,g in group.items():
        cells={}
        for lang,preds in g.items():
            scores=point(matrix(preds,sorted(preds)).sum(axis=0),len(preds))
            cells[lang]=dict(zip(FIELDS,map(float,scores)))
            receipt=read(identities[arm]/"dev"/f"{lang}.DONE.json")
            cells[lang].update(schema_valid=(1-receipt["parse_failure"])*100,errors=receipt["errors"],
                               eval_seconds=receipt["seconds"])
        cells["macro"]={f:float(np.mean([x[f] for x in cells.values()])) for f in FIELDS}
        result["scores"][arm]=cells
    selection={} if shared else read(out/"A_SELECTION.json")["selected"]
    if shared:
        comparisons=[(a,"P0") for a in ("random","label","parallel")]+[("parallel",a) for a in ("random","label")]
    else:
        comparisons=[(a,"P1") for a in group if a!="P1"]
        f=selection["functional"]
        if f:
            dose=f.rsplit("_",1)[1]
            comparisons += [(f,k) for k in group if k.rsplit("_",1)[-1]==dose and k!=f]
            comparisons += [(f,k) for k in set(selection.values()) if k and k!=f and (f,k) not in comparisons]
            result["valid_subset"][f]=contrast(group,f,"P1",[SW],shared,valid=True)
    for a,b in comparisons:
        patches={l:contrast(group,a,b,[l],shared) for l in LANGUAGES}
        patches["old_macro"]=contrast(group,a,b,OLD,shared)
        patches["macro"]=contrast(group,a,b,LANGUAGES,shared)
        result["contrasts"][f"{a}-{b}"]=patches
    (out/"RESULTS.json").write_text(json.dumps(result,indent=2))
    with (out/"SCORES.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["arm","language",*FIELDS,"schema_valid"])
        for a,cells in result["scores"].items():
            for l,v in cells.items():
                writer.writerow([a,l,*[v[f] for f in FIELDS],v.get("schema_valid","")])
    print(mode,"results",len(group),len(comparisons),flush=True)


if __name__=="__main__":
    mode,out=sys.argv[1:]
    assert mode in ("A","B")
    report(mode,Path(out).resolve())
