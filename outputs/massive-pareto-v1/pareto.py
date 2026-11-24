"""Reconstruct development-set learning and retention frontiers."""

import os as _release_os
import sys as _release_sys
from pathlib import Path
import csv
import hashlib
import json

import numpy as np

OUT=Path(__file__).resolve().parent
LOG=Path(_release_os.path.join(_release_os.environ.get("CL_EXTERNAL_ROOT", "local"), 'logs'))
NEXT=LOG/"continual_bench_next_20261122"
V2=LOG/"continual_bench_v2_20261122"
MECH=OUT.parent/"massive-mechanisms-v1"
LANGS=("de-DE","ar-SA","vi-VN","sw-KE","ur-PK")
NAME={k:k[:2].upper() for k in LANGS}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def load(path, n):
    result={}
    receipt={}
    for lang in LANGS:
        cert=read(path/f"{lang}.DONE.json")
        pred=path/f"{lang}.jsonl"
        assert cert["n"]==n and not cert["smoke"] and cert["split"]=="dev"
        assert digest(pred)==cert["prediction_sha256"]
        with pred.open() as f: rows=list(map(json.loads,f))
        result[lang]={r["id"]:(r["tp"],r["fp"],r["fn"]) for r in rows}
        assert len(result[lang])==n
        checkpoint=cert.get("full_sft",cert["checkpoint"])
        receipt[lang]={"receipt":str(path/f"{lang}.DONE.json"),"prediction_sha256":cert["prediction_sha256"],
                       "data_sha256":cert["inputs_sha256"],"manifest_sha256":cert["manifest_sha256"],
                       "checkpoint":checkpoint,"eval_gpu_seconds":cert["seconds"]}
    return result,receipt


def score(counts):
    tp,fp,fn=np.moveaxis(counts,-1,0)
    d=2*tp+fp+fn
    return np.divide(200*tp,d,out=np.zeros_like(d,dtype=float),where=d!=0)


def boot(grid,n,shared):
    rng=np.random.default_rng(42)
    reference=next(iter(grid.values()))
    ids={l:sorted(reference[l]) for l in LANGS}
    if shared:
        assert all(ids[l]==ids[LANGS[0]] for l in LANGS)
    matrix=[]
    values=[]
    draws=rng.integers(0,n,(3000,n),dtype=np.int32)
    for lang in LANGS:
        arrays=[]
        for cohort in grid.values():
            assert set(cohort[lang])==set(ids[lang])
            arrays.append([cohort[lang][i] for i in ids[lang]])
        arrays=np.asarray(arrays,dtype=np.int16)
        if not shared: draws=rng.integers(0,n,(3000,n),dtype=np.int32)
        point=score(arrays.sum(axis=1))
        samples=np.concatenate([score(arrays[:,d,:].sum(axis=2)) for d in np.array_split(draws,75)],axis=1)
        matrix.append(samples)
        values.append(point)
    return np.array(values).T,np.stack(matrix,axis=1)


def paths(order):
    swap=order=="UR→SW"
    last="05_sw-KE" if swap else "05_ur-PK"
    prev="04_ur-PK" if swap else "04_sw-KE"
    log=LOG/("massive_cl_single_order_v1_20261121" if swap else "massive_cl_v1_20261120")
    op={}
    parents={}
    def add(method,step,path,parent):
        tag=f"{method}/{step}"
        op[tag]=(method,step,path/"dev")
        parents[method]=parent/"dev"
    for method,m in [("SeqLoRA","seq"),("LoRA Replay","replay")]:
        start=log/m/prev/"step-125"
        add(method,0,start,start)
        for step in (31,63,94):
            path=log/f"progress-{m}-05"/f"step-{step:03}"
            if (path/"dev").is_dir(): add(method,step,path,start)
        add(method,125,log/m/last/"step-125",start)
    high=NEXT/"full-swap-test" if swap else LOG/"massive_full_sft_v1_20261121/evaluation"
    for method,root in [("Full-FT 5e-5",high),
                        ("Full-FT Replay 5e-5",V2/("massive-replay-swap-eval" if swap else "massive-replay-full-eval"))]:
        add(method,0,root/prev,root/prev)
        add(method,125,root/last,root/prev)
    if not swap:
        low=NEXT/"full-low-lr-dev"
        add("Full-FT 1e-5",0,low/prev,low/prev)
        add("Full-FT 1e-5",125,low/last,low/prev)
        if (OUT/"LOW_REPLAY_COMPLETE").exists():
            add("Full-FT Replay 1e-5（单阶段）",125,OUT/"low-replay-eval"/last,low/prev)
    return op


def audit_operating_points(label, ops, n, shared, current, baselines, reference):
    arrays={}
    receipts={}
    for key,(_,_,path) in ops.items():
        arrays[key],receipts[key]=load(path,n)
    keys=list(ops)
    for lang in LANGS:
        assert len({x[lang]["data_sha256"] for x in receipts.values()})==1
    points,samples=boot(arrays,n,shared)
    ix={k:i for i,k in enumerate(keys)}
    cur=LANGS.index(current)
    old=[i for i in range(5) if i!=cur]
    metrics=list(LANGS)+["old_macro"]
    def vector(x): return np.concatenate([x,x[old].mean(axis=0,keepdims=True)],axis=0)
    def ci(v,x):
        qs=np.quantile(x,[.025,.975],axis=1)
        return {m:[float(v[j]),float(qs[0,j]),float(qs[1,j])] for j,m in enumerate(metrics)}
    records=[]
    for j,key in enumerate(keys):
        method,step,_=ops[key]
        p=ix[baselines[method]]
        records.append(dict(id=key,method=method,step=step,scores=ci(vector(points[j]),vector(samples[j])),
                            changes=ci(vector(points[j]-points[p]),vector(samples[j]-samples[p]))))
    q=ix[reference]
    candidates=[j for j in range(len(keys)) if abs(points[j,cur]-points[q,cur])<=1]
    old_target="sw-KE" if current!="sw-KE" else "ur-PK"
    selected={}
    for target in [old_target,"old_macro"]:
        which=max(candidates,key=lambda j:records[j]["scores"][target][0])
        selected[target]={"winner":keys[which],"compared_with":reference,
                          "differences":ci(vector(points[which]-points[q]),vector(samples[which]-samples[q])),
                          "eligible":[keys[j] for j in candidates]}
    match={}
    for j,key in enumerate(keys):
        if key==reference: continue
        if ops[key][1] not in (125,12.5,50):continue
        match[key]=ci(vector(points[j]-points[q]),vector(samples[j]-samples[q]))
    comparable={}
    for j,key in enumerate(keys):
        if ops[key][1]==0 and key!=reference:
            continue
        for k in range(j):
            if (ops[keys[k]][1]==0 and keys[k]!=reference) or abs(points[j,cur]-points[k,cur])>1:
                continue
            comparable[f"{key} − {keys[k]}"]=ci(vector(points[j]-points[k]),vector(samples[j]-samples[k]))
    frontier={}
    for m in [*(l for l in LANGS if l!=current),"old_macro"]:
        frontier[m]=[r["id"] for r in records if not any(
            s["scores"][current][0]>=r["scores"][current][0] and s["scores"][m][0]>=r["scores"][m][0] and
            (s["scores"][current][0]>r["scores"][current][0] or s["scores"][m][0]>r["scores"][m][0])
            for s in records)]
    return dict(cohort=label,n=n,current=current,reference=reference,records=records,selected=selected,
                frontier=frontier, contrasts=match, comparable=comparable, artifacts=receipts)


def main():
    result={}
    for label,current in [("SW→UR","ur-PK"),("UR→SW","sw-KE")]:
        ops=paths(label)
        baselines={m:f"{m}/0" for m,_,_ in ops.values()}
        baselines["Full-FT Replay 1e-5（单阶段）"]="Full-FT 1e-5/0"
        result[label]=audit_operating_points(label,ops,2033,True,current,baselines,"SeqLoRA/125")
    root=MECH/"a-recovery"
    ops={k:(k,0,root/"mechanism"/k/"dev") for k in ("P0","P1")}
    for k in ("uniform","functional","random","old_loss","format"):
        ops[k+"/12.5"]=(k,12.5,root/"mechanism-minimum"/(k+"_0.125")/"dev")
    ops["random/50"]=("random",50,root/"mechanism/random_0.5/dev")
    result["独立训练侧"]=audit_operating_points("独立训练侧",ops,512,False,"ur-PK",
                                                {m:"P0" for m,_,_ in ops.values()},"P1")
    (OUT/"PARETO.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    with (OUT/"PER_LANGUAGE.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["语料","方法","阶段步或撤回百分比","语言","阶段分数","95%下界","95%上界",
                    "相对自身阶段起点差","95%下界","95%上界"])
        for label,v in result.items():
            for row in v["records"]:
                for lang in row["scores"]:
                    w.writerow([label,row["method"],row["step"],lang,*row["scores"][lang],*row["changes"][lang]])
    return result


if __name__=="__main__":
    main()
