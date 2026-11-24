"""Evaluate held-language prediction of per-example slot-F1 deterioration."""
from pathlib import Path
import json
import hashlib
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parent/"gradient"
lock=json.loads((ROOT/"LOCK.json").read_text())
features=ROOT/"features.jsonl"
status=json.loads((ROOT/"FEATURES_DONE.json").read_text())
assert hashlib.sha256(features.read_bytes()).hexdigest()==status["feature_sha256"]


def load(part,loc):
    p=ROOT/"evaluation"/part/"dev"/f"{loc}.jsonl"
    cert=json.loads(p.with_suffix(".DONE.json").read_text())
    assert hashlib.sha256(p.read_bytes()).hexdigest()==cert["prediction_sha256"]
    return {r["id"]:r for r in map(json.loads,p.read_text().splitlines())}


def f1(row):
    den=2*row["tp"]+row["fn"]+row["fp"]
    return 2*row["tp"]/den if den else 0


def fit_predict(x,y,g):
    out=np.zeros(len(y))
    for l in sorted(set(g)):
        train=g!=l
        mu=x[train].mean(axis=0)
        sd=x[train].std(axis=0)
        sd[sd==0]=1
        selected=(x[train]-mu)/sd
        beta=np.linalg.solve(selected.T@selected+np.eye(x.shape[1]),selected.T@(y[train]-y[train].mean()))
        out[~train]=y[train].mean()+(x[~train]-mu)/sd@beta
    return out


def analyze(rows,horizon):
    samples=[]
    for loc in sorted({r["locale"] for r in rows}):
        before=load("P0",loc)
        after=load("P"+str(horizon),loc)
        for r in rows:
            if r["locale"]==loc and r["slot_bearing"]:
                key=r["id"]
                samples.append((r,f1(before[key])-f1(after[key])))
    f,y=zip(*samples)
    y=np.array(y)
    g=np.array([r["locale"] for r in f])
    base=np.array([[r["base_loss"],r["answer"]["norm"],r["answer"]["random"]] for r in f])
    candidates={"base":base}
    for key in ("answer","content","format"):
        candidates[key+"_dot"]=np.column_stack((base,[r[key][f"dot{horizon}"] for r in f]))
        candidates[key+"_new_descent"]=np.column_stack((base,[r[key]["new_descent"] for r in f]))
    predictions={k:fit_predict(x,y,g) for k,x in candidates.items()}
    err={k:abs(v-y) for k,v in predictions.items()}
    rng=np.random.default_rng(42)
    draws=np.concatenate([rng.choice(np.flatnonzero(g==l),(5000,np.count_nonzero(g==l))) for l in sorted(set(g))],axis=1)
    groups={}
    for k,v in predictions.items():
        improvement=err["base"]-err[k]
        answer_key="answer"+k.removeprefix("content") if k.startswith("content") else None
        compared=err[answer_key]-err[k] if answer_key else None
        groups[k]=dict(mae=float(err[k].mean()),spearman=float(spearmanr(v,y).statistic),
                       mae_improvement=float(improvement.mean()),
                       interval=list(map(float,np.quantile(improvement[draws].mean(axis=1),[.025,.975]))),
                       vs_full_answer=([float(compared.mean()),*map(float,np.quantile(compared[draws].mean(axis=1),[.025,.975]))]
                                       if compared is not None else None))
    return dict(samples=len(y),per_locale={l:int(np.count_nonzero(g==l)) for l in sorted(set(g))},
                deterioration_count=int(np.count_nonzero(y>0)),performance=groups,
                raw_correlations={key:float(spearmanr([r[key][f"dot{horizon}"] for r in f],y).statistic)
                                  for key in ("content","answer","format")})

rows=list(map(json.loads,features.read_text().splitlines()))
assert len(rows)==256
result={str(step):analyze(rows,step) for step in (31,125)}
(ROOT/"PREDICTION_RESULTS.json").write_text(json.dumps(result,indent=2)+"\n")
print(result)
