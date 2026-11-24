"""Measure effective LoRA displacement and fixed-prompt changes."""
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file

from pareto import LANGS, MECH, OUT, boot, load, read

root=MECH/"a-recovery"
p0,p1=map(Path,read(root/"A_LOCK.json")["parents"])


def adapter(p):
    data=p/"adapter/adapter_model.safetensors"
    with data.open("rb") as f:
        assert hashlib.file_digest(f,"sha256").hexdigest()==read(p/"DONE.json")["files"]["adapter/adapter_model.safetensors"]
    cfg=read(p/"adapter/adapter_config.json")
    assert cfg["r"]==16 and cfg["lora_alpha"]==32 and not cfg.get("use_rslora")
    return load_file(data),cfg["lora_alpha"]/cfg["r"]


def displacement(candidate,baseline):
    c,s=adapter(candidate)
    b,t=adapter(baseline)
    assert s==t and c.keys()==b.keys()
    import torch
    modules={}
    for k in c:
        if ".lora_A." not in k:continue
        other=k.replace(".lora_A.",".lora_B.")
        ba,bb=b[k].double(),b[other].double()
        ca,cb=c[k].double(),c[other].double()
        # Product of narrow matrices represents s*(B_c A_c - B_b A_b).
        left=torch.cat([cb,-bb],dim=1)
        right=torch.cat([ca,ba],dim=0)
        square=s*s*torch.sum((left.T@left)*(right@right.T)).item()
        modules[k.split(".lora_A.")[0]]=max(square,0)**.5
    return {"effective_l2":sum(x*x for x in modules.values())**.5,"module_l2":modules}


def semantic(text):
    try:
        x=json.loads(text)
        if not isinstance(x,dict) or set(x)!={"intent","slots"}:return None
        return x
    except (ValueError,TypeError):
        return None


def output_change(a,b):
    out={}
    for lang in LANGS:
        path1=a/f"{lang}.jsonl"
        path2=b/f"{lang}.jsonl"
        for p in (path1,path2):
            with p.open("rb") as f:
                assert hashlib.file_digest(f,"sha256").hexdigest()==read(p.with_suffix(".DONE.json"))["prediction_sha256"]
        rows=[{r["id"]:r for r in map(json.loads,p.read_text().splitlines())} for p in (path1,path2)]
        assert set(rows[0])==set(rows[1])
        n=len(rows[0])
        raw=0
        intent=0
        slots=0
        jointly_valid=0
        for i,r in rows[0].items():
            q=rows[1][i]
            raw+=r["output"]!=q["output"]
            x,y=(semantic(v["output"]) for v in (r,q))
            if x is not None and y is not None and r["error"] is None and q["error"] is None:
                jointly_valid+=1
                intent+=x["intent"]!=y["intent"]
                slots+=sorted(map(str,x["slots"]))!=sorted(map(str,y["slots"]))
        out[lang]=dict(n=n,raw_changed=raw,both_valid=jointly_valid,intent_changed=intent,slots_changed=slots)
    return out


def main():
    eff={p.name:displacement(p,p1) for p in sorted((root/"checkpoints").iterdir())}
    eff["P0"]=displacement(p0,p1)
    prompts={a:output_change(root/"dose"/a/"dev",root/"dose/P1/dev") for a in eff}
    smallest={a.name:output_change(a/"dev",root/"mechanism/P1/dev") for a in (root/"mechanism-minimum").iterdir()}
    b=MECH/"b-locked/evaluation"
    raw={a:load(b/a/"dev",2033)[0] for a in ("random","label","parallel")}
    p,x=boot(raw,2033,True)
    from numpy import quantile
    paired={}
    for i,name in enumerate(("random","label")):
        d=x[2]-x[i]
        paired[name]={lang:[float(p[2,j]-p[i,j]),*map(float,quantile(d[j],[.025,.975]))] for j,lang in enumerate(LANGS)}
        paired[name]["old_macro"]=[float((p[2]-p[i])[:4].mean()),*map(float,quantile(d[:4].mean(axis=0),[.025,.975]))]
    (OUT/"AB_AUDIT.json").write_text(json.dumps(dict(effective_weights=eff,dose_prompt_changes=prompts,
          mechanism_minimum_prompt_changes=smallest,b_raw_recomputed=paired),indent=2)+"\n")
    print({k:round(v["effective_l2"],5) for k,v in eff.items() if k.endswith("0.125") or k=="P0"},flush=True)


if __name__=="__main__":
    main()
