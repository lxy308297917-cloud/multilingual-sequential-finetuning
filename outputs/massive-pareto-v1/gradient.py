"""Measure first-order predictors at the frozen swapped Replay stage boundary."""
from pathlib import Path
import json
import random
import sys
import time

ROOT=Path(__file__).resolve().parent
HISTORY=ROOT.parent/"massive-cl-single-order-v1"
sys.path.insert(0,str(HISTORY))
from common import CHECKPOINTS, DATA, LANGUAGES, MODEL, load, manifest, read, sha, write_rows

OLD=LANGUAGES[:-1]
CURRENT=LANGUAGES[-1]
OUT=ROOT/"gradient"
N=64


def parents():
    home=CHECKPOINTS/"replay"
    return (home/"04_ur-PK/step-125",home/"05_sw-KE/step-031",home/"05_sw-KE/step-125")


def freeze():
    OUT.mkdir()
    locks={}
    for locale in LANGUAGES:
        rows=load(f"train_{locale}.jsonl",manifest())
        random.Random(42).shuffle(rows)
        path=OUT/f"{locale}.jsonl"
        write_rows(path,rows[:N])
        locks[locale]={"sha256":sha(path),"ids":[r["id"] for r in rows[:N]]}
    (OUT/"LOCK.json").write_text(json.dumps(dict(cohorts=locks,seed=42,n=N,
      checkpoints={str(p):sha(p/"DONE.json") for p in parents()},source=sha(__file__)),indent=2))


def compute():
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM,AutoTokenizer
    start=time.monotonic()
    p0,p31,p125=parents()
    lock=read(OUT/"LOCK.json")
    assert lock["source"]==sha(__file__)
    weights=[]
    for p in parents():
        assert sha(p/"DONE.json")==lock["checkpoints"][str(p)]
        path=p/"adapter/adapter_model.safetensors"
        assert sha(path)==read(p/"DONE.json")["files"]["adapter/adapter_model.safetensors"]
        weights.append(load_file(path))
    for p in (p31,p125):
        assert read(p/"DONE.json")["continued_from_sha256"]==sha(p0/"DONE.json")
    keys=sorted(weights[0])
    vectors=[torch.cat([(w[k]-weights[0][k]).flatten() for k in keys]).cuda() for w in weights[1:]]
    rng=torch.Generator(device="cuda").manual_seed(42)
    rand=torch.randn(vectors[1].shape,device="cuda",generator=rng)
    rand=rand/rand.norm()*vectors[1].norm()
    base=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa")
    model=PeftModel.from_pretrained(base,p0/"adapter",is_trainable=True).to("cuda").train()
    base.config.use_cache=False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    named={n.replace(".default.","."):p for n,p in model.named_parameters() if p.requires_grad}
    assert set(named)==set(keys)
    params=[named[k] for k in keys]
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    receipt=manifest()
    new_mean=torch.zeros_like(rand)
    with (OUT/"features.jsonl").open("x") as stream:
        for locale in (CURRENT,*OLD):
            path=OUT/f"{locale}.jsonl"
            assert sha(path)==lock["cohorts"][locale]["sha256"]
            for row in map(json.loads,path.read_text().splitlines()):
                prefix=row["tokens"][:row["answer_start"]]
                correct=row["tokens"]
                wrong=json.loads(row["answer"])
                if wrong["slots"]:
                    s=wrong["slots"][0]
                    s[0]=next(x for x in sorted(receipt["slots"]) if x!=s[0])
                else:
                    wrong["intent"]=next(x for x in sorted(receipt["intents"]) if x!=wrong["intent"])
                other=tok(row["prompt"]+" "+json.dumps(wrong,ensure_ascii=False,separators=(",",":")),add_special_tokens=False).input_ids
                assert other[:len(prefix)]==prefix
                def losses(ids):
                    ids=torch.tensor([ids],device="cuda")
                    hidden=base.model(input_ids=ids,use_cache=False).last_hidden_state[0,len(prefix)-1:-1]
                    return torch.nn.functional.cross_entropy(base.lm_head(hidden).float(),ids[0,len(prefix):],reduction="none")
                a=losses(correct)
                if locale==CURRENT:
                    grad=torch.autograd.grad(a.mean(),params)
                    new_mean.add_(torch.cat([g.flatten() for g in grad]),alpha=1/N)
                    continue
                b=losses(other+[tok.eos_token_id])
                mask=[bool(s.strip()) and all(c in '{}[]":, ' for c in s)
                      for s in map(lambda i:tok.decode([i]),correct[len(prefix):])]
                assert any(mask)
                row_info=dict(id=row["id"],locale=locale,base_loss=a.mean().item(),slot_bearing=bool(wrong["slots"]))
                for k,obj in [("content",a.mean()-b.mean()),("answer",a.mean()),("format",a[mask].mean())]:
                    grad=torch.autograd.grad(obj,params,retain_graph=k!="format")
                    g=torch.cat([v.flatten() for v in grad])
                    row_info[k]=dict(dot31=torch.dot(g,vectors[0]).item(),dot125=torch.dot(g,vectors[1]).item(),
                                     random=torch.dot(g,rand).item(),norm=g.norm().item(),new_descent=-torch.dot(g,new_mean).item())
                print(json.dumps(row_info),file=stream,flush=True)
            print(locale,flush=True)
    (OUT/"FEATURES_DONE.json").write_text(json.dumps(dict(seconds=time.monotonic()-start,
        peak_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),
        feature_sha256=sha(OUT/"features.jsonl")),indent=2))


def evaluate():
    source=HISTORY/"evaluate.py"
    code=source.read_text().replace('counts = {"dev": 2033, "test": 2974}',f'counts = {{"dev": {N}}}')
    code=code.replace('if __name__ == "__main__":\n    main()', '')
    scope={"__file__":str(source),"__name__":"gradient_cohort"}
    exec(compile(code,str(source),"exec"),scope)
    lock=read(OUT/"LOCK.json")
    def cohort(name,receipt):
        locale=name.removeprefix("dev_").removesuffix(".jsonl")
        path=OUT/f"{locale}.jsonl"
        assert sha(path)==lock["cohorts"][locale]["sha256"]
        receipt["files"][name]=sha(path)
        return list(map(json.loads,path.read_text().splitlines()))
    scope["load"]=cohort
    sys.argv=[str(source),str(OUT/"evaluation"),*(f"{label}={p}" for label,p in zip(("P0","P31","P125"),parents())),
              "--split","dev","--locales",*LANGUAGES]
    scope["main"]()
    (OUT/"COMPLETE").touch()


if __name__=="__main__":
    {"freeze":freeze,"compute":compute,"evaluate":evaluate}[sys.argv[1]]()
