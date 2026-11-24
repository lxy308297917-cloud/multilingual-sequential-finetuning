"""Construct the preregistered MASSIVE checkpoint interventions."""
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "massive-cl-v1"))
from common import CHECKPOINTS, DATA, LANGUAGES, MODEL, load, manifest, read, sha, write_rows

DOSES = (.125, .25, .5)
METHODS = ("uniform", "functional", "random", "old_loss", "format")


def parents():
    stage = CHECKPOINTS / "seq"
    p0, p1 = [stage / s / "step-125" for s in ("04_sw-KE", "05_ur-PK")]
    assert read(p1 / "DONE.json")["continued_from_sha256"] == sha(p0 / "DONE.json")
    for p in (p0, p1):
        for name, digest in read(p / "DONE.json")["files"].items():
            assert sha(p / name) == digest
    return p0, p1


def freeze(out):
    import time
    out.mkdir()
    receipt = manifest()
    p0, p1 = parents()
    cohorts = {}
    for lang in LANGUAGES:
        rows = load(f"train_{lang}.jsonl", receipt)
        random.Random(42).shuffle(rows)
        for name, selection in [("locate", rows[:64] if lang != LANGUAGES[-1] else []),
                                ("dose", rows[64:192]), ("mechanism", rows[192:704])]:
            write_rows(out / f"{name}_{lang}.jsonl", selection)
            cohorts[f"{name}/{lang}"] = [r["id"] for r in selection]
    (out / "A_LOCK.json").write_text(json.dumps(dict(seed=42, rank=16, sketch_rows=24, doses=DOSES,
        parents=[str(p0),str(p1)], parent_sha256=[sha(p/"DONE.json") for p in (p0,p1)],
        cohorts=cohorts, seconds=time.time(), source_sha256=sha(__file__)), indent=2))


def build(out):
    import time
    start=time.monotonic()
    p0,p1=parents()
    w0,w1=[load_file(p/"adapter/adapter_model.safetensors") for p in (p0,p1)]
    keys=sorted(w1)
    assert w0.keys()==w1.keys()
    delta=torch.cat([(w1[k]-w0[k]).reshape(-1) for k in keys])
    base=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa")
    model=PeftModel.from_pretrained(base,p1/"adapter",is_trainable=True).to("cuda").train()
    base.config.use_cache=False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    named={n.replace(".default.","."):p for n,p in model.named_parameters() if p.requires_grad}
    assert set(named)==set(keys)==set(get_peft_model_state_dict(model))
    params=[named[k] for k in keys]
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    rng=random.Random(42)
    receipt=manifest()
    sketches=torch.zeros((3,24,delta.numel()))
    for lang in LANGUAGES[:-1]:
        rows=read_rows_file(out/f"locate_{lang}.jsonl")
        for row in rows:
            good=json.loads(row["answer"])
            bad=json.loads(row["answer"])
            if bad["slots"]:
                slot=bad["slots"][0]
                slot[0]=next(s for s in sorted(receipt["slots"]) if s!=slot[0])
            else:
                bad["intent"]=next(k for k in sorted(receipt["intents"]) if k!=bad["intent"])
            prefix=row["tokens"][:row["answer_start"]]
            wrong=" "+json.dumps(bad,ensure_ascii=False,separators=(",",":"))
            wrong_tokens=tok(row["prompt"]+wrong,add_special_tokens=False).input_ids
            assert wrong_tokens[:len(prefix)]==prefix
            scores=[]
            for seq in (row["tokens"],wrong_tokens+[tok.eos_token_id]):
                ids=torch.tensor([seq],device="cuda")
                h=base.model(input_ids=ids,use_cache=False).last_hidden_state[0,len(prefix)-1:-1]
                lp=-torch.nn.functional.cross_entropy(base.lm_head(h).float(),ids[0,len(prefix):],reduction="none")
                scores.append(lp)
            mask=[bool(piece.strip()) and all(c in '{}[]":, ' for c in piece)
                  for piece in [tok.decode([i]) for i in row["tokens"][len(prefix):]]]
            assert any(mask), good
            objectives=(scores[0].mean()-scores[1].mean(),-scores[0].mean(),scores[0][mask].mean())
            bucket,sign=rng.randrange(24),rng.choice((-1,1))
            for i,objective in enumerate(objectives):
                grad=torch.autograd.grad(objective,params,retain_graph=i<2)
                sketches[i,bucket].add_(torch.cat([g.flatten() for g in grad]).cpu(),alpha=sign)
        print("定位梯度完成",lang,flush=True)
    d=delta.cuda()
    directions={"uniform":d}
    for method, sketch in zip(("functional","old_loss","format"),sketches):
        sketch=sketch.cuda()
        gram=sketch@sketch.T
        eigenvalues,u=torch.linalg.eigh(gram)
        assert (eigenvalues[-16:]>0).all()
        basis=(u[:,-16:].T@sketch)/eigenvalues[-16:].sqrt()[:,None]
        directions[method]=basis.T@(basis@d)
    rng_gpu=torch.Generator(device="cuda").manual_seed(42)
    directions["random"]=torch.randn(d.shape,device=d.device,generator=rng_gpu)
    groups={}
    offset=0
    for k in keys:
        end=offset+w1[k].numel()
        groups.setdefault(k.split(".lora_")[0],[]).append(slice(offset,end))
        offset=end
    audits={}
    for method, direction in directions.items():
        audit={}
        direction=direction.clone()
        for group, spans in groups.items():
            raw=torch.cat([direction[s] for s in spans])
            update=torch.cat([d[s] for s in spans])
            orient=-1 if method=="random" and torch.dot(raw,update)<0 else 1
            ratio=(update.norm()/raw.norm()).item()
            for span in spans:
                direction[span].mul_(orient*ratio)
            audit[group]={"update_norm":update.norm().item(),"raw_direction_norm":raw.norm().item(),
                          "cosine":float(torch.nn.functional.cosine_similarity(raw,update,dim=0))*orient}
        correction=direction.cpu()
        for dose in DOSES:
            label=f"{method}_{dose:g}"
            target=out/"checkpoints"/label/"adapter"
            target.mkdir(parents=True)
            split=correction.split([w1[k].numel() for k in keys])
            weights={k:w1[k]-dose*v.reshape(w1[k].shape) for k,v in zip(keys,split)}
            save_file(weights,target/"adapter_model.safetensors",metadata={"format":"pt"})
            shutil.copy2(p1/"adapter/adapter_config.json",target)
            cert=read(p1/"DONE.json")
            cert["files"]={str(f.relative_to(target.parent)):sha(f) for f in target.iterdir()}
            cert.update(intervention=label, parent_sha256=sha(p1/"DONE.json"))
            (target.parent/"DONE.json").write_text(json.dumps(cert,indent=2))
        audits[method]={"direction_norm":correction.norm().item(),"modules":audit}
    (out/"A_DIRECTIONS.json").write_text(json.dumps(dict(methods=audits,seconds=time.monotonic()-start,
        peak_bytes=torch.cuda.max_memory_allocated()),indent=2))


def read_rows_file(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


if __name__=="__main__":
    {"freeze":freeze,"build":build}[sys.argv[1]](Path(sys.argv[2]).resolve())
