"""Freeze stronger-to-weaker courses before any V2 official-test experiments."""

import os as _release_os
import sys as _release_sys

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import shutil

from common import CKPT, HISTORICAL, LOGS, MTOP, OUTPUTS, ROOT, WORKSPACE, budgets, dataset, read, save, schedule, sha, source

OLD = OUTPUTS/"continual-bench-next"
NER = OUTPUTS/"masakhaner2-sequential-v1"
ORIGINAL = {
    "mtop": WORKSPACE/"work/mtop-audit/packed-v1",
    "ner": WORKSPACE/"work/masakhaner2/packed-v1",
}
BASE = {"mtop":"MTOP","ner":"MasakhaNER 2.0"}
PRIOR_CKPT = {
    "mtop": Path(_release_os.path.join(_release_os.environ.get("CL_EXTERNAL_ROOT", "local"), 'checkpoints/mtop_flat_qwen3_recovery_v1_20261122')),
    "ner": Path(_release_os.path.join(_release_os.environ.get("CL_EXTERNAL_ROOT", "local"), 'checkpoints/masakhaner2_sequential_v1_20261121')),
}


def row_hash(path, rows):
    with path.open("x",encoding="utf-8") as stream:
        for row in rows:
            print(json.dumps(row,ensure_ascii=False),file=stream)
    return sha(path)


def replay_schedules(order, selected):
    rng=random.Random(42+470)
    memory={}
    seen=set()
    for i,lang in enumerate(order):
        current=selected[lang]
        packed=list(current)
        changes={}
        old_tokens={}
        old_labels=order[:i]
        capacity={}
        if old_labels:
            target=round(budgets(current)["input_tokens"]*.10)
            cap,extra=divmod(1000,i)
            quota,remainder=divmod(target,i)
            positions=list(range(len(current)))
            rng.shuffle(positions)
            for j,old in enumerate(old_labels):
                memory[old]=memory[old][:cap+int(j<extra)]
                capacity[old]=len(memory[old])
                options=defaultdict(list)
                for row in memory[old]:
                    options[(len(row["tokens"]),row["answer_start"])].append(row)
                goal=quota+int(j<remainder)
                allocated=0
                for pos in positions:
                    row=current[pos]
                    key=(len(row["tokens"]),row["answer_start"])
                    if pos in changes or not options[key]:
                        continue
                    if abs(goal-allocated-key[0])>=abs(goal-allocated):
                        continue
                    replacement=rng.choice(options[key])
                    assert replacement["id"] in seen and replacement["locale"]==old
                    changes[pos]=dict(old=old,source=replacement["id"],displaced=row["id"])
                    packed[pos]=replacement
                    allocated+=key[0]
                    if allocated>=goal:
                        break
                old_tokens[old]=allocated
            assert sum(capacity.values())==1000
        fraction=sum(old_tokens.values())/budgets(current)["input_tokens"]
        if old_labels:
            assert abs(fraction-.10)<=.002,(lang,fraction,old_tokens)
        assert budgets(current)==budgets(packed)
        for original,replacement in zip(current,packed,strict=True):
            assert len(original["tokens"])==len(replacement["tokens"]) and original["answer_start"]==replacement["answer_start"]
        this_round=[r for p,r in enumerate(current) if p not in changes]
        assert all(r["locale"]==lang for r in this_round)
        memory[lang]=rng.sample(this_round,min(1000,len(this_round)))
        seen.update(row["id"] for row in this_round)
        yield lang,packed,dict(input_fraction=fraction,old_input_tokens=old_tokens,capacity=capacity,
                               replacements=len(changes),old_supervised_tokens=sum(len(packed[p]["tokens"])-packed[p]["answer_start"] for p in changes))


def score_identity(path,split,lang,expected_hash):
    cert=read(path)
    assert cert["split"]==split and cert["locale"]==lang
    assert cert["input_sha256"]==expected_hash and sha(path.with_name(f"{lang}.jsonl"))==cert["prediction_sha256"]
    return dict(path=str(path),certificate_sha256=sha(path),f1=cert["f1"],n=cert["n"],valid=cert["valid"])


def make_task(task):
    config_path=(OLD/"mtop/protocol.json") if task=="mtop" else (NER/"protocol.json")
    config=read(config_path)
    archived=read(OLD/"BASELINE_COMPARISONS.json")["controls"]
    base_dev=archived[BASE[task]+" · dev"]["Single LoRA"]
    origins={lang:ORIGINAL[task] for lang in base_dev}
    singles={lang:PRIOR_CKPT[task]/"single"/lang/"step-064" for lang in base_dev}
    candidates={}
    screen_base=None
    if task=="ner":
        screen=ROOT/"ner-screen"
        frozen=read(screen/"MANIFEST.json")
        assert (LOGS/"ner-screen/DONE.json").exists()
        for lang,m in frozen["coverage"].items():
            pair=LOGS/"ner-screen"/f"single_{lang}"/"dev"/f"{lang}.DONE.json"
            s=score_identity(pair,"dev",lang,frozen["files"][f"dev_{lang}.jsonl"])
            candidates[lang]=dict(**s,admissible=m["admissible"])
        winner=max((lang for lang,x in candidates.items() if x["admissible"]),
                   key=lambda lang:(candidates[lang]["f1"],lang))
        singles[winner]=CKPT/"ner-screen"/"single"/winner/"step-064"
        origins[winner]=screen
        base_dev={**base_dev,winner:{"receipt":candidates[winner]["path"],"score":candidates[winner]["f1"]}}
        screen_base=LOGS/"ner-screen"/"base"/"dev"/f"{winner}.DONE.json"
    order=tuple(sorted(origins,key=lambda l:(-base_dev[l]["score"],l)))
    if task=="mtop":
        assert order==MTOP
    data=ROOT/f"{task}-course"
    data.mkdir()
    files,counts={},{}
    chosen={}
    datasets={}
    for lang in order:
        before=origins[lang]
        m=read(before/"MANIFEST.json")
        datasets[lang]=sha(before/"MANIFEST.json")
        for split in ("train","dev","test"):
            key=f"{split}_{lang}.jsonl"
            assert sha(before/key)==m["files"][key]
            shutil.copyfile(before/key,data/key)
            files[key]=sha(data/key)
            packed=schedule(data,dict(files=files),key)
            counts[key]=len(packed)
            assert len(packed)==len({r["id"] for r in packed})
            if split=="train":
                assert len(packed)==1024 and all(r["locale"]==lang for r in packed)
                chosen[lang]=packed
    joint=[row for lang in order for row in chosen[lang]]
    random.Random(42+127).shuffle(joint)
    counts["joint.jsonl"]=len(joint)
    files["joint.jsonl"]=row_hash(data/"joint.jsonl",joint)
    sequential,replay,replay_stats=[],[],{}
    for i,(lang,rows,stats) in enumerate(replay_schedules(order,chosen),1):
        name=f"{i:02}_{lang}"
        current=f"train_{lang}.jsonl"
        key=f"replay_{lang}.jsonl" if i>1 else current
        if i>1:
            files[key]=row_hash(data/key,rows)
            counts[key]=len(rows)
        sequential.append(dict(name=name,file=current,**budgets(chosen[lang])))
        replay.append(dict(name=name,file=key,**budgets(rows)))
        replay_stats[lang]=stats
    ref=read(ORIGINAL[task]/"MANIFEST.json")
    merged=dict(ref,order=order,files=files,counts=counts,dataset_sources=datasets,replay_audit=replay_stats)
    if task=="ner":
        merged["generation"]=dict(ref["generation"])
        if any(origins[l]!=ORIGINAL[task] for l in order):
            m=read((ROOT/"ner-screen/MANIFEST.json"))
            merged["generation"]["max_model_len"]=max(m["generation"]["max_model_len"],ref["generation"]["max_model_len"])
    save(data/"MANIFEST.json",merged)
    matched={}
    for lang in order:
        ref_dir=singles[lang]
        cert=read(ref_dir/"DONE.json")
        assert cert["complete_stage"] and not cert["smoke"] and cert["global_step"]==64 and cert["start_step"]==0
        assert cert["seed"]==config["seed"] and cert["learning_rate"]==config["learning_rate"]
        assert cert["schedule_sha256"]==files[f"train_{lang}.jsonl"] and cert["protocol_sha256"]==sha(config_path)
        assert all(sha(ref_dir/name)==digest for name,digest in cert["files"].items())
        matched[lang]=dict(path=str(ref_dir),certificate_sha256=sha(ref_dir/"DONE.json"),step=64,
                           score=score_identity(Path(base_dev[lang]["receipt"]),"dev",lang,files[f"dev_{lang}.jsonl"]))
    plan=dict(benchmark=task, arm="course-v2",order=order,seed=config["seed"],learning_rate=config["learning_rate"],
              warmup_steps=config["warmup_steps"],micro_batch=config["micro_batch"],steps_per_stage=64,
              max_context=2048 if task=="mtop" else 1536,protocol=str(config_path),protocol_sha256=sha(config_path),
              matched_lora=matched,candidate_single_scores=candidates,sequential=sequential,replay=replay,
              single={l:budgets(chosen[l]) for l in order},joint=budgets(joint),**source(data))
    if screen_base:
        plan["candidate_base_dev"]=score_identity(screen_base,"dev",winner,files[f"dev_{winner}.jsonl"])
    save(ROOT/f"plans/{task}-course.json",plan)
    print(json.dumps(dict(order=order,dev_single={l:matched[l]["score"]["f1"] for l in order},
                          replay={l:x["input_fraction"] for l,x in replay_stats.items()},joint=plan["joint"]),indent=2))


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("task",choices=("mtop","ner"))
    make_task(p.parse_args().task)
