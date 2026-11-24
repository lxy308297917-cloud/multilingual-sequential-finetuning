"""Strict historical replay experiment with a shared UR stream."""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"massive-cl-v1"))
from common import CFG, DATA, LANGUAGES, MODEL, PREVIOUS, load, manifest, read, sha, write_rows
from prepare import budget, encode, match_key, source_records

ARMS=("random","label","parallel")
UR=LANGUAGES[-1]


def signature(row):
    return row["intent"],tuple(sorted(slot[0] for slot in json.loads(row["answer"])["slots"]))


def prepare(out):
    from transformers import AutoTokenizer
    out.mkdir()
    rng=random.Random(42)
    receipt=manifest()
    historical={lang:load(f"train_{lang}.jsonl",receipt) for lang in LANGUAGES}
    seen={r["id"] for rows in historical.values() for r in rows}
    memory={lang:rng.sample(historical[lang],250) for lang in LANGUAGES[:-1]}
    by_id={r["id"]:r for rows in memory.values() for r in rows}
    archive=source_records()
    raw={r["id"]:r for r in archive[UR]["train"]}
    ids=list(by_id)+rng.sample(sorted(set(raw)-seen),1000)
    rng.shuffle(ids)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    current=encode(tokenizer,[raw[i] for i in ids],read(PREVIOUS/"data/MANIFEST.json"),UR)
    assert {r["id"] for r in current}==set(ids)
    forbidden={r["id"] for sets in archive.values() for part in ("dev","test") for r in sets[part]}
    assert not set(ids)&forbidden
    assert sum(r["id"] in by_id for r in current)==1000
    available={}
    for lang,rows in memory.items():
        buckets=defaultdict(list)
        for row in rows:
            buckets[match_key(row)].append(row)
        available[lang]=buckets
    width=CFG["effective_batch"]
    positions=list(range(len(current)))
    rng.shuffle(positions)
    old_tokens=Counter()
    replacement={}
    keep=set()
    actual_pair=Counter()
    target=budget(current)["input_tokens"]*.10/4
    # Exact input and answer token matching for all arms and the removed row.
    for position in positions:
        batch=position//width*width
        local=range(batch,batch+width)
        pairs=[(i,by_id[current[i]["id"]]) for i in local
               if current[i]["id"] in by_id and i!=position]
        rng.shuffle(pairs)
        if position in keep:
            continue
        for i,paired in pairs:
            lang=paired["locale"]
            if i in replacement:
                continue
            if old_tokens[lang]+len(paired["tokens"])>target:
                continue
            in_batch={current[j]["id"] for j in local}
            random_pool=[r for r in available[lang][match_key(paired)] if r["id"] not in in_batch]
            label_pool=[r for r in random_pool if signature(r)==signature(paired)]
            if not label_pool:
                continue
            replacement[position]=(rng.choice(random_pool),rng.choice(label_pool),paired)
            keep.add(i)
            old_tokens[lang]+=len(paired["tokens"])
            actual_pair[lang]+=1
            break
    for position in positions:
        if position in replacement or position in keep:
            continue
        langs=sorted(memory,key=lambda lang:old_tokens[lang])
        step=position//width*width
        in_batch={r["id"] for r in current[step:step+width]}
        for lang in langs:
            candidates=[r for r in available[lang][match_key(current[position])] if r["id"] not in in_batch]
            if not candidates or old_tokens[lang]+len(current[position]["tokens"])>target:
                continue
            row=rng.choice(candidates)
            replacement[position]=(row,row,row)
            old_tokens[lang]+=len(row["tokens"])
            break
    fraction=sum(old_tokens.values())/budget(current)["input_tokens"]
    assert .097<=fraction<=.10, (fraction,old_tokens)
    assert all(actual_pair.values()) and len(actual_pair)==4,actual_pair
    write_rows(out/"ur_stream.jsonl",current)
    schedules={}
    for arm_index,arm in enumerate(ARMS):
        rows=list(current)
        for position,choices in replacement.items():
            rows[position]=choices[arm_index]
        assert len(rows)==len(current)
        p=out/f"{arm}.jsonl"
        write_rows(p,rows)
        schedules[p.name]=sha(p)
        if arm_index==0:
            matched_budget=budget(rows)
        assert budget(rows)==matched_budget
    audit={str(i):{a:r["id"] for a,r in zip(ARMS,picks)} for i,picks in replacement.items()}
    (out/"B_LOCK.json").write_text(json.dumps(dict(seed=42, budget=matched_budget, fraction=sum(old_tokens.values())/matched_budget["input_tokens"],
        old_tokens=old_tokens, parallel_counts=actual_pair, replacements=audit, schedules=schedules,
        memory={k:[r["id"] for r in v] for k,v in memory.items()},
        ur_ids=[r["id"] for r in current], protected_positions=sorted(keep),
        manifest_sha256=sha(DATA/"MANIFEST.json"),source_sha256=sha(__file__)),indent=2))
    print("B lock", fraction,old_tokens,actual_pair,flush=True)


def train_arm(out, arm, target):
    import train
    from common import CHECKPOINTS, read_rows
    lock=read(out/"B_LOCK.json")
    assert lock["manifest_sha256"]==sha(DATA/"MANIFEST.json")
    p=out/f"{arm}.jsonl"
    assert sha(p)==lock["schedules"][p.name]
    receipt=manifest()
    train.manifest=lambda:dict(receipt,files={**receipt["files"],p.name:sha(p)})
    train.load=lambda name,info:read_rows(p)
    train.CFG=dict(train.CFG,progress_steps=[125])
    parent=CHECKPOINTS/"seq/04_sw-KE/step-125"
    sys.argv=["train.py",p.name,target,"--continue-from",str(parent),"--trajectory","seq"]
    train.main()


if __name__=="__main__":
    mode,out,*other=sys.argv[1:]
    out=Path(out).resolve()
    if mode=="prepare":
        prepare(out)
    else:
        train_arm(out,*other)
