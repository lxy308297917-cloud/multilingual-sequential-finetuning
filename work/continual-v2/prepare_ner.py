"""Pack four MasakhaNER extension candidates with the frozen official prompt."""

from collections import Counter
import importlib.util
import json
import random
import sys

from common import MODEL, OUTPUTS, ROOT, WORKSPACE, budgets, read, read_rows, save, sha, source

NER = OUTPUTS/"masakhaner2-sequential-v1"
OLD = WORKSPACE/"work/masakhaner2"
CANDIDATES = ("ibo","kin","zul","wol")
LEGACY = ("hau","yor","swa","lug")
SPLITS = ("train","dev","test")
TRAIN_ROWS, DEV_ROWS = 1024,384


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    lib=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    return lib


def get_prepare():
    common=sys.modules["common"]
    try:
        sys.modules["common"]=module("frozen_ner_common",NER/"common.py")
        return module("frozen_ner_prepare",NER/"prepare.py")
    finally:
        sys.modules["common"]=common


def dump_rows(path, rows):
    with path.open("x",encoding="utf-8") as stream:
        for row in rows:
            print(json.dumps(row,ensure_ascii=False),file=stream)
    return sha(path)


def build():
    from transformers import AutoTokenizer

    pack = get_prepare()
    raw = ROOT/"ner-raw"
    commit=read(OLD/"raw/SOURCE.json")["commit"]
    archived=read(OLD/"packed-v1/MANIFEST.json")
    all_rows={}
    upstream={}
    for lang in (*LEGACY,*CANDIDATES):
        for split in SPLITS:
            f=OLD/"raw"/f"data/{lang}/{split}.txt" if lang in LEGACY else raw/f"{lang}/{split}.txt"
            all_rows[lang,split]=pack.read_conll(f,lang,split)
            upstream[f"{lang}/{split}"]=dict(size=f.stat().st_size,sha256=sha(f),n=len(all_rows[lang,split]))
    test_keys={pack.identity(row) for l in (*LEGACY,*CANDIDATES) for row in all_rows[l,"test"]}
    held=test_keys|{pack.identity(row) for l in (*LEGACY,*CANDIDATES) for row in all_rows[l,"dev"]}
    model=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    data=ROOT/"ner-screen"
    data.mkdir()
    files,coverage={},{}
    old_train={}
    for l in LEGACY:
        p=OLD/"packed-v1"/f"train_{l}.jsonl"
        assert sha(p)==archived["files"][p.name]
        old_train[l]={pack.identity(r) for r in read_rows(p)}
    max_prompt=0
    for i,l in enumerate(CANDIDATES):
        rng=random.Random(46+i)
        train=all_rows[l,"train"]
        held_overlap={s: {lang:len(items & {pack.identity(r) for r in all_rows[l,s]}) for lang,items in old_train.items()}
                      for s in ("dev","test")}
        eligible=[row for row in train if pack.identity(row) not in held]
        rng.shuffle(eligible)
        encoded=list(pack.encode(eligible[:TRAIN_ROWS+160],model))
        chosen=[r for r in encoded if len(r["tokens"])<=1536 and len(r["tokens"])-r["answer_start"]<=160][:TRAIN_ROWS]
        dev_raw=[r for r in all_rows[l,"dev"] if pack.identity(r) not in test_keys]
        assert len(chosen)==TRAIN_ROWS and len(dev_raw)>=DEV_ROWS
        dev=list(pack.encode(rng.sample(dev_raw,DEV_ROWS),model))
        test=list(pack.encode(all_rows[l,"test"],model))
        max_prompt=max(max_prompt,max(r["answer_start"] for r in [*dev,*test]))
        types=Counter(span[2] for row in chosen for span in row["spans"])
        assert set(types)=={"PER","LOC","ORG","DATE"}
        for split,rows in {"train":chosen,"dev":dev,"test":test}.items():
            assert len(rows)==len({r["id"] for r in rows})
            n=f"{split}_{l}.jsonl"
            files[n]=dump_rows(data/n,rows)
        coverage[l]=dict(
            train_total=len(train),train_after_overlap=len(eligible),train=budgets(chosen),
            dev_total=len(all_rows[l,"dev"]),dev_used=len(dev),test=len(test),
            historical_training_overlap=held_overlap,
            admissible=not any(count for split in held_overlap.values() for count in split.values()),
            max_encoded_training=max(len(r["tokens"]) for r in chosen),types=types)
    generation=dict(max_model_len=max(1536,((max_prompt+160+255)//256)*256),max_new_tokens=160)
    lock=dict(commit=commit,upstream=upstream,files=files,order=CANDIDATES,coverage=coverage,
              generation=generation,scorer=archived["scorer"],prompt=pack.PROMPT,prefill=pack.PREFILL,
              legacy_manifest_sha256=sha(OLD/"packed-v1/MANIFEST.json"))
    save(data/"MANIFEST.json",lock)
    config=read(NER/"protocol.json")
    plan=dict(benchmark="ner",arm="screen",order=CANDIDATES,seed=42,
              learning_rate=config["learning_rate"],warmup_steps=config["warmup_steps"],
              micro_batch=config["micro_batch"],steps_per_stage=64,max_context=1536,
              protocol=str(NER/"protocol.json"),protocol_sha256=sha(NER/"protocol.json"),**source(data))
    save(ROOT/"plans/ner-screen.json",plan)
    for l,v in coverage.items():
        print(l,v["train_after_overlap"],v["dev_total"],v["test"],v["admissible"],
              v["historical_training_overlap"],flush=True)


if __name__=="__main__":
    build()
