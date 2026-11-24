"""Freeze shared-source MASSIVE 1.1 SFT and full official validation inputs."""

import os as _release_os
import sys as _release_sys
from collections import Counter, defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from itertools import groupby
import hashlib
import json
from pathlib import Path
import random
import tarfile

ROOT = Path(__file__).resolve().parent
PKG = ROOT.parent
UPSTREAM = PKG.parents[1] / "work/followup-upstream/massive"
ARCHIVE = Path(_release_os.path.join(_release_os.environ.get("CL_EXTERNAL_ROOT", "local"), 'datasets/unified-language-cl-20261104/sources/AmazonScience--massive/amazon-massive-dataset-1.1.tar.gz'))
MODEL = _release_os.environ.get('CL_MODEL', 'Qwen3-4B')
LOCALES = ("en-US", "de-DE", "fr-FR", "ar-SA", "vi-VN", "ur-PK", "bn-BD", "th-TH",
           "sw-KE", "am-ET", "jv-ID", "km-KH")
PROMPT = """Identify the intent and extract the slots of the utterance. Return one JSON object with exactly two keys: "intent" and "slots". Use one canonical intent from the list. Each slot is [canonical slot type, exact text copied from the utterance, occurrence]. Occurrence starts at 1 and counts exact word-sequence matches in the utterance from left to right. Occurrence is counted separately for each exact text, never as the slot number. If two different texts each appear once, both slots have occurrence 1. Output the slots in utterance order. Use an empty list when no slot applies.
Example utterance: set an alarm for tomorrow at seven
Example JSON: {"intent":"alarm_set","slots":[["date","tomorrow",1],["time","seven",1]]}
"""


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read_rows(path):
    with path.open() as f:
        return list(map(json.loads,f))


def write_rows(path, data):
    with path.open("x") as f:
        for row in data:
            f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")


def find_matches(tokens, span):
    return [i for i in range(len(tokens)-len(span)+1) if tokens[i:i+len(span)] == span]


def gold_output(row):
    tokens, tags = row["utt"], row["slots_str"]
    assert len(tags) == len(tokens) and tokens
    spans = []
    cursor = 0
    for tag, indices in groupby(tags):
        width = len(list(indices))
        if tag != "Other":
            text = tokens[cursor:cursor+width]
            ordinal = find_matches(tokens,text).index(cursor)+1
            spans.append([tag, " ".join(text), ordinal])
        cursor += width
    return {"intent":row["intent_str"],"slots":spans}


def decode(value, tokens, intent_set, slot_set):
    def object_hook(pairs):
        obj = dict(pairs)
        if len(obj) != len(pairs):
            raise ValueError("Duplicate object key")
        return obj

    invalid_intent = "<invalid>"
    tags = ["Other"]*len(tokens)
    try:
        obj = json.loads(value, object_pairs_hook=object_hook)
    except (TypeError, ValueError, RecursionError):
        return invalid_intent, tags, "invalid_json"
    if not isinstance(obj, dict) or set(obj) != {"intent","slots"}:
        return invalid_intent, tags, "schema"
    if not isinstance(obj["intent"], str) or obj["intent"] not in intent_set or not isinstance(obj["slots"],list):
        return invalid_intent, tags, "schema"
    end = 0
    for item in obj["slots"]:
        if not isinstance(item,list) or len(item) != 3:
            return obj["intent"], ["Other"]*len(tokens), "schema"
        label, text, occurrence = item
        if not isinstance(label,str) or label not in slot_set or not isinstance(text,str) or type(occurrence) is not int:
            return obj["intent"], ["Other"]*len(tokens), "schema"
        words = text.split(" ")
        matches = find_matches(tokens,words) if text else []
        if not 1 <= occurrence <= len(matches) or matches[occurrence-1] < end:
            return obj["intent"], ["Other"]*len(tokens), "span"
        start = matches[occurrence-1]
        end = start+len(words)
        tags[start:end]=[label]*len(words)
    return obj["intent"], tags, None


def split_records():
    spec = spec_from_file_location("upstream_massive_dataset",UPSTREAM/"scripts/create_hf_dataset.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    datasets = {}
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            locale = Path(member.name).stem
            if locale not in LOCALES or not member.name.endswith(".jsonl"):
                continue
            with archive.extractfile(member) as stream:
                all_rows = [json.loads(line) for line in stream]
            partitions = Counter(row["partition"] for row in all_rows)
            assert {k:partitions[k] for k in ("train","dev","test")} == dict(train=11514,dev=2033,test=2974)
            training, validation, _, _ = module.DatasetCreator._build_in_mem_dicts(
                [row for row in all_rows if row["partition"] in ("train","dev")])
            datasets[locale] = {
                split: [dict(zip(partition,values,strict=True)) for values in zip(*partition.values(),strict=True)]
                for split, partition in (("train",training),("dev",validation))
            }
    assert set(datasets) == set(LOCALES)
    return datasets


def prepare():
    from transformers import AutoTokenizer

    destination = ROOT/"data"
    destination.mkdir()
    datasets = split_records()
    canonical = {row["id"]:row["intent_str"] for row in datasets["en-US"]["train"]}
    validation_ids = {row["id"]:row["intent_str"] for row in datasets["en-US"]["dev"]}
    assert not canonical.keys() & validation_ids.keys()
    intents = set(canonical.values())
    slots = {tag for locale in datasets.values() for row in locale["train"]
             for tag in row["slots_str"] if tag != "Other"}
    assert len(intents)==60 and len(slots)==55, (len(intents),len(slots))
    by_intent = defaultdict(list)
    for item,label in canonical.items():
        by_intent[label].append(item)
    rng = random.Random(42)
    selected = []
    remainder = []
    for label, ids in sorted(by_intent.items()):
        ids.sort()
        rng.shuffle(ids)
        count = max(1,len(ids)*2000//len(canonical))
        selected.extend(ids[:count])
        remainder.extend(ids[count:])
    rng.shuffle(remainder)
    selected.extend(remainder[:2000-len(selected)])
    assert len(set(selected)) == 2000
    rng.shuffle(selected)
    tok = AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    prefix = PROMPT+"Canonical intents: "+", ".join(sorted(intents))+"\nCanonical slot types: "+", ".join(sorted(slots))+"\nUtterance: "
    info = {}
    for language in LOCALES:
        parts = datasets[language]
        assert {row["id"]:row["intent_str"] for row in parts["train"]} == canonical
        assert {row["id"]:row["intent_str"] for row in parts["dev"]} == validation_ids
        by_id = {r["id"]:r for r in parts["train"]}
        counts = {}
        for split, cohort in (("train",[by_id[i] for i in selected]),("dev",parts["dev"])):
            content = []
            for row in cohort:
                text = prefix+" ".join(row["utt"])
                prompt = tok.apply_chat_template([{"role":"user","content":text}],tokenize=False,
                                                 add_generation_prompt=True,enable_thinking=False)+"JSON:"
                target = gold_output(row)
                answer = " " + json.dumps(target,ensure_ascii=False,separators=(",",":"))
                intent, tags, error = decode(answer,row["utt"],intents,slots)
                assert not error and intent == row["intent_str"] and tags == row["slots_str"]
                content.append(dict(id=row["id"],utt=row["utt"],tags=tags,intent=intent,prompt=prompt,answer=answer))
            input_ids = tok([r["prompt"] for r in content],add_special_tokens=False)["input_ids"]
            full_ids = tok([r["prompt"]+r["answer"] for r in content],add_special_tokens=False)["input_ids"]
            counts[split] = dict(samples=len(content),input_tokens=0,supervised_tokens=0,
                                 max_prompt=0,max_response=0,intents=len({r["intent"] for r in content}),
                                 slot_types=len({t for r in content for t in r["tags"] if t!="Other"}))
            for row, pids, ids in zip(content,input_ids,full_ids,strict=True):
                assert ids[:len(pids)] == pids
                row["tokens"] = ids+[tok.eos_token_id]
                row["answer_start"] = len(pids)
                stat=counts[split]
                stat["input_tokens"]+=len(row["tokens"])
                stat["supervised_tokens"]+=len(row["tokens"])-len(pids)
                stat["max_prompt"]=max(stat["max_prompt"],len(pids))
                stat["max_response"]=max(stat["max_response"],len(row["tokens"])-len(pids))
                assert len(pids)+192<=1536 and stat["max_response"]<=192
            write_rows(destination/f"{split}_{language}.jsonl",content)
        info[language]=counts
        print(language,counts,flush=True)
    frozen = dict(source=str(ARCHIVE),sha256=sha(ARCHIVE),locales=LOCALES,train_ids=selected,
                  intents=sorted(intents),slots=sorted(slots),data=info, prompt=prefix,
                  upstream_commit="f966f21846043aabef9b0f974fa7970027f43738",
                  files={file.name:sha(file) for file in sorted(destination.glob("*.jsonl"))})
    (destination/"MANIFEST.json").write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    prepare()
