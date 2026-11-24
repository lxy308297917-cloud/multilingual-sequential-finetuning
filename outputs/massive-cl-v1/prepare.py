"""Select disjoint MASSIVE training IDs and match Replay token budgets exactly."""
from collections import Counter, defaultdict
from importlib.util import module_from_spec, spec_from_file_location
import hashlib
import json
import random
import subprocess
import tarfile

from common import (ARCHIVE, CFG, DATA, EVAL, LANGUAGES, MODEL, PREVIOUS, PROTOCOL, UPSTREAM,
                    atomic_json, decode, gold_output, read, read_rows, sha, write_rows)


def source_records():
    spec = spec_from_file_location("upstream_massive_dataset", UPSTREAM / "scripts/create_hf_dataset.py")
    upstream = module_from_spec(spec)
    spec.loader.exec_module(upstream)
    datasets = {}
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            locale = member.name.rsplit("/", 1)[-1].removesuffix(".jsonl")
            if locale not in EVAL or not member.name.endswith(".jsonl"):
                continue
            with archive.extractfile(member) as handle:
                rows = list(map(json.loads, handle))
            parts = upstream.DatasetCreator._build_in_mem_dicts(
                [row for row in rows if row["partition"] in ("train", "dev", "test")])[:3]
            datasets[locale] = {}
            for split, part, expected in zip(("train", "dev", "test"), parts, (11514, 2033, 2974), strict=True):
                entries = [dict(zip(part, values, strict=True)) for values in zip(*part.values(), strict=True)]
                assert len(entries) == expected and len({r["id"] for r in entries}) == expected
                datasets[locale][split] = entries
    assert set(datasets) == set(EVAL)
    return datasets


def assign_ids(by_intent):
    rng = random.Random(CFG["seed"])
    size = CFG["train_rows_per_language"]
    selected = {language: [] for language in LANGUAGES}
    count = sum(map(len, by_intent.values()))
    tail = []
    for label, ids in sorted(by_intent.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        width = min(len(ids) // len(LANGUAGES), max(1, len(ids) * size // count))
        if width == 0:
            recipients = rng.sample(list(LANGUAGES), len(ids))
            for lang, sid in zip(recipients, ids, strict=True):
                selected[lang].append(sid)
            continue
        for i, locale in enumerate(LANGUAGES):
            selected[locale].extend(ids[i * width:(i + 1) * width])
        tail.extend(ids[len(LANGUAGES) * width:])
    rng.shuffle(tail)
    for lang in LANGUAGES:
        n_missing = size - len(selected[lang])
        assert n_missing >= 0
        selected[lang].extend(tail[:n_missing])
        del tail[:n_missing]
        assert len(set(selected[lang])) == size
        rng.shuffle(selected[lang])
    assert len({i for ids in selected.values() for i in ids}) == size * len(LANGUAGES)
    return selected


def encode(tokenizer, incoming, previous, locale):
    intents, slots = set(previous["intents"]), set(previous["slots"])
    batch = []
    for record in incoming:
        text = previous["prompt"] + " ".join(record["utt"])
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False) + "JSON:"
        answer = " " + json.dumps(gold_output(record), ensure_ascii=False, separators=(",", ":"))
        result = decode(answer, record["utt"], intents, slots)
        assert result == (record["intent_str"], record["slots_str"], None)
        batch.append(dict(id=record["id"], utt=record["utt"], tags=record["slots_str"],
                          intent=record["intent_str"], locale=locale, prompt=prompt, answer=answer))
    inputs = tokenizer([r["prompt"] for r in batch], add_special_tokens=False)["input_ids"]
    combined = tokenizer([r["prompt"] + r["answer"] for r in batch], add_special_tokens=False)["input_ids"]
    for row, prompt, full in zip(batch, inputs, combined, strict=True):
        assert full[:len(prompt)] == prompt
        row.update(tokens=full + [tokenizer.eos_token_id], answer_start=len(prompt))
        assert len(prompt) + 192 <= 1536 and len(full) + 1 - len(prompt) <= 192
    return batch


def budget(rows):
    return dict(rows=len(rows), input_tokens=sum(len(r["tokens"]) for r in rows),
                supervised_tokens=sum(len(r["tokens"]) - r["answer_start"] for r in rows))


def sample_memory(rows, n):
    return sorted(rows, key=lambda r: hashlib.sha256(f"42/memory/{r['id']}".encode()).hexdigest())[:n]


def match_key(record):
    return len(record["tokens"]), record["answer_start"]


def replay_stages(selected):
    rng = random.Random(CFG["seed"] + 470)
    memory = {}
    audits = {}
    for i, lang in enumerate(LANGUAGES):
        current = selected[lang]
        mixed = list(current)
        replaced = {}
        exposures = Counter()
        buffers = {}
        if i:
            previously_seen = LANGUAGES[:i]
            target = round(budget(current)["input_tokens"] * CFG["replay_input_fraction"])
            positions = list(range(len(current)))
            rng.shuffle(positions)
            for j, old in enumerate(previously_seen):
                capacity, extra = divmod(CFG["replay_buffer"], i)
                memory[old] = memory[old][:capacity + int(j < extra)]
                buffers[old] = [row["id"] for row in memory[old]]
                options = defaultdict(list)
                for row in memory[old]:
                    options[match_key(row)].append(row)
                quota, remainder = divmod(target, i)
                goal = quota + int(j < remainder)
                for pos in positions:
                    key = match_key(current[pos])
                    if pos in replaced or not options[key]:
                        continue
                    if abs(goal - exposures[old] - key[0]) >= abs(goal - exposures[old]):
                        continue
                    choice = rng.choice(options[key])
                    mixed[pos] = choice
                    replaced[pos] = {"removed": current[pos]["id"], "replay": choice["id"], "locale": old}
                    exposures[old] += key[0]
                    if exposures[old] >= goal:
                        break
                assert abs(exposures[old] - goal) <= 700, (old, goal, exposures[old])
            assert sum(map(len, buffers.values())) == CFG["replay_buffer"]
            frac = sum(exposures.values()) / budget(current)["input_tokens"]
            assert abs(frac - CFG["replay_input_fraction"]) <= CFG["replay_max_fraction_error"]
        else:
            frac = 0.0
        assert budget(mixed) == budget(current)
        assert all(match_key(a) == match_key(b) for a, b in zip(current, mixed, strict=True))
        write_rows(DATA / f"replay_{lang}.jsonl", mixed)
        audits[lang] = dict(input_fraction=frac, input_tokens_by_old_language=dict(exposures),
                             buffer_ids=buffers, replacements=replaced, budget=budget(mixed),
                             old_supervised_tokens=sum(len(mixed[p]["tokens"]) - mixed[p]["answer_start"] for p in replaced))
        memory[lang] = sample_memory([r for p, r in enumerate(current) if p not in replaced], CFG["replay_buffer"])
    return audits


def main():
    from transformers import AutoTokenizer
    DATA.mkdir()
    previous_manifest = PREVIOUS / "data/MANIFEST.json"
    previous = read(previous_manifest)
    assert sha(ARCHIVE) == previous["sha256"]
    head = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    assert head == previous["upstream_commit"]
    datasets = source_records()
    reference = {split: {r["id"]: r["intent_str"] for r in data}
                 for split, data in datasets["en-US"].items()}
    assert sum(map(len, reference.values())) == len(set().union(*[set(v) for v in reference.values()]))
    for splits in datasets.values():
        for split, rows in splits.items():
            assert {r["id"]: r["intent_str"] for r in rows} == reference[split]
    intent_ids = defaultdict(list)
    for sid, label in reference["train"].items():
        intent_ids[label].append(sid)
    ids = assign_ids(intent_ids)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    enc = {}
    summaries = {}
    for lang in EVAL:
        prior_dev = PREVIOUS / "data" / f"dev_{lang}.jsonl"
        assert sha(prior_dev) == previous["files"][prior_dev.name]
        earlier = read_rows(prior_dev)
        assert [r["id"] for r in earlier] == [r["id"] for r in datasets[lang]["dev"]]
        (DATA / prior_dev.name).symlink_to(prior_dev)
        test = encode(tokenizer, datasets[lang]["test"], previous, lang)
        write_rows(DATA / f"test_{lang}.jsonl", test)
        if lang not in LANGUAGES:
            continue
        idx = {r["id"]: r for r in datasets[lang]["train"]}
        enc[lang] = encode(tokenizer, [idx[sid] for sid in ids[lang]], previous, lang)
        write_rows(DATA / f"train_{lang}.jsonl", enc[lang])
        coverage = Counter(tag for row in enc[lang] for tag in row["tags"] if tag != "Other")
        summaries[lang] = dict(budget(enc[lang]), intents=dict(Counter(r["intent"] for r in enc[lang])),
                               slot_token_counts=dict(coverage), missing_slot_types=sorted(set(previous["slots"]) - set(coverage)))
        summaries[lang]["missing_intents"] = sorted(set(previous["intents"]) - set(summaries[lang]["intents"]))
        print(lang, budget(enc[lang]), "slot_types", len(coverage), flush=True)
    audits = replay_stages(enc)
    joint = [row for lang in LANGUAGES for row in enc[lang]]
    random.Random(CFG["seed"] + 480).shuffle(joint)
    write_rows(DATA / "joint.jsonl", joint)
    model = read(PREVIOUS.parents[1] / "sequential-lora/data/MANIFEST.json")["model"]
    artifact = dict(protocol_sha256=sha(PROTOCOL), prior_manifest_sha256=sha(previous_manifest),
                    upstream_commit=head, archive=str(ARCHIVE), archive_sha256=previous["sha256"],
                    prompt=previous["prompt"], slots=previous["slots"], intents=previous["intents"],
                    model=model, train_ids=ids, train=summaries, joint=budget(joint), replay=audits,
                    files={item.name: sha(item) for item in sorted(DATA.glob("*.jsonl"))})
    atomic_json(DATA / "MANIFEST.json", artifact)
    print("Replay input proportions", {lang: r["input_fraction"] for lang, r in audits.items()}, flush=True)


if __name__ == "__main__":
    main()
