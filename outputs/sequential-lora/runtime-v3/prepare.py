import argparse
import hashlib
import json
import random
import re
import tarfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from common import CFG, LABELS, LANGUAGES, MODEL, ROOT, prompt, sha, tokenizer, write_json, write_rows

SOURCE = Path(CFG["source_root"])
TOKENIZER_CONTROL = re.compile(r"<\|[^\r\n]*?\|>")


def normalize(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.split())


def premise_group(value):
    return hashlib.sha256(normalize(value).encode()).hexdigest()


def parquet_rows(paths):
    offset = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        features = json.loads(parquet.schema_arrow.metadata[b"huggingface"])["info"]["features"]
        if tuple(features["label"]["names"]) != LABELS:
            raise ValueError(f"Unexpected label mapping in {path}")
        for batch in parquet.iter_batches(batch_size=4096, columns=["premise", "hypothesis", "label"]):
            for row in batch.to_pylist():
                hypo = dict(zip(row["hypothesis"]["language"], row["hypothesis"]["translation"], strict=True))
                yield offset, row["label"], row["premise"], hypo
                offset += 1


def valid(premise, hypothesis):
    return bool(premise.strip() and hypothesis.strip()) and not (
        TOKENIZER_CONTROL.search(premise) or TOKENIZER_CONTROL.search(hypothesis)
    )


def read_eval(paths, data, tok):
    forbidden = {lang: set() for lang in LANGUAGES}
    splits = {}
    label_lengths = {label: len(tok.encode(CFG["label_delimiter"] + label, add_special_tokens=False))
                     for label in LABELS}
    for split, expected in (("validation", 2490), ("test", 5010)):
        by_lang = {lang: [] for lang in LANGUAGES}
        for index, label, premises, hypotheses in parquet_rows([paths[split]]):
            group = premise_group(premises["en"])
            for lang in LANGUAGES:
                p, h = premises[lang], hypotheses[lang]
                if not valid(p, h):
                    raise ValueError(f"Invalid official XNLI row {split}/{index}/{lang}")
                context = prompt(tok, p, h)
                prefix_ids = tok.encode(context, add_special_tokens=False)
                for answer, length in label_lengths.items():
                    whole = tok.encode(context + CFG["label_delimiter"] + answer, add_special_tokens=False)
                    if whole[:len(prefix_ids)] != prefix_ids or len(whole) - len(prefix_ids) != length:
                        raise ValueError(f"Candidate boundary mismatch for {split}/{index}/{lang}/{answer}")
                if len(prefix_ids) + max(label_lengths.values()) > CFG["max_length"]:
                    raise ValueError(f"Official row exceeds context budget: {split}/{index}/{lang}")
                by_lang[lang].append(dict(id=index, premise_group=group, label=label, prompt=context))
                forbidden[lang].add(normalize(p))
        for lang, dataset in by_lang.items():
            if len(dataset) != expected or Counter(row["label"] for row in dataset) != {
                i: expected // 3 for i in range(len(LABELS))
            }:
                raise ValueError(f"Unexpected official XNLI split {split}/{lang}")
            write_rows(data / f"{split}_{lang}.jsonl", dataset)
        splits[split] = by_lang
    return forbidden, splits, label_lengths


def select_train(paths, tok, forbidden):
    seed = CFG["seed"]
    pools = defaultdict(list)
    discarded = Counter()
    for index, label, premises, hypotheses in parquet_rows(paths):
        p_en, h_en = premises["en"], hypotheses["en"]
        normalized_en = normalize(p_en)
        group = premise_group(p_en)
        bucket = int.from_bytes(hashlib.blake2b(f"{seed}/{group}".encode(), digest_size=8).digest(), "big")
        lang = LANGUAGES[bucket % len(LANGUAGES)]
        p, h = premises[lang], hypotheses[lang]
        if not valid(p_en, h_en) or not valid(p, h):
            discarded["invalid"] += 1
        elif normalized_en in forbidden["en"] or normalize(p) in forbidden[lang]:
            discarded["official_premise_overlap"] += 1
        else:
            pools[lang, label].append((index, group, p, h, normalize(h_en)))
    count = CFG["train_examples_per_language"]
    selected = {}
    for lang_index, lang in enumerate(LANGUAGES):
        by_lang = []
        pairs = set()
        for label in range(len(LABELS)):
            # Rotate the single residual example to keep both per-language and global ratios balanced.
            quota = count // len(LABELS) + int((label - lang_index) % len(LABELS) < count % len(LABELS))
            pool = pools[lang, label]
            random.Random(seed + 101 * lang_index + label).shuffle(pool)
            n = 0
            for index, group, p, h, h_en in pool:
                pair = group, h_en
                if pair in pairs:
                    discarded["repeated_source_pair_encountered"] += 1
                    continue
                context = prompt(tok, p, h)
                answer = CFG["label_delimiter"] + LABELS[label]
                context_ids = tok.encode(context, add_special_tokens=False)
                full = tok.encode(context + answer, add_special_tokens=False) + [tok.eos_token_id]
                if full[:len(context_ids)] != context_ids or len(full) <= len(context_ids) + 1:
                    raise ValueError(f"Candidate token boundary moves at training row {index}/{lang}")
                if len(full) > CFG["max_length"]:
                    discarded["overlength_encountered"] += 1
                    continue
                pairs.add(pair)
                by_lang.append(dict(id=index, lang=lang, label=label, group=group,
                                    tokens=full, answer_start=len(context_ids)))
                n += 1
                if n == quota:
                    break
            if n != quota:
                raise ValueError(f"Insufficient training rows in {lang}/{label}: {n}/{quota}")
        random.Random(seed + 2000 + lang_index).shuffle(by_lang)
        selected[lang] = by_lang
    n_ids = len({row["id"] for dataset in selected.values() for row in dataset})
    group_to_lang = {(row["group"], row["lang"]) for dataset in selected.values() for row in dataset}
    if n_ids != count * len(LANGUAGES) or len(group_to_lang) != len({group for group, _ in group_to_lang}):
        raise ValueError("Selected IDs or English premise groups overlap between languages")
    return selected, discarded


def replay_schedule(selected, directory):
    rng = random.Random(CFG["seed"] + 4000)
    summary = {}
    memory = {}

    def rank_seen(records):
        ranked = sorted(
            records, key=lambda row: hashlib.sha256(f"{CFG['seed']}/memory/{row['id']}".encode()).hexdigest(),
        )
        return ranked[:CFG["replay_buffer"]]

    for stage, lang in enumerate(LANGUAGES):
        current = selected[lang]
        if stage == 0:
            memory[lang] = rank_seen(current)
            continue
        previous = LANGUAGES[:stage]
        keys = [(len(row["tokens"]), row["label"], row["answer_start"]) for row in current]
        available = defaultdict(list)
        for idx, key in enumerate(keys):
            available[key].append(idx)
        for indices in available.values():
            rng.shuffle(indices)
        targets = round(sum(key[0] for key in keys) * CFG["replay_input_ratio"])
        replacements = {}
        used = {}
        budgets = Counter()
        buffers = {}
        for i, old in enumerate(previous):
            buffer_quota = CFG["replay_buffer"] // stage + int(i < CFG["replay_buffer"] % stage)
            buffer_rows = memory[old][:buffer_quota]
            memory[old] = buffer_rows
            buffers[old] = [row["id"] for row in buffer_rows]
            matched = defaultdict(list)
            for row in buffer_rows:
                key = len(row["tokens"]), row["label"], row["answer_start"]
                if available[key]:
                    matched[key].append(row)
            positions = [idx for key in matched for idx in available[key]]
            rng.shuffle(positions)
            goal = targets // stage + int(i < targets % stage)
            for idx in positions:
                key = keys[idx]
                if budgets[old] and abs(budgets[old] - goal) < abs(budgets[old] + key[0] - goal):
                    continue
                replacement = rng.choice(matched[key])
                replacements[idx] = replacement["id"]
                available[key].remove(idx)
                budgets[old] += key[0]
                used.setdefault(old, Counter())[replacement["id"]] += 1
                if budgets[old] >= goal:
                    break
            if abs(budgets[old] - goal) > CFG["max_length"]:
                raise ValueError(f"Could not match replay token quota for {lang}/{old}")
        memory[lang] = rank_seen(row for idx, row in enumerate(current) if idx not in replacements)
        total = sum(key[0] for key in keys)
        realized = sum(budgets.values()) / total
        if abs(realized - CFG["replay_input_ratio"]) > CFG["replay_ratio_tolerance"]:
            raise ValueError(f"Replay fraction outside tolerance for {lang}: {realized}")
        write_json(directory / f"replay_{lang}.json", dict(buffers=buffers, replacements=replacements))
        summary[lang] = dict(buffer_size=sum(map(len, buffers.values())), input_fraction=realized,
                             replay_input_tokens=dict(budgets), replaced_rows=len(replacements),
                             old_examples_used={old: len(uses) for old, uses in used.items()},
                             replay_supervised_tokens=sum(keys[idx][0] - keys[idx][2] for idx in replacements))
    return summary


def massive_audit(destination):
    source = SOURCE / "AmazonScience--massive" / "amazon-massive-dataset-1.1.tar.gz"
    per_lang = {}
    examples = {}
    id_sets = {}
    locales = CFG["locales"]
    all_intents, all_slots = set(), set()
    for lang in LANGUAGES:
        per_lang[lang] = dict()
        examples[lang] = dict()
        locale = locales[lang]
        with tarfile.open(source, "r|gz") as archive:
            for item in archive:
                if item.name != f"1.1/data/{locale}.jsonl":
                    continue
                with archive.extractfile(item) as stream:
                    for line in stream:
                        record = json.loads(line)
                        split = record["partition"]
                        if record["locale"] != locale:
                            raise ValueError(f"MASSIVE locale mismatch at {locale}/{record['id']}")
                        lookup = examples[lang].setdefault(split, {})
                        if record["id"] in lookup:
                            raise ValueError(f"Repeated MASSIVE id {locale}/{record['id']}")
                        slots = re.findall(r"\[([a-z0-9_]+)\s*:", record["annot_utt"])
                        if len(slots) != record["annot_utt"].count("["):
                            raise ValueError(f"Unparsed slot syntax {locale}/{record['id']}")
                        lookup[record["id"]] = (record["intent"], slots)
                break
        counts = {split: len(rows) for split, rows in examples[lang].items()}
        if counts != {"train": 11514, "dev": 2033, "test": 2974}:
            raise ValueError(f"Unexpected MASSIVE split counts {locale}: {counts}")
        lang_ids = {split: set(rows) for split, rows in examples[lang].items()}
        if sum(map(len, lang_ids.values())) != len(set.union(*lang_ids.values())):
            raise ValueError(f"Split overlap in MASSIVE/{locale}")
        if id_sets and lang_ids != id_sets:
            raise ValueError(f"Parallel source ID mismatch in MASSIVE/{locale}")
        id_sets = lang_ids
        for split, records in examples[lang].items():
            per_lang[lang][split] = dict(
                examples=len(records),
                intents=len({intent for intent, _ in records.values()}),
                slot_types=len({slot for _, slots in records.values() for slot in slots}),
            )
            all_intents.update(intent for intent, _ in records.values())
            all_slots.update(slot for _, slots in records.values() for slot in slots)
    by_intent = defaultdict(list)
    for idx, (intent, _) in examples["en"]["train"].items():
        if any(examples[lang]["train"][idx][0] != intent for lang in LANGUAGES):
            raise ValueError(f"Unaligned MASSIVE intent at {idx}")
        by_intent[intent].append(idx)
    proposed = {lang: [] for lang in LANGUAGES}
    rng = random.Random(CFG["seed"] + 5000)
    unused = []
    for intent, candidates in sorted(by_intent.items()):
        rng.shuffle(candidates)
        n = min(len(candidates) // len(LANGUAGES),
                max(1, len(candidates) * CFG["massive_train_per_language"] // len(id_sets["train"])))
        for i, lang in enumerate(LANGUAGES):
            proposed[lang].extend(candidates[i*n:(i+1)*n])
        unused.extend(candidates[n*len(LANGUAGES):])
    rng.shuffle(unused)
    proposed_stats = {}
    for lang in LANGUAGES:
        n = CFG["massive_train_per_language"] - len(proposed[lang])
        if not 0 <= n <= len(unused):
            raise ValueError(f"Unable to fill MASSIVE quotas: {lang}")
        proposed[lang].extend(unused[:n])
        del unused[:n]
        records = [examples[lang]["train"][idx] for idx in proposed[lang]]
        proposed_stats[lang] = dict(
            intent_counts=dict(Counter(intent for intent, _ in records)),
            slot_counts=dict(Counter(slot for _, slots in records for slot in slots)),
        )
    write_json(destination / "massive_proposed_ids.json", proposed)
    return dict(archive=str(source), sha256=sha(source), locales=locales, per_language=per_lang,
                all_intents=len(all_intents), all_slot_types=len(all_slots),
                proposed_subset=proposed_stats,
                proposed_unique_ids=len(set().union(*map(set, proposed.values()))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True)
    source = SOURCE / "facebook--xnli"
    meta = json.loads((source / "metadata.json").read_text())
    if meta["sha"] != CFG["xnli_revision"]:
        raise ValueError("Unexpected XNLI revision")
    folder = source / "all_languages"
    paths = {split: next(folder.glob(f"{split}-*.parquet")) for split in ("validation", "test")}
    train = sorted(folder.glob("train-*.parquet"))
    if len(train) != 4:
        raise ValueError("Expected four XNLI training shards")
    tok = tokenizer()
    print("Checking official XNLI validation and test", flush=True)
    forbidden, splits, label_lengths = read_eval(paths, out, tok)
    print("Sampling disjoint XNLI training examples", flush=True)
    selected, discarded = select_train(train, tok, forbidden)
    for lang, dataset in selected.items():
        write_rows(out / f"train_{lang}.jsonl", dataset)
    replay = replay_schedule(selected, out)
    full_ids = [row["id"] for dataset in selected.values() for row in dataset]
    random.Random(CFG["seed"] + 3000).shuffle(full_ids)
    write_json(out / "joint_order.json", full_ids)
    smoke = defaultdict(list)
    for lang, dataset in splits["validation"].items():
        # Use three fixed development examples per class to exercise all label lengths.
        for label in range(len(LABELS)):
            smoke[lang].extend([row for row in dataset if row["label"] == label][:3])
        write_rows(out / f"smoke_{lang}.jsonl", sorted(smoke[lang], key=lambda row: row["id"]))
    for split in ("test", "validation", "smoke"):
        cfgs = out / "tasks" / split
        cfgs.mkdir(parents=True)
        for lang in LANGUAGES:
            task = dict(task=f"seqlora_xnli_{split}_{lang}", dataset_path="json",
                        dataset_kwargs={"data_files": {"test": str(out / f"{split}_{lang}.jsonl")}},
                        test_split="test", output_type="multiple_choice", doc_to_text="{{prompt}}",
                        doc_to_choice=list(LABELS), doc_to_target="label",
                        target_delimiter=CFG["label_delimiter"], num_fewshot=0,
                        metric_list=[dict(metric="acc", aggregation="mean", higher_is_better=True)])
            (cfgs / f"{lang}.yaml").write_text(yaml.safe_dump(task, sort_keys=False))
    print("Auditing official MASSIVE archive", flush=True)
    audit = massive_audit(out)
    if audit["proposed_unique_ids"] != CFG["massive_train_per_language"] * len(LANGUAGES):
        raise ValueError("MASSIVE proposed source IDs overlap")
    val_groups = {x["premise_group"] for x in splits["validation"]["en"]}
    test_groups = {x["premise_group"] for x in splits["test"]["en"]}
    summary = dict(
        xnli_revision=meta["sha"], train_source_rows=sum(pq.ParquetFile(path).metadata.num_rows for path in train),
        official_counts={split: {lang: len(dataset) for lang, dataset in langs.items()}
                         for split, langs in splits.items()},
        validation_test_shared_english_premise_groups=len(val_groups & test_groups),
        model={path.name: sha(path) for path in sorted(MODEL.iterdir())
               if path.name.endswith((".safetensors", ".json", ".txt"))},
        source_files={str(path): sha(path) for path in [*train, *paths.values()]},
        protocol_sha256=sha(ROOT / "protocol.json"),
        label_token_lengths=label_lengths, discarded=dict(discarded), replay=replay, massive=audit,
        training={lang: dict(examples=len(dataset), labels=dict(Counter(row["label"] for row in dataset)),
                             input_tokens=sum(len(row["tokens"]) for row in dataset),
                             supervised_tokens=sum(len(row["tokens"]) - row["answer_start"] for row in dataset),
                             unique_english_premises=len({row["group"] for row in dataset}),
                             max_length=max(len(row["tokens"]) for row in dataset))
                  for lang, dataset in selected.items()},
        files={str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file()},
    )
    write_json(out / "MANIFEST.json", summary)
    print(f"Prepared and verified: {out}", flush=True)


if __name__ == "__main__":
    main()
