import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "protocol.json").read_text())
LANGUAGES = tuple(CFG["languages"])
LABELS = tuple(CFG["labels"])
MODEL = Path(CFG["model"])


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle]


def write_rows(path, records):
    with Path(path).open("x") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, local_files_only=True)


def prompt(tokenizer, premise, hypothesis):
    content = CFG["xnli_user_prompt"].format(premise=premise, hypothesis=hypothesis)
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return template + CFG["xnli_assistant_prefill"]
