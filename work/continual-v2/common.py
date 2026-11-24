"""Frozen source identity and output paths for Benchmark V2."""

import os as _release_os
import sys as _release_sys

from hashlib import file_digest
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
OUTPUTS = WORKSPACE / "outputs"
SUMMARY = OUTPUTS / "continual-bench-v2"
ACCOUNT = Path(_release_os.environ.get('CL_ACCOUNT_ROOT', 'local'))
HISTORICAL = ACCOUNT / "logs"
LOGS = HISTORICAL / "continual_bench_v2_20261122"
MODEL = ACCOUNT / "models/Qwen3-4B"
CKPT = Path(_release_os.path.join(_release_os.environ.get("CL_EXTERNAL_ROOT", "local"), 'checkpoints/continual_bench_v2_20261122'))
TRAIN = Path(_release_os.environ.get('CL_TRAIN_PYTHON', _release_sys.executable))
EVAL = Path(_release_os.environ.get('CL_EVAL_PYTHON', _release_sys.executable))
MASSIVE = ("de-DE", "ar-SA", "vi-VN", "sw-KE", "ur-PK")
MTOP = ("fr", "es", "hi", "de", "th")
SEED = 42


def sha(path):
    with Path(path).open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(".part")
    assert not path.exists(), path
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temp.rename(path)


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return list(map(json.loads, stream))


def budgets(rows):
    return {
        "rows": len(rows),
        "input_tokens": sum(len(row["tokens"]) for row in rows),
        "supervised_tokens": sum(len(row["tokens"]) - row["answer_start"] for row in rows),
    }


def source(data):
    p = Path(data) / "MANIFEST.json"
    digest = sha(p)
    return dict(data=str(p.parent), manifest_sha256=digest)


def dataset(plan):
    folder = Path(plan["data"])
    path = folder / "MANIFEST.json"
    assert sha(path) == plan["manifest_sha256"], path
    return folder, read(path)


def schedule(data, manifest, filename):
    path = data / filename
    assert sha(path) == manifest["files"][filename], path
    return read_rows(path)
