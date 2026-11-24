"""Locations and provenance for the frozen MASSIVE sequential protocol."""

import os as _release_os
import sys as _release_sys
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "followup/massive"
WORK = ROOT.parents[1] / "work"
DATA = WORK / "massive-cl-single-order-v1-data"
LEGACY_TRAINER = ROOT.parent / "sequential-lora/train.py"
sys.path.insert(0, str(PREVIOUS))
from data import ARCHIVE, MODEL, UPSTREAM, decode, gold_output, read_rows, sha, write_rows
sys.path.remove(str(PREVIOUS))

PROTOCOL = ROOT / "protocol.json"
CFG = json.loads(PROTOCOL.read_text())
LANGUAGES = tuple(CFG["languages"])
EVAL = tuple(CFG["evaluation_order"])
CHECKPOINTS = Path(_release_os.path.join(_release_os.environ.get("CL_EXTERNAL_ROOT", "local"), 'checkpoints/massive_cl_single_order_v1_20261121'))
TRAIN_PY = _release_os.environ.get('CL_TRAIN_PYTHON', _release_sys.executable)
EVAL_PY = _release_os.environ.get('CL_EVAL_PYTHON', _release_sys.executable)


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    with tmp.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(tmp, path)


def manifest():
    receipt = read(DATA / "MANIFEST.json")
    assert receipt["protocol_sha256"] == sha(PROTOCOL)
    return receipt


def load(name, receipt):
    path = DATA / name
    assert sha(path) == receipt["files"][name], path
    return read_rows(path)


def current_model_verified(receipt):
    for name, digest in receipt["model"].items():
        assert sha(Path(MODEL) / name) == digest, name
