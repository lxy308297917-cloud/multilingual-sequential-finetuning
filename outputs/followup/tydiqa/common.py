"""Isolated data and checkpoint paths for the GoldP Base/Single screen."""

import os as _release_os
import sys as _release_sys
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
DATA = WORKSPACE / 'work/tydiqa-goldp/classic-v1'
RAW = DATA.parent / 'source'
LEGACY_TRAINER = WORKSPACE / 'outputs/sequential-lora/train.py'
MODEL = _release_os.environ.get('CL_MODEL', 'Qwen3-4B')
MODELS = Path(_release_os.path.join(_release_os.environ.get("CL_EXTERNAL_ROOT", "local"), 'checkpoints/tydiqa_goldp_single_v1_20261121'))
TRAIN_PY = _release_os.environ.get('CL_TRAIN_PYTHON', _release_sys.executable)
EVAL_PY = _release_os.environ.get('CL_EVAL_PYTHON', _release_sys.executable)
PROTOCOL = ROOT / 'protocol.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    pending = path.with_suffix('.part')
    with pending.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    assert not path.exists()
    pending.rename(path)


CFG = read(PROTOCOL)


def manifest():
    assert sha(ROOT / 'DATA_AUDIT.json') == sha(DATA / 'MANIFEST.json')
    receipt = read(DATA / 'MANIFEST.json')
    assert receipt['max_context'] == CFG['generation']['max_model_len']
    assert receipt['max_answer'] == CFG['generation']['max_tokens']
    assert receipt['seed'] == CFG['seed']
    return receipt


def load(name, receipt):
    path = DATA / name
    assert sha(path) == receipt['files'][name], path
    with path.open() as stream:
        return list(map(json.loads, stream))


def current_model_verified(_):
    identity = read(WORKSPACE / 'outputs/massive-cl-v1/DATA_MANIFEST.json')['model']
    assert all(sha(Path(MODEL) / name) == digest for name, digest in identity.items())
