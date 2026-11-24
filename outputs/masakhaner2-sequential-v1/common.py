"""Fixed paths and annotation conventions for the MasakhaNER 2.0 sequence."""

import os as _release_os
import sys as _release_sys

import json
from hashlib import file_digest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PILOT = WORKSPACE/'outputs/followup/tydiqa'
LEGACY_TRAINER = WORKSPACE/'outputs/sequential-lora/train.py'
PROTOCOL = ROOT/'protocol.json'
WORK = WORKSPACE/'work/masakhaner2'
DATA = WORK/'packed-v1'
MODEL = _release_os.environ.get('CL_MODEL', 'Qwen3-4B')
MODEL_ROOT = Path(_release_os.path.join(_release_os.environ.get("CL_EXTERNAL_ROOT", "local"), 'checkpoints/masakhaner2_sequential_v1_20261121'))
TRAIN_PY = _release_os.environ.get('CL_TRAIN_PYTHON', _release_sys.executable)
EVAL_PY = _release_os.environ.get('CL_EVAL_PYTHON', _release_sys.executable)
LOGS = Path(_release_os.path.join(_release_os.environ.get("CL_EXTERNAL_ROOT", "local"), 'logs/masakhaner2_sequential_v1_20261121'))
ORDER = ('hau', 'yor', 'swa', 'lug')
STAGES = tuple(f'{i:02d}_{code}' for i,code in enumerate(ORDER, 1))
TYPES = frozenset(('PER','ORG','LOC','DATE'))
TRAIN_ROWS, DEV_ROWS = 1024, 384
STEPS = TRAIN_ROWS//16

sys.path.insert(0, str(WORK/'vendor'))


def sha(path):
    with Path(path).open('rb') as source:
        return file_digest(source, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


CFG = read(PROTOCOL)


def current_model_verified(_):
    identity=read(WORKSPACE/'outputs/massive-cl-v1/DATA_MANIFEST.json')['model']
    assert all(sha(Path(MODEL)/name)==digest for name,digest in identity.items())


def atomic_json(path, value):
    part = path.with_suffix('.part')
    with part.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    assert not path.exists()
    part.rename(path)


def manifest():
    data = DATA/'MANIFEST.json'
    assert sha(data)==sha(ROOT/'DATA_MANIFEST.json')
    return read(data)


def load(name, receipt):
    item = DATA/name
    assert sha(item) == receipt['files'][name], item
    with item.open(encoding='utf-8') as stream:
        return list(map(json.loads, stream))


def spans_to_tags(spans, length):
    if not isinstance(spans, list):
        raise ValueError('Expected list of spans')
    tags=['O']*length
    for span in spans:
        if not isinstance(span,list) or len(span)!=3:
            raise ValueError('Expected [start,end,type]')
        start,end,label=span
        if (type(start) is not int or type(end) is not int or not isinstance(label,str)
                or not 0 <= start < end <= length or label not in TYPES):
            raise ValueError('Invalid span or type')
        if any(tag!='O' for tag in tags[start:end]):
            raise ValueError('Overlapping entities')
        tags[start:end]=[f'B-{label}', *([f'I-{label}']*(end-start-1))]
    return tags
