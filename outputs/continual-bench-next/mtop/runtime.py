"""Fixed MTOP data and protocol helpers shared by training and evaluation."""

import json
from pathlib import Path

from prepare import OUT, HERE, MODEL, ORDER, sha


PROTOCOL=HERE/'protocol.json'
CONFIG=json.loads(PROTOCOL.read_text())


def manifest():
    path=OUT/'MANIFEST.json'
    assert sha(path)==(HERE/'DATA_MANIFEST.sha256').read_text().strip()
    return json.loads(path.read_text())


def read_rows(name,receipt):
    path=OUT/name
    assert sha(path)==receipt['files'][name],path
    with path.open(encoding='utf-8') as stream:
        rows=list(map(json.loads,stream))
    assert len(rows)==receipt['counts'][name]
    return rows


def verify_model(receipt):
    assert all(sha(MODEL/name)==digest for name,digest in receipt['model'].items())
