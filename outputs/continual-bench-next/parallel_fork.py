"""Compare fixed-budget ordinary, source-aligned and unaligned replay forks."""

import argparse
import json
from pathlib import Path
import sys

from driver import CKPT, EVAL, MASSIVE, OUTPUTS, ROOT, TRAIN, digest, read
from handoff import execute_checked as execute


ARMS=('ordinary','unaligned','aligned')
DATA=ROOT.parents[1]/'work/massive-parallel-fork-v1'
LOCK=ROOT/'PARALLEL_FORK_MANIFEST.json'
MODEL=(CKPT/'massive_parallel_fork_v1_20261122').resolve()


def worker(argv):
    output, arm=argv
    sys.path.insert(0,str(OUTPUTS/'massive-cl-v1'))
    import train
    from common import read_rows
    recipe=read(LOCK)
    assert digest(train.DATA/'MANIFEST.json')==recipe['original_manifest_sha256']
    filename=f'fork_{arm}.jsonl'
    path=DATA/filename
    assert digest(path)==recipe['schedule_hashes'][filename]
    receipt=train.manifest()
    extended=dict(receipt,files={**receipt['files'],filename:digest(path)})
    train.manifest=lambda:extended
    ref_load=train.load
    train.load=lambda name,info:read_rows(path) if name==filename else ref_load(name,info)
    train.CFG=dict(train.CFG,progress_steps=[125])
    parent=Path(recipe['source_checkpoint'])
    assert digest(parent/'DONE.json')==recipe['source_checkpoint_sha256']
    sys.argv=['train.py',filename,output,'--trajectory','seq','--continue-from',str(parent)]
    train.main()


def main():
    if sys.argv[1:2]==['worker']:
        worker(sys.argv[2:])
        return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('role',choices=('parallel-fork',))
    p.add_argument('output',type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True)
    recipe=read(LOCK)
    for arm in ARMS:
        target=MODEL/arm
        execute([TRAIN,'-u',Path(__file__),'worker',target,arm])
        model=target/'step-125'
        receipt=read(model/'DONE.json')
        assert receipt['global_step']==625 and receipt['rows']==recipe['exposures']
        assert sum(receipt['input_tokens'].values())==recipe['budget']['input_tokens']
        assert sum(receipt['supervised_tokens'].values())==recipe['budget']['supervised_tokens']
        assert receipt['continued_from_sha256']==recipe['source_checkpoint_sha256']
        execute([EVAL,'-u',OUTPUTS/'massive-cl-v1/evaluate.py',a.output,f'{arm}={model}','--split','dev','--locales',*MASSIVE])
        for lang in MASSIVE:
            folder=a.output/arm/'dev'
            scored=read(folder/f'{lang}.DONE.json')
            assert not scored['smoke'] and scored['prediction_sha256']==digest(folder/f'{lang}.jsonl')
    with (a.output/'DONE.json').open('x') as stream:
        json.dump(dict(role=a.role,prototype_sha256=digest(LOCK),driver_sha256=digest(Path(__file__))),stream)


if __name__=='__main__':
    main()
