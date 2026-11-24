"""Evaluate each GoldP endpoint and confirm the SW contrasts on full author dev."""

import os as _release_os
import sys as _release_sys

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from train_series import LOCALES, MODEL_ROOT, ORDER, PILOT, STAGES, STEPS, check_checkpoint, preflight
from common import EVAL_PY, MODELS, WORKSPACE, atomic_json, manifest, read, sha


OFFICIAL = WORKSPACE / 'work/tydiqa-goldp/sequence-official-v1'
TRAIN_QUEUE = Path(_release_os.path.join(_release_os.environ.get("CL_EXTERNAL_ROOT", "local"), 'gpu-queue/queues/followup-tydiqa-goldp-sequence-training-v1/state.json'))


def official_data():
    lock = PILOT.parent.parent / 'tydiqa-goldp-sequential-v1/OFFICIAL_DATA.json'
    assert sha(lock) == sha(OFFICIAL / 'MANIFEST.json')
    data = read(lock)
    assert tuple(data['languages']) == LOCALES
    pilot = manifest()
    assert data['preliminary_data_manifest_sha256'] == sha(PILOT / 'DATA_AUDIT.json')
    assert pilot['scoring_sha256'] == data['scorer_sha256']
    for locale in LOCALES:
        name = f'dev_{locale}.jsonl'
        assert sha(OFFICIAL / name) == data['files'][name]
        assert data['sizes'][locale]['rows'] == pilot['coverage'][locale]['official_dev']
    return lock, data


def official_worker(argv):
    import evaluate

    lock, held = official_data()
    pilot = manifest()

    def load_held(name, _):
        with (OFFICIAL / name).open(encoding='utf-8') as stream:
            return list(map(json.loads, stream))

    def write_held(path, result):
        assert result['n'] == held['sizes'][result['locale']]['rows']
        atomic_json(path, dict(result, split=held['split'],
                              evaluation_data_manifest_sha256=sha(lock)))

    evaluate.CFG = dict(evaluate.CFG, generation=held['generation'])
    evaluate.manifest = lambda: dict(pilot, files=held['files'])
    evaluate.load = load_held
    evaluate.atomic_json = write_held
    sys.argv = ['evaluate.py', *argv]
    evaluate.main()


def evaluate(output, locales, checkpoint=None, held_out=False):
    subprocess.run([sys.executable, str(PILOT.parent / 'check_gpu.py')], check=True, timeout=60)
    prefix = [str(Path(__file__).resolve()), '--official-worker'] if held_out else [str(PILOT / 'evaluate.py')]
    command = [EVAL_PY, '-u', *prefix, str(output), *locales]
    if checkpoint:
        command.extend(['--checkpoint', str(checkpoint)])
    print(json.dumps(dict(split='author-dev' if held_out else 'internal', command=command)), flush=True)
    subprocess.run(command, check=True, timeout=10800)
    for code in locales:
        result = read(output / f'{code}.DONE.json')
        assert result['prediction_sha256'] == sha(output / f'{code}.jsonl')


def wait_for(stage, root):
    trained = root / f'{stage}.TRAINED.json'
    deadline = time.monotonic() + 10800
    while time.monotonic() < deadline:
        if trained.is_file():
            return read(trained)
        if TRAIN_QUEUE.is_file():
            status = read(TRAIN_QUEUE)
            if status['phase'] in {'failed', 'completed', 'cancelled', 'canceled', 'stopped'}:
                if trained.is_file():
                    return read(trained)
                raise RuntimeError(f'Trainer became {status["phase"]} without the {stage} receipt')
        time.sleep(25)
    raise TimeoutError(f'No verified checkpoint receipt for {stage}')


def main():
    if sys.argv[1:2] == ['--official-worker']:
        official_worker(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    training = args.output.parent / 'training'
    spec = preflight()
    official, data = official_data()
    args.output.mkdir(parents=True)
    atomic_json(args.output / 'EVALUATION_PROTOCOL.json', dict(
        order=ORDER, internal_locales=LOCALES, author_dev_base_final=LOCALES,
        author_dev_sw_intermediate=['single_sw', '03_ru', '04_sw'],
        official_manifest_sha256=sha(official), official_generation=data['generation'],
        internal_manifest_sha256=spec['pilot_data_sha256'], scorer_sha256=data['scorer_sha256'],
        original_evaluator_sha256=sha(PILOT/'evaluate.py'), evaluator_sha256=sha(Path(__file__))))
    run = args.output
    evaluate(run / 'official' / 'base', LOCALES, held_out=True)
    evaluate(run / 'official' / 'single_sw', ['sw'], MODELS / 'single/sw/step-125', held_out=True)
    previous = None
    for index, stage in enumerate(STAGES):
        signal = wait_for(stage, training)
        checkpoint = MODEL_ROOT / stage / f'step-{STEPS:03d}'
        assert signal['path'] == str(checkpoint) and signal['receipt_sha256'] == sha(checkpoint / 'DONE.json')
        check_checkpoint(checkpoint, index, previous, spec)
        previous = checkpoint
        evaluate(run / 'internal' / stage, LOCALES, checkpoint)
        if stage in {'03_ru', '04_sw', '05_ar'}:
            evaluate(run / 'official' / stage, LOCALES if stage == STAGES[-1] else ['sw'], checkpoint, held_out=True)
    atomic_json(run / 'DONE.json', dict(stages=STAGES, evaluation_protocol_sha256=sha(run/'EVALUATION_PROTOCOL.json')))


if __name__ == '__main__':
    main()
