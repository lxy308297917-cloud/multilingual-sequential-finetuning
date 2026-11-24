"""Run the frozen five-stage GoldP SeqLoRA training trajectory."""

import os as _release_os
import sys as _release_sys

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent / 'followup/tydiqa'
sys.path.insert(0, str(PILOT))
from common import CFG, DATA, LEGACY_TRAINER, atomic_json, load, manifest, read, sha

ORDER = ('te', 'id', 'ru', 'sw', 'ar')
STEPS = 125
LOCALES = (*ORDER, 'en')
STAGES = tuple(f'{i:02d}_{locale}' for i, locale in enumerate(ORDER, 1))
MODEL_ROOT = Path(_release_os.path.join(_release_os.environ.get("CL_EXTERNAL_ROOT", "local"), 'checkpoints/tydiqa_goldp_sequential_v1_20261121'))


def preflight():
    verified = manifest()
    for line in (PILOT / 'FROZEN_SHA256SUMS').read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        assert sha(PILOT / path) == digest, path
    pilot = read(PILOT / 'RESULTS.json')['results']
    assert set(ORDER) == set(verified['languages']) and ORDER[-2] == 'sw'
    single_f1 = {code: pilot[code]['single']['f1'] for code in ORDER}
    ordered = sorted(single_f1, key=single_f1.get, reverse=True)
    ordered.remove('sw')
    ordered.insert(-1, 'sw')
    assert tuple(ordered) == ORDER
    budgets = {}
    for language in LOCALES:
        dev = load(f'dev_{language}.jsonl', verified)
        assert len(dev) == len({row['id'] for row in dev}) == 200
        if language == 'en':
            continue
        name = f'train_{language}.jsonl'
        rows = load(name, verified)
        assert len(rows) == len({row['id'] for row in rows}) == STEPS * CFG['effective_batch']
        assert {row['locale'] for row in rows} == {language}
        assert {row['id'] for row in rows}.isdisjoint(row['id'] for row in dev)
        budgets[language] = dict(rows=len(rows), optimizer_steps=STEPS, schedule_sha256=verified['files'][name],
                                 input_tokens=sum(len(row['tokens']) for row in rows),
                                 supervised_tokens=sum(len(row['tokens'])-row['answer_start'] for row in rows))
        coverage = verified['coverage'][language]
        assert budgets[language]['input_tokens'] == coverage['training_input_tokens']
        assert budgets[language]['supervised_tokens'] == coverage['supervised_tokens']
    return dict(order=ORDER, stages=STAGES, locales=LOCALES, per_stage_steps=STEPS, budgets=budgets,
                single_f1=single_f1, pilot_data_sha256=sha(DATA / 'MANIFEST.json'),
                pilot_results_sha256=sha(PILOT / 'RESULTS.json'), base_protocol_sha256=sha(PILOT / 'protocol.json'),
                training=CFG, trainer_sha256=sha(PILOT / 'train.py'),
                selective_loss_sha256=sha(LEGACY_TRAINER), checkpoints=[STEPS],
                optimizer='continue the same AdamW moments and CPU/CUDA RNG across languages')


def check_checkpoint(path, stage, previous, spec):
    certificate = read(path / 'DONE.json')
    locale = ORDER[stage]
    assert certificate['trajectory'] == 'seq' and not certificate['smoke'] and certificate['complete_stage']
    assert certificate['start_step'] == stage * STEPS and certificate['global_step'] == (stage + 1) * STEPS
    assert certificate['manifest_sha256'] == spec['pilot_data_sha256']
    assert certificate['protocol_sha256'] == spec['base_protocol_sha256']
    assert certificate['trainer_sha256'] == spec['trainer_sha256']
    assert certificate['continued_from'] == (str(previous) if previous else None)
    assert certificate['continued_from_sha256'] == (sha(previous / 'DONE.json') if previous else None)
    budget = spec['budgets'][locale]
    assert certificate['schedule_sha256'] == budget['schedule_sha256']
    for field in ('rows', 'input_tokens', 'supervised_tokens'):
        assert certificate[field] == {locale: budget[field]}
    for name, digest in certificate['files'].items():
        assert sha(path / name) == digest, (path, name)
    return certificate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='?', type=Path)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--worker', nargs=2, metavar=('SCHEDULE', 'OUTPUT'))
    parser.add_argument('--continue-from', type=Path)
    args = parser.parse_args()
    if args.worker:
        import train

        sys.argv = ['train.py', *args.worker, '--trajectory', 'seq']
        if args.continue_from:
            sys.argv.extend(['--continue-from', str(args.continue_from)])
        train.CFG = dict(CFG, progress_steps=[STEPS])
        train.main()
        return

    spec = preflight()
    if args.preflight:
        print(json.dumps(spec, ensure_ascii=False, indent=2))
        return

    args.logs.mkdir(parents=True)
    MODEL_ROOT.mkdir()
    atomic_json(args.logs / 'PROTOCOL.json', spec)
    completed, previous = {}, None
    for index, (stage, locale) in enumerate(zip(STAGES, ORDER, strict=True)):
        subprocess.run([sys.executable, str(PILOT.parent / 'check_gpu.py')], check=True, timeout=60)
        output = MODEL_ROOT / stage
        worker = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', f'train_{locale}.jsonl', str(output)]
        if previous:
            worker.extend(['--continue-from', str(previous)])
        print(json.dumps(dict(stage=stage, command=worker)), flush=True)
        subprocess.run(worker, check=True, timeout=5400)
        checkpoint = output / f'step-{STEPS:03d}'
        verified = check_checkpoint(checkpoint, index, previous, spec)
        assert read(output / 'TRAINING_DONE.json')['global_step'] == verified['global_step']
        previous = checkpoint
        completed[stage] = str(checkpoint)
        atomic_json(args.logs / f'{stage}.TRAINED.json', dict(stage=stage, path=str(checkpoint),
                    receipt_sha256=sha(checkpoint / 'DONE.json'), checkpoint=verified))
    atomic_json(args.logs / 'TRAINING_DONE.json', dict(stages=completed, protocol_sha256=sha(args.logs/'PROTOCOL.json')))


if __name__ == '__main__':
    main()
