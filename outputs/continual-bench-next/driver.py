"""Run missing benchmark cells in independent directories and preserve historical receipts."""

import os as _release_os
import sys as _release_sys

import argparse
from datetime import datetime
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT.parent
ACCOUNT = Path(_release_os.environ.get('CL_ACCOUNT_ROOT', 'local'))
HISTORICAL = ACCOUNT/'logs'
CKPT = ACCOUNT/'models/checkpoint'
ENV = Path(_release_os.path.join(_release_os.environ.get("CL_EXTERNAL_ROOT", "local"), 'envs'))
TRAIN = ENV/'qwen3-multilingual-20261031/bin/python'
EVAL = ENV/'qwen3-eval-vllm-20261101/bin/python'
MASSIVE = ('de-DE','ar-SA','vi-VN','sw-KE','ur-PK')
SWAPPED = (*MASSIVE[:3], MASSIVE[4], MASSIVE[3])
NER = ('hau','yor','swa','lug')


def launch_cap():
    return 7 if datetime.now(ZoneInfo('America/Halifax')).weekday() >= 5 else 4


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def guard(count):
    requested = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    selected = set(requested)
    assert len(requested) == len(selected) == count and all(x.startswith('GPU-') for x in selected)
    for _ in range(2):
        snapshot = ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],timeout=20))
        own, free = set(), set()
        for gpu in snapshot.findall('gpu'):
            uuid = gpu.findtext('uuid')
            busy = False
            for process in gpu.findall('processes/process_info'):
                memory = int(process.findtext('used_memory').split()[0])
                if (process.findtext('process_name') == '/usr/lib/xorg/Xorg' and
                    process.findtext('type') == 'G' and memory <= 64):
                    continue
                busy = True
                pid = process.findtext('pid')
                try:
                    if Path(f'/proc/{pid}').stat().st_uid == os.getuid():
                        own.add(uuid)
                except FileNotFoundError:
                    pass
            if (not busy and int(gpu.findtext('fb_memory_usage/free').split()[0]) >= 22000 and
                int(gpu.findtext('utilization/gpu_util').split()[0]) <= 5):
                free.add(uuid)
        if not selected <= free or len(selected | own) > launch_cap():
            raise RuntimeError(f'Unsafe GPU handoff: requested {requested}, free {sorted(free)}, account {sorted(own)}')
        time.sleep(2)
    print(json.dumps({'gpus':requested,'own_before':sorted(own),'cap':launch_cap()}),flush=True)


def execute(args, gpus=1, timeout=10800):
    guard(gpus)
    args = list(map(str,args))
    print(json.dumps({'command':args}),flush=True)
    subprocess.run(args,check=True,timeout=timeout)


def verify_predictions(folder, languages):
    for language in languages:
        receipt = read(folder/f'{language}.DONE.json')
        data = folder/f'{language}.jsonl'
        assert receipt['prediction_sha256'] == digest(data), data
        assert not receipt.get('smoke',False) and receipt['split'] == 'test'


def stages(order):
    return tuple(f'{i:02}_{name}' for i,name in enumerate(order,1))


def massive_full(original, output):
    label = 'massive_full_sft_v1_20261121' if original else 'massive_full_sft_swap_v1_20261122'
    order = MASSIVE if original else SWAPPED
    checkpoints = CKPT/label/'train'
    assert read(checkpoints/'DONE.json')['steps']==625
    for step in stages(order):
        historical = HISTORICAL/label/'evaluation'/step/'test'
        existing = [lang for lang in order if original and (historical/f'{lang}.DONE.json').is_file()]
        verify_predictions(historical,existing)
        pending = [lang for lang in order if lang not in existing]
        if pending:
            execute([EVAL,'-u',OUTPUTS/'massive-full-sft-v1/evaluate_full.py',checkpoints/step,output,'test',*pending])
            verify_predictions(output/step/'test',pending)


def massive_lora(output, arm):
    for label, order, protocol in [('massive_cl_v1_20261120', MASSIVE, 'massive-cl-v1'),
                                   ('massive_cl_single_order_v1_20261121', SWAPPED, 'massive-cl-single-order-v1')]:
        src = HISTORICAL/label/f'confirm-{arm}'
        for step in stages(order):
            if arm=='replay' and step.startswith('01_'):
                branch='seq'
            else:
                branch=arm
            checkpoint = CKPT/label/branch/step/'step-125'
            assert read(checkpoint/'DONE.json')['global_step']==int(step[:2])*125
            if step.startswith('05_'):
                verify_predictions(src/arm/'test',order)
                continue
            existing = {}
            if label=='massive_cl_v1_20261120':
                existing={'03_vi-VN':('sw-KE','before-sw'),'04_sw-KE':('sw-KE','after-sw')}
            else:
                existing={'03_vi-VN':('ur-PK','before-ur'),'04_ur-PK':('ur-PK','after-ur')}
            skip = []
            if step in existing:
                lang,suffix=existing[step]
                prior=src/f'{arm}-{suffix}'/'test'
                verify_predictions(prior,[lang])
                skip.append(lang)
                if label.endswith('20261121') and step=='04_ur-PK':
                    verify_predictions(src/f'{arm}-before-sw'/'test',['sw-KE'])
                    skip.append('sw-KE')
            pending=[lang for lang in order if lang not in skip]
            execute([EVAL,'-u',OUTPUTS/protocol/'evaluate.py',output/label,f'{step}={checkpoint}',
                     '--split','test','--locales',*pending])
            verify_predictions(output/label/step/'test',pending)


def ner_matrix(output):
    label='masakhaner2_sequential_v1_20261121'
    script=OUTPUTS/'masakhaner2-sequential-v1/evaluate.py'
    for step in stages(NER):
        historical=HISTORICAL/label/'seq/test'/step
        existing=[lang for lang in NER if (historical/f'{lang}.DONE.json').is_file()]
        verify_predictions(historical,existing)
        pending=[lang for lang in NER if lang not in existing]
        checkpoint=CKPT/label/'seq'/step/'step-064'
        if pending:
            execute([EVAL,'-u',script,'test',output/step,*pending,'--checkpoint',checkpoint])
            verify_predictions(output/step,pending)
    for lang in NER:
        historical=HISTORICAL/label/'controls/test'/f'single_{lang}'
        if (historical/f'{lang}.DONE.json').is_file():
            verify_predictions(historical,[lang])
            continue
        checkpoint=CKPT/label/'single'/lang/'step-064'
        execute([EVAL,'-u',script,'test',output/f'single_{lang}',lang,'--checkpoint',checkpoint])
        verify_predictions(output/f'single_{lang}',[lang])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role',choices=('full-swap-train','full-original-test','full-swap-test','lora-seq-test','lora-replay-test','ner-test'))
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    done=args.output/'DONE.json'
    assert not done.exists()
    if args.role=='full-swap-train':
        execute([TRAIN,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=4','--max-restarts=0',
                 ROOT/'train_swapped.py','train'],4,14400)
        result=CKPT/'massive_full_sft_swap_v1_20261122/train/DONE.json'
        assert read(result)['steps']==625
    elif args.role=='ner-test':
        ner_matrix(args.output)
    elif args.role.startswith('lora-'):
        massive_lora(args.output,args.role.split('-')[1])
    else:
        massive_full(args.role=='full-original-test',args.output)
    with done.open('x') as stream:
        json.dump({'role':args.role,'driver_sha256':digest(Path(__file__))},stream)


if __name__=='__main__':
    main()
