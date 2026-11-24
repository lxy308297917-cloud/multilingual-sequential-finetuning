"""Run Qwen3 generation on frozen NER rows and score with upstream seqeval."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from common import CFG, DATA, MODEL, ORDER, atomic_json, current_model_verified, load, manifest, read, sha, spans_to_tags
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.metrics.sequence_labeling import get_entities


def predict_tags(text, length):
    try:
        spans=json.loads(text)
        return spans_to_tags(spans,length), True
    except (ValueError, TypeError, RecursionError):
        return ['O']*length, False


def metrics(gold, predictions):
    return dict(precision=100*precision_score(gold,predictions,zero_division=0),
                recall=100*recall_score(gold,predictions,zero_division=0),
                f1=100*f1_score(gold,predictions,zero_division=0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('split',choices=['dev','test'])
    p.add_argument('output',type=Path)
    p.add_argument('locales',nargs='+',choices=ORDER)
    p.add_argument('--checkpoint',type=Path)
    args=p.parse_args()
    spec=manifest()
    current_model_verified(spec)
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    assert gpu.startswith('GPU-') and ',' not in gpu

    from vllm_compat import install
    install()
    from vllm import LLM, SamplingParams, __version__ as vllm_version
    from vllm.lora.request import LoRARequest

    request, cert=None,None
    if args.checkpoint:
        cert=read(args.checkpoint/'DONE.json')
        assert cert['manifest_sha256']==sha(DATA/'MANIFEST.json') and cert['complete_stage']
        for name,digest in cert['files'].items():
            if name.startswith('adapter/'):
                assert sha(args.checkpoint/name)==digest
        request=LoRARequest('masakhaner2',1,str(args.checkpoint/'adapter'))
    cfg=CFG['generation']
    engine=LLM(model=MODEL,tokenizer=MODEL,dtype='bfloat16',tensor_parallel_size=1,
               max_model_len=spec['generation']['max_model_len'],max_num_seqs=cfg['max_sequences'],
               max_num_batched_tokens=cfg['max_batched_tokens'],gpu_memory_utilization=.78,seed=42,
               generation_config='vllm',enable_prefix_caching=True,enforce_eager=True,
               enable_lora=request is not None,max_lora_rank=16)
    sampling=SamplingParams(temperature=0,top_p=1,max_tokens=spec['generation']['max_new_tokens'],
                            seed=42,stop_token_ids=cfg['stop_token_ids'],skip_special_tokens=True)
    args.output.mkdir(parents=True)
    for code in args.locales:
        name=f'{args.split}_{code}.jsonl'
        data=load(name,spec)
        started=time.monotonic()
        records=[]
        for index in range(0,len(data),cfg['batch']):
            batch=data[index:index+cfg['batch']]
            prompts=[dict(prompt_token_ids=row['tokens'][:row['answer_start']]) for row in batch]
            outputs=engine.generate(prompts,sampling,lora_request=request,use_tqdm=False)
            for row,prompt,out in zip(batch,prompts,outputs,strict=True):
                assert prompt['prompt_token_ids']==list(out.prompt_token_ids)
                choice=out.outputs[0]; text=choice.text.strip()
                labels,valid=predict_tags(text,len(row['tags']))
                gold, pred=set(get_entities(row['tags'])),set(get_entities(labels))
                records.append(dict(id=row['id'],output=text,gold=row['tags'],pred=labels,valid=valid,
                                    finish=choice.finish_reason,tp=len(gold&pred),fp=len(pred-gold),fn=len(gold-pred)))
        assert len(records)==len(data)
        path=args.output/f'{code}.jsonl'
        with path.open('x',encoding='utf-8') as stream:
            for row in records: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        score=metrics([r['gold'] for r in records],[r['pred'] for r in records])
        tp,fp,fn=(sum(row[key] for row in records) for key in ('tp','fp','fn'))
        assert abs(score['f1']-(200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0))<1e-8
        receipt=dict(split=args.split,locale=code,n=len(records),**score,valid=sum(r['valid'] for r in records),
                     tp=tp,fp=fp,fn=fn,prediction_sha256=sha(path),input_sha256=spec['files'][name],
                     checkpoint=cert,device=gpu,vllm=vllm_version,seqeval=spec['scorer'],
                     generation=spec['generation'],seconds=time.monotonic()-started)
        atomic_json(args.output/f'{code}.DONE.json',receipt)
        print(json.dumps({k:receipt[k] for k in ('split','locale','n','f1','valid','seconds')}),flush=True)


if __name__=='__main__':
    main()
