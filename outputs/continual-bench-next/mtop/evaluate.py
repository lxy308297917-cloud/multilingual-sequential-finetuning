"""Evaluate MTOP flat intent and slots against the author token boundaries."""

import argparse
from itertools import chain
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from seqeval.metrics import f1_score
from sklearn.metrics import f1_score as macro_f1

from runtime import CONFIG, MODEL, ORDER, OUT, manifest, read_rows, sha, verify_model


def tags(spans, width, known):
    if not isinstance(spans, list):
        raise ValueError('slots must be a list')
    last=0
    out=['O']*width
    for span in spans:
        if not isinstance(span,list) or len(span)!=3:
            raise ValueError('expected [start,end,type]')
        first,end,kind=span
        if type(first) is not int or type(end) is not int or not (last<=first<end<=width) or not isinstance(kind,str) or kind not in known:
            raise ValueError('invalid slot')
        out[first:end]=[f'B-{kind}',*([f'I-{kind}']*(end-first-1))]
        last=end
    return out


def prediction(output, width, intentions, slots):
    def keyed(pairs):
        obj=dict(pairs)
        if len(obj)!=len(pairs):
            raise ValueError('duplicate JSON key')
        return obj
    try:
        obj=json.loads(output,object_pairs_hook=keyed)
        if not isinstance(obj,dict) or set(obj)!={'intent','slots'} or not isinstance(obj['intent'],str) or obj['intent'] not in intentions:
            raise ValueError('invalid JSON or intent')
    except (ValueError,TypeError,RecursionError):
        return None,['O']*width,set(),False
    try:
        return obj['intent'],tags(obj['slots'],width,slots),set(map(tuple,obj['slots'])),True
    except (ValueError,TypeError,RecursionError):
        return obj['intent'],['O']*width,set(),False


def grade(gold, outputs, receipt):
    records=[]
    intents, slots=set(receipt['intents']),set(receipt['slots'])
    for row,raw in zip(gold,outputs,strict=True):
        width=len(row['words'])
        intent,bio,pred,valid=prediction(raw,width,intents,slots)
        expected=set(map(tuple,row['slots']))
        true_tags=tags(row['slots'],width,{slot[-1] for slot in row['slots']})
        records.append(dict(id=row['id'],output=raw,gold=[row['intent'],*true_tags],
                       pred=[intent,*bio],valid=valid,intent_correct=intent==row['intent'],
                       exact=valid and intent==row['intent'] and expected==pred,
                       tp=len(pred&expected),fp=len(pred-expected),fn=len(expected-pred)))
    true=[r['gold'][1:] for r in records]
    predicted=[r['pred'][1:] for r in records]
    tp,fp,fn=(sum(row[key] for row in records) for key in ('tp','fp','fn'))
    slot=100*f1_score(true,predicted,zero_division=0)
    assert np.isclose(slot,(200*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.)
    bio=100*macro_f1(list(chain.from_iterable(true)),list(chain.from_iterable(predicted)),
                    average='macro',zero_division=0)
    return records,dict(f1=slot,token_macro_bio_f1=bio,intent_acc=100*np.mean([r['intent_correct'] for r in records]),
                        exact=100*np.mean([r['exact'] for r in records]),valid=sum(r['valid'] for r in records))


def generate(args,receipt):
    verify_model(receipt)
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    assert gpu.startswith('GPU-') and ',' not in gpu

    from vllm_compat import install
    install()
    from vllm import LLM, SamplingParams, __version__
    from vllm.lora.request import LoRARequest

    model=None
    request=None
    if args.checkpoint:
        model=json.loads((args.checkpoint/'DONE.json').read_text())
        assert not model['smoke'] and model['complete_stage'] and model['manifest_sha256']==sha(OUT/'MANIFEST.json')
        assert all(sha(args.checkpoint/name)==digest for name,digest in model['files'].items() if name.startswith('adapter/'))
        request=LoRARequest('mtop',1,str(args.checkpoint/'adapter'))
    options=CONFIG['generation']
    engine=LLM(model=str(MODEL),tokenizer=str(MODEL),dtype='bfloat16',tensor_parallel_size=1,
               max_model_len=options['max_model_len'],max_num_seqs=options['max_num_seqs'],
               max_num_batched_tokens=options['max_num_batched_tokens'],gpu_memory_utilization=.78,
               seed=42,generation_config='vllm',enable_prefix_caching=True,enforce_eager=True,
               enable_lora=request is not None,max_lora_rank=16)
    sampling=SamplingParams(temperature=0,top_p=1,max_tokens=options['max_new_tokens'],
                            seed=42,stop_token_ids=options['stop_token_ids'],skip_special_tokens=True)
    args.output.mkdir(parents=True)
    for lang in args.languages:
        start=time.monotonic()
        name=f'{args.split}_{lang}.jsonl'
        rows=read_rows(name,receipt)
        texts=[]
        batch_size=options['submit_batch']
        for i in range(0,len(rows),batch_size):
            prompts=[dict(prompt_token_ids=r['tokens']) for r in rows[i:i+batch_size]]
            found=engine.generate(prompts,sampling,lora_request=request,use_tqdm=False)
            for prompt,result in zip(prompts,found,strict=True):
                assert prompt['prompt_token_ids']==list(result.prompt_token_ids)
                texts.append(result.outputs[0].text.strip())
        records,scores=grade(rows,texts,receipt)
        data=args.output/f'{lang}.jsonl'
        with data.open('x',encoding='utf-8') as f:
            for item in records:
                print(json.dumps(item,ensure_ascii=False),file=f)
        cert=dict(split=args.split,locale=lang,n=len(rows),**scores,input_sha256=receipt['files'][name],
                  prediction_sha256=sha(data),checkpoint=model,scorer_sha256=sha(Path(__file__)),
                  generation=options,vllm=__version__,gpu=gpu,seconds=time.monotonic()-start)
        out=args.output/f'{lang}.DONE.json'
        temp=out.with_suffix('.part')
        temp.write_text(json.dumps(cert,indent=2)+'\n')
        temp.rename(out)
        print(json.dumps({k:cert[k] for k in ('locale','n','f1','valid','seconds')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('split',choices=('dev','test','gold-check'))
    p.add_argument('output',type=Path)
    p.add_argument('languages',nargs='+',choices=ORDER)
    p.add_argument('--checkpoint',type=Path)
    a=p.parse_args()
    receipt=manifest()
    if a.split=='gold-check':
        for lang in a.languages:
            dataset=read_rows(f'dev_{lang}.jsonl',receipt)
            _,scores=grade(dataset,[row['answer'] for row in dataset],receipt)
            assert all(np.isclose(scores[k],100) for k in ('f1','token_macro_bio_f1','intent_acc','exact')),(lang,scores)
            print(lang,len(dataset),scores,flush=True)
    else:
        generate(a,receipt)


if __name__=='__main__':
    main()
