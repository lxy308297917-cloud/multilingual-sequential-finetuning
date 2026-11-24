"""Generate GoldP answers with vLLM and score with the original SQuAD evaluator."""
import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys
import time

from common import CFG, DATA, MODEL, RAW, atomic_json, current_model_verified, load, manifest, read, sha


def grader(path):
    spec = spec_from_file_location('goldp_original_squad_v1_1', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('locales', nargs='+')
    cli.add_argument('--checkpoint', type=Path)
    cli.add_argument('--smoke', action='store_true')
    args = cli.parse_args()
    receipt = manifest()
    current_model_verified(receipt)
    eval_hash = sha(RAW / 'evaluate-v1.1.py')
    assert eval_hash == receipt['scoring_sha256']
    official = grader(RAW / 'evaluate-v1.1.py')
    assert set(args.locales) <= {*receipt['languages'], 'en'}
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    assert gpu.startswith('GPU-') and ',' not in gpu

    from vllm_compat import install
    install()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    request, model_receipt = None, None
    manifest_hash = sha(DATA / 'MANIFEST.json')
    if args.checkpoint:
        model_receipt = read(args.checkpoint / 'DONE.json')
        assert model_receipt['manifest_sha256'] == manifest_hash and model_receipt['smoke'] == args.smoke
        for name, expected in model_receipt['files'].items():
            if name.startswith('adapter/'):
                assert sha(args.checkpoint / name) == expected
        request = LoRARequest('goldp-single', 1, str(args.checkpoint / 'adapter'))
    cfg = CFG['generation']
    engine = LLM(model=MODEL, tokenizer=MODEL, dtype='bfloat16', tensor_parallel_size=1, max_model_len=cfg['max_model_len'],
                 max_num_seqs=cfg['max_sequences'], max_num_batched_tokens=cfg['max_batched_tokens'],
                 gpu_memory_utilization=.78, seed=CFG['seed'], generation_config='vllm', enable_prefix_caching=True,
                 enforce_eager=True, enable_lora=request is not None, max_lora_rank=16)
    sampling = SamplingParams(temperature=cfg['temperature'], max_tokens=cfg['max_tokens'], top_p=1.0,
                              stop_token_ids=cfg['stop_token_ids'], seed=CFG['seed'], skip_special_tokens=True)
    args.output.mkdir(parents=True)
    for locale in args.locales:
        rows = load(f'dev_{locale}.jsonl', receipt)
        if args.smoke:
            rows = rows[:12]
        payload = [{'prompt_token_ids': row['tokens'][:row['answer_start']]} for row in rows]
        predictions, records = {}, []
        started = time.monotonic()
        for index in range(0, len(rows), cfg['submit_batch']):
            prompts = payload[index:index+cfg['submit_batch']]
            outputs = engine.generate(prompts, sampling, lora_request=request, use_tqdm=False)
            for row, prompt, result in zip(rows[index:index+len(prompts)], prompts, outputs, strict=True):
                assert prompt['prompt_token_ids'] == list(result.prompt_token_ids)
                generated = result.outputs[0]
                text = generated.text.strip()
                predictions[row['id']] = text
                gold = row['answers']
                records.append(dict(id=row['id'], title=row['title'], output=text, answers=gold,
                                    in_context=bool(text and text in row['context']), finish=generated.finish_reason,
                                    em=int(official.metric_max_over_ground_truths(official.exact_match_score, text, gold)),
                                    f1=official.metric_max_over_ground_truths(official.f1_score, text, gold)))
        assert len(records) == len(predictions) == len(rows)
        data = [{'paragraphs': [{'qas': [{'id': r['id'], 'answers': [{'text': t} for t in r['answers']]}
                                         for r in rows]}]}]
        scores = official.evaluate(data, predictions)
        assert abs(scores['exact_match'] - 100*sum(r['em'] for r in records)/len(rows)) < 1e-8
        assert abs(scores['f1'] - 100*sum(r['f1'] for r in records)/len(rows)) < 1e-8
        output = args.output / f'{locale}.jsonl'
        with output.open('x') as file:
            for item in records:
                file.write(json.dumps(item, ensure_ascii=False) + '\n')
        result = dict(locale=locale, n=len(rows), smoke=args.smoke, split='internal-train-derived-dev',
                      manifest_sha256=manifest_hash, input_sha256=receipt['files'][f'dev_{locale}.jsonl'],
                      prediction_sha256=sha(output), official_scorer_sha256=eval_hash, generation=cfg, device=gpu,
                      checkpoint=model_receipt, exact_match=scores['exact_match'], f1=scores['f1'],
                      span_mismatch_rate=1-sum(r['in_context'] for r in records)/len(rows),
                      seconds=time.monotonic()-started)
        atomic_json(args.output / f'{locale}.DONE.json', result)
        print(json.dumps({key:result[key] for key in ('locale','n','smoke','exact_match','f1','span_mismatch_rate')}),flush=True)


if __name__ == '__main__':
    main()
