"""Verify the upstream MasakhaNER split files and encode the fixed prompt."""

from collections import Counter
from hashlib import sha256
import json
import random
import unicodedata

from transformers import AutoTokenizer

from common import DATA, DEV_ROWS, MODEL, ORDER, ROOT, STAGES, STEPS, TRAIN_ROWS, TYPES, WORK, atomic_json, read, sha, spans_to_tags
from seqeval.metrics.sequence_labeling import get_entities


PROMPT = ('Identify every named entity in this numbered token sequence. Use PER for a person, ORG for an organization, '
          'LOC for a location, and DATE for a date or time expression. Return only a JSON list of [start,end,"TYPE"] triples. '
          'Token numbering starts at 0. The start is included and the end is excluded. Return [] if there is no entity.\n\n'
          'Tokens:\n{words}')
PREFILL='Entities:'
SEED=42
MAX_TRAIN_CONTEXT=1536
MAX_NEW=160


def read_conll(path, code, split):
    values, sentence, tags = [],[],[]
    for line in [*path.read_text(encoding='utf-8-sig').splitlines(), '']:
        if line.strip():
            token, tag = line.rsplit(maxsplit=1)
            assert token and (tag=='O' or (tag[:2] in ('B-','I-') and tag[2:] in TYPES))
            sentence.append(token); tags.append(tag)
        elif sentence:
            spans=[[start,end+1,label] for label,start,end in get_entities(tags)]
            assert get_entities(spans_to_tags(spans,len(tags))) == get_entities(tags)
            values.append(dict(id=f'{code}/{split}/{len(values)+1}', locale=code, words=sentence, tags=tags, spans=spans))
            sentence,tags=[],[]
    assert values
    return values


def identity(row):
    return tuple(unicodedata.normalize('NFKC', word).casefold() for word in row['words'])


def encode(data, tokenizer):
    for start in range(0,len(data),128):
        batch=data[start:start+128]
        messages=[tokenizer.apply_chat_template([{'role':'user', 'content':PROMPT.format(
                words=' '.join(f'{i}:{word}' for i,word in enumerate(row['words'])))}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)+PREFILL for row in batch]
        texts=[json.dumps(row['spans'], ensure_ascii=False,separators=(',',':')) for row in batch]
        prompts=tokenizer(messages, add_special_tokens=False, verbose=False)['input_ids']
        full=tokenizer([p+' '+a for p,a in zip(messages,texts,strict=True)], add_special_tokens=False,verbose=False)['input_ids']
        for row,pre,target in zip(batch,prompts,full,strict=True):
            assert target[:len(pre)] == pre
            yield dict(row,tokens=target+[tokenizer.eos_token_id],answer_start=len(pre))


def save(name, rows):
    path=DATA/name
    with path.open('x',encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    return sha(path)


def main():
    raw=WORK/'raw'
    source=read(raw/'SOURCE.json')
    assert all(sha(raw/name)==ref['sha256'] for name,ref in source['files'].items())
    splits=('train','dev','test')
    all_rows = {(code,split):read_conll(raw/f'data/{code}/{split}.txt',code,split) for code in ORDER for split in splits}
    test_text={identity(row) for code in ORDER for row in all_rows[code,'test']}
    held_text=test_text|{identity(row) for code in ORDER for row in all_rows[code,'dev']}
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    DATA.mkdir()
    files,coverage={},{}
    longest=0
    for code in ORDER:
        seeds={'hau':42,'yor':43,'swa':44,'lug':45}
        rng=random.Random(seeds[code])
        train=all_rows[code,'train']; dev=all_rows[code,'dev']; test=all_rows[code,'test']
        available=[row for row in train if identity(row) not in held_text]
        rng.shuffle(available)
        # Encode a bounded draw that leaves room for any over-length examples.
        candidates=list(encode(available[:TRAIN_ROWS+128],tokenizer))
        eligible=[r for r in candidates if len(r['tokens'])<=MAX_TRAIN_CONTEXT and len(r['tokens'])-r['answer_start']<=MAX_NEW]
        assert len(eligible)>=TRAIN_ROWS
        selected=eligible[:TRAIN_ROWS]
        stage_dev=rng.sample([row for row in dev if identity(row) not in test_text],DEV_ROWS)
        evaluated={'dev':list(encode(stage_dev,tokenizer)), 'test':list(encode(test,tokenizer))}
        longest=max(longest,max(row['answer_start'] for values in evaluated.values() for row in values))
        stats=dict(train_total=len(train),dev_total=len(dev),test_total=len(test),train_overlap_removed=len(train)-len(available),
                   dev_overlap_removed=sum(identity(row) in test_text for row in dev),train_rows=len(selected),dev_rows=len(stage_dev),
                   input_tokens=sum(len(r['tokens']) for r in selected),supervised_tokens=sum(len(r['tokens'])-r['answer_start'] for r in selected),
                   dev_max_gold=max(len(r['tokens'])-r['answer_start'] for r in evaluated['dev']),
                   test_max_gold=max(len(r['tokens'])-r['answer_start'] for r in evaluated['test']),
                   training_entities=Counter(span[2] for row in selected for span in row['spans']))
        assert set(stats['training_entities'])==TYPES
        for split,values in dict(train=selected,**evaluated).items():
            assert len(values)==len({r['id'] for r in values})
            name=f'{split}_{code}.jsonl'; files[name]=save(name,values)
        coverage[code]=stats
    config=dict(max_model_len=max(MAX_TRAIN_CONTEXT,((longest+MAX_NEW+255)//256)*256),max_new_tokens=MAX_NEW)
    result=dict(source=source,order=ORDER,stages=STAGES,steps=STEPS,seed=SEED,names=coverage,files=files,
                prompt=PROMPT,prefill=PREFILL,max_train_context=MAX_TRAIN_CONTEXT,generation=config,
                longest_eval_prompt_tokens=longest,scorer='seqeval 1.2.2 default entity-level micro-F1',prepare_sha256=sha(__file__))
    atomic_json(DATA/'MANIFEST.json',result)
    atomic_json(ROOT/'DATA_MANIFEST.json',result)
    print(json.dumps({'coverage':coverage,'inference':config},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
