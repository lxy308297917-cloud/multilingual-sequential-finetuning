"""Build fixed MTOP pilot data from the author flat slot annotations."""

import os as _release_os
import sys as _release_sys

from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
import random
from zipfile import ZipFile

from transformers import AutoTokenizer


HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]/'work/mtop-audit'
ARCHIVE=WORK/'mtop.zip'
OUT=WORK/'packed-v1'
MODEL=Path(_release_os.environ.get('CL_MODEL', 'Qwen3-4B'))
ORDER=('de','fr','es','hi','th')
SPLITS=('train','eval','test')
TRAIN_ROWS=1024
MAX_CONTEXT=2048
MAX_GENERATION=160
ARCHIVE_SHA='0c086500aed2ed48d3df383c91664b152e4a46e2adef6a635fa16d508ebaec0b'


def sha(path):
    with path.open('rb') as handle:
        return file_digest(handle,'sha256').hexdigest()


def read_source(archive,lang,split):
    entries, labels, issues, seen = {}, set(), Counter(), set()
    with archive.open(f'mtop/{lang}/{split}.txt') as records:
        for raw in records:
            sid,intent,slots,utterance,_,locale,_,serialized=raw.decode().rstrip('\r\n').split('\t')
            assert sid not in seen and intent.startswith('IN:') and locale[:2]==lang
            seen.add(sid)
            parsed=[]
            labels.add(intent)
            for raw_slot in filter(None,slots.split(',')):
                left,right,label=raw_slot.split(':',2)
                parsed.append((int(left),int(right),label))
                labels.add(label)
            try:
                tg=json.loads(serialized)
            except json.JSONDecodeError:
                issues['invalid json']+=1
                continue
            starts={span['start']:i for i,span in enumerate(tg['tokenSpans'])}
            ends={span['start']+span['length']:i+1 for i,span in enumerate(tg['tokenSpans'])}
            if any(left not in starts or right not in ends for left,right,_ in parsed):
                issues['unaligned span']+=1
                continue
            triples=[[starts[left],ends[right],label] for left,right,label in sorted(parsed)]
            if any(past[1]>after[0] for past,after in zip(triples,triples[1:])):
                issues['overlapping spans']+=1
                continue
            entries[sid]=dict(id=sid,locale=lang,intent=intent,slots=triples,words=tg['tokens'])
    return entries,labels,issues,seen


def prompt(intents, slots):
    return ('Classify the utterance intent and extract the top-level slots. Return a JSON object with keys '
            '"intent" and "slots". Each slot is [start,end,"TYPE"], with zero-based token positions, inclusive '
            'start and exclusive end. Use the canonical labels exactly. Output slots in token order. '
            'Use [] when there is no slot.\n'
            'Example tokens: 0:play 1:some 2:jazz\n'
            'Example JSON: {"intent":"IN:PLAY_MUSIC","slots":[[2,3,"SL:MUSIC_GENRE"]]}\n'
            'Canonical intents: '+', '.join(intents)+'\nCanonical slot types: '+', '.join(slots)+'\n\nTokens:\n')


def pack(tokenizer, instruction, row, training):
    words=' '.join(f'{idx}:{word}' for idx,word in enumerate(row['words']))
    context=tokenizer.apply_chat_template([dict(role='user',content=instruction+words)],
                                         tokenize=False,add_generation_prompt=True,enable_thinking=False)+'JSON:'
    label=' '+json.dumps(dict(intent=row['intent'],slots=row['slots']),ensure_ascii=False,separators=(',',':'))
    data=dict(row,answer=label)
    ids=tokenizer.encode(context,add_special_tokens=False)
    if training:
        suffix=tokenizer.encode(label+'<|im_end|>',add_special_tokens=False)
        assert tokenizer.encode(context+label+'<|im_end|>',add_special_tokens=False)==ids+suffix
        data.update(tokens=ids+suffix,answer_start=len(ids))
    else:
        data['tokens']=ids
    assert len(data['tokens'])<=MAX_CONTEXT-(0 if training else MAX_GENERATION),(row['locale'],row['id'],len(data['tokens']))
    return data


def build():
    assert sha(ARCHIVE)==ARCHIVE_SHA
    with ZipFile(ARCHIVE) as archive:
        data={split:{lang:read_source(archive,lang,split) for lang in (*ORDER,'en')} for split in SPLITS}
    unions=[set.union(*(x[3] for x in data[split].values())) for split in SPLITS]
    assert sum(map(len,unions))==len(set.union(*unions))
    labels=set.union(*(x[1] for x in data['train'].values()))
    intents=sorted(x for x in labels if x.startswith('IN:'))
    slots=sorted(x for x in labels if x.startswith('SL:'))
    assert {'IN:PLAY_MUSIC','SL:MUSIC_GENRE'}<=labels
    assert all(not x[2] for split in ('eval','test') for x in data[split].values())
    unseen={split:{lang:sorted(x[1]-labels) for lang,x in data[split].items() if x[1]-labels}
            for split in ('eval','test')}
    ids=sorted(set.intersection(*(set(data['train'][lang][0]) for lang in ORDER)))
    rng=random.Random(42)
    rng.shuffle(ids)
    assert len(ids)>=TRAIN_ROWS*len(ORDER)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    instruction=prompt(intents, slots)
    model=json.loads((HERE.parents[1]/'massive-cl-v1/DATA_MANIFEST.json').read_text())['model']
    OUT.mkdir()
    files, counts, train_ids, lengths={}, {}, {}, {}
    for split in SPLITS:
        for i,lang in enumerate(ORDER):
            original=data[split][lang][0]
            if split=='train':
                keys=ids[i*TRAIN_ROWS:(i+1)*TRAIN_ROWS]
                train_ids[lang]=keys
            else:
                keys=original
            title='dev' if split=='eval' else split
            path=OUT/f'{title}_{lang}.jsonl'
            longest=0
            with path.open('x') as stream:
                for k in keys:
                    row=pack(tokenizer,instruction,original[k],split=='train')
                    longest=max(longest,len(row['tokens']))
                    print(json.dumps(row,ensure_ascii=False,separators=(',',':')),file=stream)
            files[path.name]=sha(path)
            counts[path.name]=len(keys)
            lengths[path.name]=longest
    receipt=dict(archive_sha256=ARCHIVE_SHA,model=model,files=files,counts=counts,
                 order=ORDER,prompt=instruction,train_ids=train_ids,eligible_common_train_ids=len(ids),
                 intents=intents,slots=slots,held_out_labels_absent_from_training_ontology=unseen,max_used_tokens=lengths,
                 parser_sha256=sha(Path(__file__)))
    (OUT/'MANIFEST.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(files=counts,eligible=len(ids),intents=len(intents),slots=len(slots)),ensure_ascii=False))


if __name__=='__main__':
    build()
