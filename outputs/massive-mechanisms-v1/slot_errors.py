"""Decompose generation errors into exact-boundary type and overlap errors."""
import json
from pathlib import Path
import sys
from collections import Counter

from seqeval.metrics.sequence_labeling import get_entities

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/"massive-cl-v1"))
from common import LANGUAGES, decode, load, manifest, read_rows
from evaluate import inherit_scorer

mode,out=sys.argv[1:]
out=Path(out)
mo=manifest()
convert_to_bio,_=inherit_scorer().__globals__["official_scorer"]()
split="evaluation" if mode=="B" else "mechanism"
rows={}
for p in sorted((out/split).iterdir()):
    by_language={}
    for lang in LANGUAGES:
        truth=load(f"dev_{lang}.jsonl",mo) if mode=="B" else read_rows(out/f"mechanism_{lang}.jsonl")
        prediction=read_rows(p/"dev"/f"{lang}.jsonl")
        stats=Counter()
        for datum,hyp in zip(truth,prediction,strict=True):
            assert datum["id"]==hyp["id"]
            _,tags,error=decode(hyp["output"],datum["utt"],set(mo["intents"]),set(mo["slots"]))
            if error:
                stats[error]+=1
                continue
            gold=set(get_entities(convert_to_bio(datum["tags"])))
            pred=set(get_entities(convert_to_bio(tags)))
            stats["valid"]+=1
            for label,start,end in sorted(pred-gold):
                if any((start,end)==(s,e) for _,s,e in gold):
                    stats["type_error"]+=1
                elif any(label==t and max(start,s)<=min(end,e) for t,s,e in gold):
                    stats["boundary_error"]+=1
                elif any(max(start,s)<=min(end,e) for _,s,e in gold):
                    stats["boundary_and_type_error"]+=1
                else:
                    stats["spurious_span"]+=1
            for _,start,end in gold-pred:
                if not any(max(start,s)<=min(end,e) for _,s,e in pred):
                    stats["missing_span"]+=1
        by_language[lang]=dict(stats)
    rows[p.name]=by_language
(out/"SLOT_ERRORS.json").write_text(json.dumps(rows,indent=2))
