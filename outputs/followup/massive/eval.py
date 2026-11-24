"""Generate MASSIVE outputs with vLLM and grade them using the authors' scorer."""
import argparse
import ast
from functools import cache
from math import sqrt
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from seqeval.metrics.sequence_labeling import get_entities

from data import LOCALES,MODEL,ROOT,UPSTREAM,decode,read_rows,sha,write_rows

OLD = ROOT.parents[1]/"sequential-lora"


@cache
def official_scorer():
    import sklearn.metrics as sklm
    from seqeval.metrics import f1_score

    path=UPSTREAM/"src/massive/utils/training_utils.py"
    tree=ast.parse(path.read_text(),filename=str(path))
    names={"eval_preds","convert_to_bio"}
    modules=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in modules}==names
    scope=dict(np=np,sklm=sklm,f1_score=f1_score,sqrt=sqrt)
    # Execute the two upstream function definitions unmodified, without importing old training APIs.
    exec(compile(ast.Module(body=modules,type_ignores=[]),str(path),"exec"),scope)
    return scope["convert_to_bio"],scope["eval_preds"]


def grade(cohort,generations,manifest):
    convert_to_bio, eval_preds = official_scorer()

    pred_int, gold_int, pred_slots, gold_slots = [],[],[],[]
    results = []
    for row,out in zip(cohort,generations,strict=True):
        intent,tags,error = decode(out["output"],row["utt"],set(manifest["intents"]),set(manifest["slots"]))
        pred_int.append(intent); gold_int.append(row["intent"])
        pred_slots.append(tags); gold_slots.append(row["tags"])
        pred_entities = set(get_entities(convert_to_bio(tags)))
        gold_entities = set(get_entities(convert_to_bio(row["tags"])))
        results.append({**out,"id":row["id"],"error":error,
                        "intent_correct":intent==row["intent"],
                        "exact":intent==row["intent"] and pred_entities==gold_entities,
                        "strict_exact":not error and intent==row["intent"] and pred_entities==gold_entities,
                        "tp":len(pred_entities&gold_entities),"fp":len(pred_entities-gold_entities),
                        "fn":len(gold_entities-pred_entities)})
    official = eval_preds(pred_int, gold_int, pred_slots, gold_slots, eval_metrics="all")
    tp,fp,fn=[sum(r[k] for r in results) for k in ("tp","fp","fn")]
    assert np.isclose(official["slot_micro_f1"],2*tp/(2*tp+fp+fn) if tp+fp+fn else 0)
    assert np.isclose(official["ex_match_acc"],np.mean([r["exact"] for r in results]))
    assert np.isclose(official["intent_acc"],np.mean([r["intent_correct"] for r in results]))
    return results,dict(official=official,intent_acc=official["intent_acc"],slot_micro_f1=official["slot_micro_f1"],
                        exact=official["ex_match_acc"],strict_exact=float(np.mean([r["strict_exact"] for r in results])),
                        errors=dict(Counter(r["error"] or "ok" for r in results)),
                        parse_failure=float(np.mean([r["error"] is not None for r in results])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("locales",choices=LOCALES,nargs="+")
    parser.add_argument("--adapter",type=Path)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()

    from vllm_compat import install
    install()
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest

    path=ROOT/"data/MANIFEST.json"
    manifest=json.loads(path.read_text())
    upstream_commit=__import__("subprocess").check_output(
        ["git","-C",str(UPSTREAM),"rev-parse","HEAD"],text=True).strip()
    assert upstream_commit==manifest["upstream_commit"]
    for key,digest in json.loads((OLD/"data/MANIFEST.json").read_text())["model"].items():
        assert sha(Path(MODEL)/key)==digest,key
    args.output.mkdir(parents=True)
    adapter=None
    if args.adapter:
        identity=json.loads((args.adapter/"DONE.json").read_text())
        assert identity["manifest_sha256"]==sha(path) and args.locales==[identity["locale"]]
        assert sha(args.adapter/"adapter/adapter_model.safetensors")==identity["weights_sha256"]
        adapter=LoRARequest(args.locales[0],1,str(args.adapter/"adapter"))
    gpu=os.environ["CUDA_VISIBLE_DEVICES"]
    assert gpu.startswith("GPU-") and "," not in gpu
    engine=LLM(model=MODEL,tokenizer=MODEL,dtype="bfloat16",tensor_parallel_size=1,
               max_model_len=1536,max_num_seqs=64,max_num_batched_tokens=8192,
               gpu_memory_utilization=.78,seed=42,generation_config="vllm",
               enable_prefix_caching=True,trust_remote_code=False,enforce_eager=True,
               enable_lora=adapter is not None,max_lora_rank=16)
    sampling=SamplingParams(temperature=0,top_p=1,max_tokens=192,seed=42,
                            stop_token_ids=[151645,151643],skip_special_tokens=True)
    for locale in args.locales:
        source=ROOT/"data"/f"dev_{locale}.jsonl"
        assert sha(source)==manifest["files"][source.name]
        cohort=read_rows(source)
        assert len(cohort)==2033
        if args.smoke:
            cohort=cohort[:12]
            gold=[{"output":row["answer"]} for row in cohort]
            _,audit=grade(cohort,gold,manifest)
            assert all(audit[key]==1 for key in ("intent_acc","slot_micro_f1","exact","strict_exact"))
        raw=[]
        start=time.monotonic()
        for off in range(0,len(cohort),128):
            chunk=cohort[off:off+128]
            batch=[{"prompt_token_ids": row["tokens"][:row["answer_start"]]} for row in chunk]
            responses=engine.generate(batch,sampling,lora_request=adapter,use_tqdm=False)
            for inp,response in zip(batch,responses,strict=True):
                assert inp["prompt_token_ids"]==list(response.prompt_token_ids)
                answer=response.outputs[0]
                raw.append(dict(output=answer.text,generated_tokens=len(answer.token_ids),finish=answer.finish_reason))
            print(json.dumps(dict(locale=locale,completed=len(raw),n=len(cohort),seconds=time.monotonic()-start)),flush=True)
        graded,metrics=grade(cohort,raw,manifest)
        pred=args.output/f"{locale}.jsonl"
        write_rows(pred,graded)
        result=dict(locale=locale,n=len(graded),smoke=args.smoke,manifest_sha256=sha(path),
                    adapter=identity if adapter else None,public_split="dev",gpu=gpu,seconds=time.monotonic()-start,
                    max_new_tokens=192,generation_batch=64,seed=42,inputs_sha256=sha(source),prediction_sha256=sha(pred),
                    length_stop=sum(x["finish"]=="length" for x in raw), **metrics)
        f=args.output/f"{locale}.tmp"
        f.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
        f.rename(args.output/f"{locale}.DONE.json")
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
