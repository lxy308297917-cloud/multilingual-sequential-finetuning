"""Run the original full-model MASSIVE scorer on the locked V2 data identity."""

import argparse
import importlib.util
from pathlib import Path
import sys

from common import OUTPUTS, dataset, read, schedule, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan",type=Path)
    p.add_argument("checkpoint",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("split",choices=("dev","test"))
    p.add_argument("languages",nargs="+")
    a=p.parse_args()
    plan=read(a.plan)
    data,receipt=dataset(plan)
    assert set(a.languages)<=set(plan["order"])
    card=read(a.checkpoint/"DONE.json")
    assert card["manifest_sha256"]==plan["manifest_sha256"]
    frozen=OUTPUTS/"massive-full-sft-v1/evaluate_full.py"
    before=sys.modules.pop("common")
    try:
        spec=importlib.util.spec_from_file_location("prior_full_eval",frozen)
        scorer=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scorer)
    finally:
        sys.modules["common"]=before
    scorer.DATA=data
    scorer.EVAL=tuple(plan["order"])
    scorer.evaluate.DATA=data
    scorer.evaluate.EVAL=scorer.EVAL
    scorer.evaluate.manifest=lambda:receipt
    scorer.evaluate.load=lambda name,lock:schedule(data,lock,name)
    sys.argv=[str(frozen),str(a.checkpoint),str(a.out),a.split,*a.languages]
    scorer.main()
    for l in a.languages:
        folder=a.out/card["stage"]/a.split
        score=read(folder/f"{l}.DONE.json")
        assert score["prediction_sha256"]==sha(folder/f"{l}.jsonl") and score["full_sft"]==card


if __name__=="__main__":
    main()
