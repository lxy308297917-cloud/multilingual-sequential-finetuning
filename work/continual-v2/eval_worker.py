"""Use the frozen NER/MTOP scorers with fixed V2 inputs and LoRA or full weights."""

import argparse
import importlib.util
from pathlib import Path
import sys

from common import MODEL, OUTPUTS, ROOT, WORKSPACE, dataset, read, save, schedule, sha


def import_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def evaluator(plan,data,receipt):
    if plan["benchmark"]=="ner":
        here=OUTPUTS/"masakhaner2-sequential-v1"
        before=sys.modules["common"]
        try:
            sys.modules["common"]=import_at("ner_common",here/"common.py")
            m=import_at("ner_evaluator",here/"evaluate.py")
        finally:
            sys.modules["common"]=before
        m.CFG=read(plan["protocol"])
        m.DATA=data
        m.ORDER=tuple(plan["order"])
        m.manifest=lambda:receipt
        m.load=lambda name,lock:schedule(data,lock,name)
        return m
    here=OUTPUTS/"continual-bench-next/mtop"
    sys.path.insert(0,str(here))
    try:
        m=import_at("mtop_evaluator",here/"evaluate.py")
    finally:
        sys.path.remove(str(here))
    m.CONFIG=read(plan["protocol"])
    m.ORDER=tuple(plan["order"])
    m.OUT=data
    m.manifest=lambda:receipt
    m.read_rows=lambda name,lock:schedule(data,lock,name)
    return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan",type=Path)
    p.add_argument("split",choices=("dev","test"))
    p.add_argument("output",type=Path)
    p.add_argument("langs",nargs="+")
    group=p.add_mutually_exclusive_group()
    group.add_argument("--lora",type=Path)
    group.add_argument("--full",type=Path)
    a=p.parse_args()
    plan=read(a.plan)
    data,receipt=dataset(plan)
    assert sha(plan["protocol"])==plan["protocol_sha256"] and set(a.langs)<=set(plan["order"])
    fitted=evaluator(plan,data,receipt)
    mounted=a.lora or a.full
    cert_path=mounted/"DONE.json" if mounted else None
    cert=read(cert_path) if cert_path else None
    if a.lora and cert["manifest_sha256"]!=plan["manifest_sha256"]:
        ref=next(v for v in plan["matched_lora"].values() if Path(v["path"])==a.lora)
        assert sha(cert_path)==ref["certificate_sha256"]
        old=WORKSPACE/"work/mtop-audit/packed-v1" if plan["benchmark"]=="mtop" else WORKSPACE/"work/masakhaner2/packed-v1"
        if "ner-screen" in a.lora.parts:
            old=ROOT/"ner-screen"
        assert sha(old/"MANIFEST.json")==cert["manifest_sha256"]
        if plan["benchmark"]=="ner":
            fitted.DATA=old
        else:
            fitted.OUT=old
    if a.full:
        assert cert["manifest_sha256"]==plan["manifest_sha256"] and not cert["smoke"]
        assert cert["benchmark"]==plan["benchmark"]
        assert all(sha(a.full/name)==d for name,d in cert["weights"].items())
        # Compatibility setup is done in the original evaluators before engine creation.

        from vllm_compat import install
        install()
        import vllm
        old_engine=vllm.LLM
        def engine(**kwargs):
            kwargs["model"]=str(a.full/"model")
            kwargs["tokenizer"]=str(MODEL)
            return old_engine(**kwargs)
        vllm.LLM=engine
    sys.argv=["evaluate.py",a.split,str(a.output),*a.langs]
    if a.lora:
        sys.argv.extend(["--checkpoint",str(a.lora)])
    fitted.main()
    for lang in a.langs:
        card=read(a.output/f"{lang}.DONE.json")
        assert card["input_sha256"]==receipt["files"][f"{a.split}_{lang}.jsonl"]
        assert card["prediction_sha256"]==sha(a.output/f"{lang}.jsonl")
    save(a.output/"MODEL_IDENTITY.json",dict(training=str(mounted) if mounted else None,
        certificate_sha256=sha(cert_path) if cert_path else None,plan_sha256=sha(a.plan),locales=a.langs,split=a.split))


if __name__=="__main__":
    main()
