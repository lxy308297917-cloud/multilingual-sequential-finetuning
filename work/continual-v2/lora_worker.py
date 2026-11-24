"""Run the existing Qwen3 LoRA kernel on one frozen Benchmark V2 schedule."""

import argparse
import importlib.util
from pathlib import Path
import sys

from common import MODEL, OUTPUTS, ROOT, dataset, read, schedule, sha


def old_trainer():
    before=sys.modules["common"]
    directory=OUTPUTS/"massive-cl-v1"
    sys.path.insert(0,str(directory))
    del sys.modules["common"]
    try:
        path=directory/"train.py"
        s=importlib.util.spec_from_file_location("frozen_lora_train",path)
        model=importlib.util.module_from_spec(s)
        s.loader.exec_module(model)
        return model
    finally:
        sys.modules["common"]=before
        sys.path.remove(str(directory))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan",type=Path)
    p.add_argument("schedule")
    p.add_argument("output",type=Path)
    p.add_argument("--arm",choices=("seq","replay","single","joint"),required=True)
    p.add_argument("--parent",type=Path)
    args=p.parse_args()
    plan=read(args.plan)
    data,receipt=dataset(plan)
    protocol=Path(plan["protocol"])
    assert sha(protocol)==plan["protocol_sha256"]
    config=read(protocol)
    assert config["learning_rate"]==plan["learning_rate"] and config["warmup_steps"]==plan["warmup_steps"]
    model=old_trainer()
    model.CFG=dict(config,progress_steps=[plan["steps_per_stage"]])
    model.DATA=data
    model.PROTOCOL=protocol
    model.manifest=lambda:receipt
    model.load=lambda name,manifest:schedule(data,manifest,name)
    base=read(OUTPUTS/"massive-cl-v1/DATA_MANIFEST.json")["model"]
    def verify(_):
        assert all(sha(MODEL/name)==digest for name,digest in base.items())
    model.current_model_verified=verify
    if args.parent:
        cert_path=args.parent/"DONE.json"
        cert=read(cert_path)
        expected=plan.get("matched_lora",{}).get(plan["order"][0])
        first=expected and cert["global_step"]==plan["steps_per_stage"]
        if first:
            assert Path(expected["path"])==args.parent and sha(cert_path)==expected["certificate_sha256"]
            assert cert["schedule_sha256"]==receipt["files"][f"train_{plan['order'][0]}.jsonl"]
            assert cert["protocol_sha256"]==plan["protocol_sha256"]
            original_read=model.read
            def mapped(path):
                if Path(path)==cert_path:
                    return dict(cert,manifest_sha256=plan["manifest_sha256"],trajectory=args.arm)
                return original_read(path)
            model.read=mapped
    sys.argv=[str(OUTPUTS/"massive-cl-v1/train.py"),args.schedule,str(args.output),"--trajectory",args.arm]
    if args.parent:
        sys.argv.extend(["--continue-from",str(args.parent)])
    model.main()


if __name__=="__main__":
    main()
