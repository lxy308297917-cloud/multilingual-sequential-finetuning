"""Reuse the locked MASSIVE scoring protocol with a full model checkpoint."""

import os as _release_os
import sys as _release_sys

import argparse
from pathlib import Path
import sys


ROOT=Path(__file__).resolve().parent
REFERENCE=ROOT.parent/'massive-cl-v1'
UPSTREAM_DEPS=ROOT.parents[1]/'work/followup-upstream'
sys.path[:0]=[str(UPSTREAM_DEPS/'python'),str(UPSTREAM_DEPS/'massive/src')]
sys.path.insert(0,str(REFERENCE))
from common import DATA, EVAL, atomic_json, manifest, read, sha
import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('split',choices=['dev','test'])
    p.add_argument('locales',nargs='+',choices=EVAL)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    checkpoint=args.checkpoint.resolve(strict=True)
    cert=read(checkpoint/'DONE.json')
    assert cert['manifest_sha256']==sha(DATA/'MANIFEST.json') and cert['smoke']==args.smoke
    assert all(sha(checkpoint/name)==digest for name,digest in cert['weights'].items())
    assert len([name for name in cert['weights'] if name.endswith('.safetensors')])>=1

    def verify_same_tokenizer(receipt):
        for name,digest in receipt['model'].items():
            if name in ('tokenizer.json','tokenizer_config.json'):
                assert sha(Path(_release_os.environ.get('CL_MODEL', 'Qwen3-4B'))/name)==digest

    def write_full(path, payload):
        atomic_json(path,dict(payload,full_sft=cert,full_checkpoint=str(checkpoint)))

    from vllm import LLM as original_llm
    import vllm

    def engine_with_base_tokenizer(**kwargs):
        kwargs['tokenizer']=_release_os.environ.get('CL_MODEL', 'Qwen3-4B')
        return original_llm(**kwargs)

    vllm.LLM=engine_with_base_tokenizer
    evaluate.MODEL=str(checkpoint/'model')
    evaluate.current_model_verified=verify_same_tokenizer
    evaluate.atomic_json=write_full
    sys.argv=['evaluate.py',str(args.output),f"{cert['stage']}=base",'--split',args.split,
              '--locales',*args.locales,*(['--smoke'] if args.smoke else [])]
    evaluate.main()


if __name__=='__main__':main()
