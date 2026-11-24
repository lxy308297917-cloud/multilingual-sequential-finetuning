"""Use the audited Full-SFT trainer with UR and SW exchanged."""

import os as _release_os
import sys as _release_sys

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / 'massive-full-sft-v1/train_full.py'
DESTINATION = Path(_release_os.path.join(_release_os.environ.get("CL_EXTERNAL_ROOT", "local"), 'checkpoints/massive_full_sft_swap_v1_20261122'))

spec = importlib.util.spec_from_file_location('frozen_full_sft', REFERENCE)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
assert trainer.LANGUAGES == ('de-DE', 'ar-SA', 'vi-VN', 'sw-KE', 'ur-PK')
trainer.LANGUAGES = (*trainer.LANGUAGES[:3], *trainer.LANGUAGES[3:][::-1])
trainer.STAGES = tuple(f'{i:02}_{code}' for i, code in enumerate(trainer.LANGUAGES, 1))
trainer.DEST = DESTINATION

if __name__ == '__main__':
    trainer.main()
