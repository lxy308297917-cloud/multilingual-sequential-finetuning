"""Measure the actual serialized parameter perturbation for every adapter."""
import json
from pathlib import Path
import sys
import torch
from safetensors.torch import load_file

out=Path(sys.argv[1])
lock=json.loads((out/"A_LOCK.json").read_text())
baseline=load_file(Path(lock["parents"][1])/"adapter/adapter_model.safetensors")
rows={}
reference=None
for arm in sorted((out/"checkpoints").iterdir()):
    weights=load_file(arm/"adapter/adapter_model.safetensors")
    group={}
    for key in baseline:
        module=key.split(".lora_")[0]
        square=torch.sum((baseline[key].double()-weights[key].double())**2).item()
        group[module]=group.get(module,0)+square
    norm={k:v**.5 for k,v in group.items()}
    dose=float(arm.name.rsplit("_",1)[1])
    rows[arm.name]=dict(l2=sum(group.values())**.5, module_l2=norm)
    scaled=torch.tensor(list(norm.values()),dtype=torch.float64)/dose
    if reference is None:
        reference=scaled
    assert torch.allclose(reference,scaled,rtol=1e-5,atol=2e-7),arm
(out/"A_REAL_PERTURBATIONS.json").write_text(json.dumps(rows,indent=2))
