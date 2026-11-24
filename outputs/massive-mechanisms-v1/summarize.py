"""Write experiment decisions and publication-ready dose curves."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(sys.argv[1])
a=root/"a-recovery"
b=root/"b-locked"
a_result=json.loads((a/"RESULTS.json").read_text())
b_result=json.loads((b/"RESULTS.json").read_text())
selection=json.loads((a/"A_SELECTION.json").read_text())
methods=selection["selected"]
f=methods["functional"]
metric="slot_f1"

def decide(positives,noninferiority):
    if all(x["lo"]>0 for x in positives) and all(x["lo"]>-1 for x in noninferiority):
        return "SUPPORTED"
    if any(x["hi"]<=0 for x in positives) or any(x["hi"]<-1 for x in noninferiority):
        return "NOT SUPPORTED"
    return "INCONCLUSIVE"

if f is None:
    verdict_a="INCONCLUSIVE"
else:
    v=a_result["contrasts"][f+"-P1"]
    valid=a_result["valid_subset"][f][metric]
    dose=f.rsplit("_",1)[1]
    checks=[v[k][metric] for k in ("sw-KE","old_macro")]
    checks.extend(a_result["contrasts"][f"{f}-{arm}_{dose}"]["old_macro"][metric]
                  for arm in ("uniform","random","old_loss"))
    checks += [a_result["contrasts"][f+"-"+k]["old_macro"][metric]
               for arm,k in methods.items() if arm in ("uniform","random","old_loss")
               and k and k.rsplit("_",1)[1]!=dose]
    verdict_a=decide(checks,[v["ur-PK"][metric]])
    if valid["delta"]<=0 and verdict_a=="SUPPORTED":
        verdict_a="INCONCLUSIVE"
v_b=[b_result["contrasts"]["parallel-"+k] for k in ("random","label")]
verdict_b=decide([x["old_macro"][metric] for x in v_b],[x["ur-PK"][metric] for x in v_b])

fig,ax=plt.subplots(figsize=(7,5),layout="constrained")
for method in methods:
    pairs=sorted((float(k.rsplit("_",1)[1]),v) for k,v in selection["scores"].items() if k.startswith(method+"_"))
    ax.plot([100*v["ur"] for _,v in pairs],[100*v["old"] for _,v in pairs],"o-",label=method)
    for dose,v in pairs:
        ax.annotate(f"{dose:.0%}",(100*v["ur"],100*v["old"]),xytext=(4,4),textcoords="offset points",fontsize=8)
ax.axvline(100*selection["baseline_ur"]-1,ls="--",color="gray",label="UR noninferiority threshold")
ax.set(xlabel="UR slot micro-F1 (%)",ylabel="Mean slot micro-F1 across four old languages (%)",
       title="MASSIVE checkpoint rollback — dose selection split")
ax.legend(loc="best",fontsize=8)
fig.savefig(a/"A_DOSE_CURVES.png",dpi=180)

def interval(x):
    return f"{x['delta']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}]"

body=["# MASSIVE 机制与回放结果","","[冻结设计、任务提示与指标](PROTOCOL_ZH.md)","","## A 方向撤回", "",
      f"判定 **{verdict_a}**。选中剂量见 A_SELECTION.json。"]
for name,result,folder in (("机制验证",a_result,a),("## B 历史回放",b_result,b)):
    if result is b_result:
        body.extend(["",name,"",f"判定 **{verdict_b}**。"])
    body.extend(["","| 条件 | DE | AR | VI | SW | UR | 五语均值 |","| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for k,v in result["scores"].items():
        body.append("| "+" | ".join([k,*(f"{x[metric]:.2f}" for x in v.values())])+" |")
    body.extend(["","| 配对差 | SW | 旧语均值 | UR |","| --- | ---: | ---: | ---: |"])
    for k,v in result["contrasts"].items():
        body.append("| "+" | ".join([k,*(interval(v[l][metric]) for l in ("sw-KE","old_macro","ur-PK"))])+" |")
    body.extend(["",f"[逐语言分数]({folder.name}/SCORES.csv) · [配对 95% 区间]({folder.name}/RESULTS.json)"])
body.extend(["","[剂量曲线](a-recovery/A_DOSE_CURVES.png) · [实际扰动范数](a-recovery/A_REAL_PERTURBATIONS.json)",
             "",f"A {'进入交换顺序复核' if verdict_a=='SUPPORTED' else '暂不开发功能方向训练约束'}。B {'进入分布混杂核验与多种子复核' if verdict_b=='SUPPORTED' else '停止把同源优先回放作为主方法'}。"])
(root/"RESULTS_ZH.md").write_text("\n".join(body)+"\n")
print(verdict_a,verdict_b)
