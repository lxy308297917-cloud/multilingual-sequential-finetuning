"""Render cohort-specific Pareto plots in Chinese."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

out=Path(__file__).resolve().parent
font="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(font)
plt.rcParams.update({"font.family":font_manager.FontProperties(fname=font).get_name(),
                     "axes.unicode_minus":False,"font.size":12})
result=json.loads((out/"PARETO.json").read_text())
methods=["SeqLoRA","LoRA Replay","Full-FT 5e-5","Full-FT Replay 5e-5","Full-FT 1e-5","Full-FT Replay 1e-5（单阶段）"]
colors=["#087e8b","#7855a4","#be544b","#d18a26","#3475b2","#176240"]

fig,grid=plt.subplots(2,2,figsize=(13,9))
fig.subplots_adjust(top=.85,bottom=.2,hspace=.46,wspace=.29)
fig.text(.075,.955,"MASSIVE 最后一阶段的学习—保留边界",fontsize=24,weight="bold")
fig.text(.075,.905,"官方开发集每语 2,033 题。横、纵轴均为槽位 Micro-F1，向右上更好。",fontsize=14)
handles={}
for i,(label,x) in enumerate(list(result.items())[:2]):
    current=x["current"]; other=("sw-KE" if i==0 else "ur-PK")
    base=next(r for r in x["records"] if r["id"]==x["reference"])["scores"][current][0]
    for ax,target in zip(grid[i],(other,"old_macro")):
        ax.axvspan(base-1,base+1,color="#e4ebee",alpha=.8,zorder=0)
        for style,(m,c) in enumerate(zip(methods,colors)):
            pts=[r for r in x["records"] if r["method"]==m]
            if not pts:continue
            xs=[r["scores"][current][0] for r in pts]
            ys=[r["scores"][target][0] for r in pts]
            h,=ax.plot(xs,ys,"o-" if style<2 else "s--",label=m,color=c,alpha=.9,lw=1.8,ms=4.5)
            handles[m]=h
            for r,xx,yy in zip(pts,xs,ys):
                if style<2 and r["step"] in (94,125):
                    ax.annotate(str(r["step"]),(xx,yy),xytext=(4,5 if style==0 else -11),textcoords="offset points",fontsize=8,color=c)
        p=[r for r in x["records"] if r["id"] in x["frontier"][target]]
        for r in p:
            ax.scatter(r["scores"][current][0],r["scores"][target][0],s=100,facecolors="none",edgecolors="#202c3a",linewidth=1.3)
        ax.set_title(label+"｜"+("旧语 "+other[:2].upper() if target!="old_macro" else "四种旧语等权平均"),loc="left",weight="bold")
        ax.set_xlabel("当前语言 "+current[:2].upper()+" 槽位 F1（%）")
        ax.set_ylabel("旧语言槽位 F1（%）")
        ax.grid(alpha=.15)
        ax.spines[["top","right"]].set_visible(False)
fig.legend(handles.values(),handles.keys(),ncol=3,loc="lower center",bbox_to_anchor=(.5,.075),frameon=False,fontsize=11)
fig.text(.075,.046,"点为实际测过的检查点。数字为本阶段训练步数。空心圈标出同题点估计的非支配点。",fontsize=11)
fig.text(.075,.023,"浅色区为相对同顺序 SeqLoRA 终点，当前语言 F1 相差不超过 1 个百分点。线段只连接实测点。",fontsize=11)
for fmt in ("png","pdf"):fig.savefig(out/f"PARETO_DEV.{fmt}",dpi=170)

fig,grid=plt.subplots(2,5,figsize=(18,7),layout="constrained")
fig.suptitle("MASSIVE 逐语言学习—保留｜官方开发集每语 2,033 题",fontsize=18)
for i,(label,x) in enumerate(list(result.items())[:2]):
    current=x["current"]
    for ax,target in zip(grid[i],(l for l in (*x["records"][0]["scores"],) if l!=current)):
        for m,c in zip(methods,colors):
            pts=[r for r in x["records"] if r["method"]==m]
            if pts:
                ax.plot([r["scores"][current][0] for r in pts],[r["scores"][target][0] for r in pts],"o-",color=c,label=m,ms=4)
        ax.set_title(label+"  "+(target[:2].upper() if target!="old_macro" else "旧语均值"),fontsize=12)
        ax.set_xlabel("当前语 F1（%）",fontsize=10)
        ax.set_ylabel("旧语 F1（%）",fontsize=10)
        ax.grid(alpha=.18)
grid[0,0].legend(fontsize=7,frameon=False)
fig.savefig(out/"PARETO_PER_LANGUAGE.png",dpi=170)

x=result["独立训练侧"]
fig,ax=plt.subplots(figsize=(10,5.4),layout="constrained")
ax.set_title("撤回诊断｜当前 UR 与旧语言均值",loc="left",fontsize=19)
labels={"P0":"P0 学完 SW","P1":"P1 学完 UR","uniform":"整体撤回","functional":"功能方向","random":"随机方向","old_loss":"旧答案损失","format":"格式方向"}
base=next(r for r in x["records"] if r["id"]=="P1")["scores"]["ur-PK"][0]
ax.axvspan(base-1,base+1,color="#e4ebee",alpha=.8)
for i,r in enumerate(x["records"]):
    current=r["scores"]["ur-PK"][0];old=r["scores"]["old_macro"][0]
    ax.scatter(current,old,s=55)
    ax.annotate(labels[r["method"]]+(f' {r["step"]:g}%' if r["step"] else ""),(current,old),xytext=(5,8 if i%2 else -13),textcoords="offset points",fontsize=10)
ax.set(xlabel="UR 槽位 F1（%），训练侧机制集每语 512 题",ylabel="四种旧语平均槽位 F1（%）")
ax.set_xlim(45,77)
ax.grid(alpha=.15)
fig.savefig(out/"PARETO_ROLLBACK.png",dpi=170)
