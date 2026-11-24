# 多语言顺序微调

围绕 Qwen3-4B Base，比较 MASSIVE、MasakhaNER 2.0 和 MTOP 在五语课程下的 Sequential、Replay 与 Joint。三个基准各有 LoRA 和全参微调六组主基线，另有 Base、逐语 Single 和部分替代课程及低学习率对照。

- [实验设计审计](DESIGN_AUDIT_ZH.md)
- [正式结果与热力图](results/sequential-finetuning/v2/README_ZH.md)
- [功能方向撤回和历史平行回放](results/massive-mechanisms-v1/RESULTS_ZH.md)
- [阶段内学习与保留的取舍](results/massive-pareto-v1/REPORT_ZH.md)
- [复现入口和边界](REPRODUCE.md)

在 Python 3.11 下运行 `python audit_results.py` 可从随附的 20 套官方测试矩阵复算 AP、BWT、FWT、习得、峰值遗忘和负 BWT。阶段矩阵和关键结果沿用原始测评产物。路径配置的迁移记录以 [原始源码哈希](SOURCE_SHA256.json) 为锚。历史训练的 checkpoint 及原始数据需由各自的来源获取。
