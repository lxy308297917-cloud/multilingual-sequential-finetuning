# 复现入口

在仓库根目录运行下列命令。点指标复算使用 Python 3.11 标准库。

```sh
python audit_results.py
```

训练与测评源码沿用已经运行过的 V2 路径关系。仓库提供协议、冻结计划、打包及评分入口、结果和代码哈希。机器路径集中为环境变量与相对路径。完整历史复跑需要原始上游版本、打包后的数据及起点 checkpoint。模型权重、优化器、原始题面、逐题生成和运行日志由实验者保存在自己的环境。历史编排脚本可能依赖服务器的 GPU 调度环境，下面的训练和测评 worker 可直接调用。

- `CL_MODEL` 是 Qwen3-4B Base 的本地完整目录。训练前按 `outputs/massive-cl-v1/DATA_MANIFEST.json` 检查文件哈希。
- `CL_EXTERNAL_ROOT` 默认为 `local`。配置和预处理代码有各自的上游输入要求。训练环境原来使用 Python 3.11、torch 2.6.0+cu118、transformers 4.57.6、peft 0.18.1。评测原运行使用独立的 vLLM 0.9.2 环境。
- 在 `work/followup-upstream/massive` 获取 MASSIVE 作者仓库并检出 `f966f21846043aabef9b0f974fa7970027f43738`。数据版本、官方评分协议及 NER 来源版本见 [实验协议](results/sequential-finetuning/v2/PROTOCOL_ZH.md)。
- 计划使用相对路径，运行时可在计划副本修改 `data`、`protocol` 和既有 checkpoint 路径。数据 `MANIFEST.json`、协议和顺序文件仍按原 SHA-256 校验。修改序列、模板或数据版本属于新的实验设置。

```sh
export PYTHONPATH="$PWD:$PWD/work/followup-upstream/python:$PWD/work/followup-upstream/massive/src"
export CL_MODEL=/path/to/Qwen3-4B
python work/continual-v2/lora_worker.py --help
python work/continual-v2/full_train.py --help
python work/continual-v2/eval_worker.py --help
python work/continual-v2/massive_eval_v2.py --help
```

以 NER 为例，`lora_worker.py PLAN train_swa.jsonl RUN --arm seq` 训练首阶段。后续阶段必须使用父目录中的 `step-064` 通过 `--parent` 传入，以继承 LoRA 权重和优化器。回放阶段选择计划中的 replay 日程。全参训练使用 `torchrun --standalone --nproc-per-node=4 work/continual-v2/full_train.py PLAN full-seq RUN`。MASSIVE 的全参 Replay 可按冻结计划传入已核验的训练起点。运行 GPU worker 前按执行机实际设备分配 `CUDA_VISIBLE_DEVICES`。历史脚本的预检与 checkpoint 哈希检查必须通过。

发布核验包括 Python 编译、worker 帮助入口、打包数据的匹配计划预检、阶段指标复算和仓库文件扫描。迁移后的多卡训练和 GPU 评估尚需从新仓库实际跑通。原始完整实验结果及已公开提交来自 [原实验汇总](https://github.com/lxy308297917-cloud/Multilingual-CL/pull/2)。
