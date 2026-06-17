# GRPO 24-Point Solver

基于 Qwen2.5-1.5B-Instruct 和 TRL GRPO 的 24 点游戏强化学习项目。目标是让模型按 R1 风格输出：

```text
<think>...</think><answer>...</answer>
```

其中 `<answer>` 内必须是合法算式：只使用题目给出的 4 个数字各一次、运算符 `+ - * /` 和括号，并且结果等于 24。

## 云端环境

推荐选择截图里的 PyTorch x86 镜像之一：

- `2.1.0-cuda12.1-py3.10.6-ubuntu22.04-x86_64`
- 或 `2.1.0-cuda12.1-py3.9.11-ubuntu22.04-x86_64`

Python 3.10 更推荐。镜像已含 CUDA/PyTorch 时，先不要重复安装 torch。

## 快速开始

```bash
git clone <your-github-repo-url>
cd 24point
python -m pip install -U pip
python -m pip install -r requirements.txt
```

`requirements.txt` 会以 editable 模式安装本项目，因此后续脚本可以直接导入 `twentyfour` 包。

如果云端 PyTorch 镜像是 `PyTorch-2.1.0`，不要安装最新版 `transformers/trl/peft`。本项目已固定一组兼容 PyTorch 2.1 的版本；如果之前已经装过最新版，执行：

```bash
python -m pip install --force-reinstall -r requirements.txt
```

先跑本地逻辑测试：

```bash
python -m pytest -q
```

快速检查数据集处理和 prompt：

```bash
python scripts/prepare_data.py --train-limit 8 --eval-limit 8 --out-dir data/processed
```

启动 GRPO 训练：

```bash
accelerate launch scripts/train_grpo.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --train_file data/processed/train_nlile_solvable.jsonl \
  --output_dir outputs/qwen2.5-1.5b-24point-grpo \
  --max_steps 800 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_generations 4 \
  --learning_rate 5e-6 \
  --bf16
```

显存紧张时可加：

```bash
  --use_peft --lora_r 16 --lora_alpha 32 --gradient_checkpointing
```

训练后评测：

```bash
python scripts/evaluate.py \
  --model_path outputs/qwen2.5-1.5b-24point-grpo \
  --split hard \
  --limit 100
```

如果训练阶段再次遇到 Hugging Face dataset cache 报错，确认先执行过 `scripts/prepare_data.py`，并在训练命令里保留 `--train_file data/processed/train_nlile_solvable.jsonl`。这样训练会直接读取本地 JSONL，不再重新解析远端数据集缓存。

## 项目结构

```text
src/twentyfour/
  data.py        数据集加载、字段归一化、prompt 构造
  prompts.py     R1 风格提示词
  rewards.py     GRPO 奖励函数
  verifier.py    24 点答案解析与程序化校验
scripts/
  prepare_data.py
  train_grpo.py
  evaluate.py
  infer.py
tests/
  test_verifier.py
```

## 数据集

训练默认使用 `nlile/24-game` 中可解样本；评测使用：

- `test-time-compute/game-of-24`：尤其适合取论文常用 `indices 900-1000` 难题。
- `nlile/24-game` 中不可解样本：检查模型是否会瞎编。

数据集会通过 Hugging Face `datasets` 在线下载。云服务器如果无法访问 Hugging Face，请先配置镜像源或提前缓存数据。

## 监控指标

训练脚本会记录可验证奖励：

- `answer_format_reward`：是否包含合法 `<answer>` 标签。
- `valid_expression_reward`：算式是否合法、数字是否刚好使用一次。
- `correct_reward`：算式是否等于 24。

定量分析建议报告：

- in-distribution success rate。
- hard split success rate。
- unsolvable split false-positive rate。
- 若启用 OOD，可加入 Countdown 任务扩展结果。

## GitHub 上传

本地初始化并提交：

```bash
git init
git add .
git commit -m "Initial GRPO 24-point solver"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
