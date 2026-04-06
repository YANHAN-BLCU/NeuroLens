# NeuroLens `engine/neurons` 模块 — 分步运行指南

> 面向计算机专业学生，基于 NeuroBreak 论文的安全神经元定位系统完整运行流程。
>
> 本文档涵盖从 SNIP 打分到梯度依赖分析的全部 8 个核心步骤，以及毒性向量提取的前置步骤。
>
> **适用模型**：Meta-Llama-3-8B-Instruct（或任何 HuggingFace 因果语言模型）
> **AutoDL 路径示例**： `/root/autodl-tmp/neurolens`
> **Windows 本地路径示例**：`D:\NeuroLens-master-0329\NeuroLens-master`

---

## 目录

- [前置准备：提取毒性向量 NPZ 文件](#前置准备提取毒性向量-npz-文件)
- [第一步：SNIP 打分 — 安全神经元 S(q)](#第一步snip-打分--安全神经元-sq)
- [第二步：SNIP 打分 — 效用神经元 U(p)](#第二步snip-打分--效用神经元-up)
- [第三步：计算专属安全神经元 D(p,q)](#第三步计算专属安全神经元-dpq)
- [第四步：参数对齐 $S_i^k$](#第四步参数对齐-s_ik)
- [第五步：激活投影 $A_i^k$](#第五步激活投影-a_ik)
- [第六步：四象限分类](#第六步四象限分类)
- [第七步：梯度依赖分析 $G_{i,j}$](#第七步梯度依赖分析-g_ij)
- [附录：关键参数说明](#附录关键参数说明)

---

## 前置准备：提取毒性向量 NPZ 文件

在运行第四步（参数对齐）和第五步（激活投影）之前，必须先将探针训练得到的线性探针权重提取为统一的 `.npz` 格式。

> **注意**：以下命令在 **AutoDL Linux** 环境下运行，行尾用 `\` 续行；Windows PowerShell 用 `` ` `` 续行。

```bash
# AutoDL / Linux 环境
python3 scripts/extract_toxic_vectors.py \
    --probes_dir /root/autodl-tmp/neurolens/outputs/linear_probes/layers \
    --output /root/autodl-tmp/neurolens/outputs/toxic_vectors/toxic_vectors.npz
```

```powershell
# Windows PowerShell 环境
python scripts/extract_toxic_vectors.py `
    --probes_dir outputs/linear_probes/layers `
    --output outputs/toxic_vectors/toxic_vectors.npz
```

**说明**：
- 该脚本从 `outputs/linear_probes/layers/`（或 `--probes_dir` 指定目录）下的**各层子目录**中读取 `probe.pt`。子目录命名支持 `layer_1` / `layer_12` 与 `layer01` / `layer32` 等常见格式（`train_probes_balanced` 等脚本产出多为 `layerNN`）。
- 若只传 `--probes_dir .../outputs/linear_probes` 且层目录实际在 `layers/` 下，脚本会自动尝试 `.../linear_probes/layers`。
- 不传 `--probes_dir` 时，会依次搜索 `outputs/linear_probes/layers`、`outputs/linear_probes`、`outputs/probes`。
- 每层目录内必须有 `probe.pt`；若仅有 `metrics.json` 而无权重文件，需先完成探针训练再运行本脚本。
- 从 `probe.pt` 中提取每层的毒性向量 $w_{\text{toxic}} = w_{\text{unsafe}} - w_{\text{safe}}$
- 输出文件为 `outputs/toxic_vectors/toxic_vectors.npz`，包含：
  - `vectors`：所有层 $w_{\text{toxic}}$ 的堆叠矩阵，形状为 `(num_layers, hidden_dim)`
  - `biases`：每层的偏置项
  - `layer_indices`：层索引数组
  - `layer_{i}`：第 $i$ 层独立的 $w_{\text{toxic}}$ 向量

**输出示例**：

```
[Dir] 探针目录: /root/autodl-tmp/neurolens/outputs/linear_probes/layers
[Dir] 发现 32 层: 1 ~ 32
[Extract] 逐层提取 w_toxic...
  Layer  1: dim=4096 |w|=37.6763  bias=0.0000
  Layer  2: dim=4096 |w|=35.2111  bias=0.0000
  ...
  Layer 32: dim=4096 |w|=3.7765  bias=0.0000
[Save] 毒性向量字典: outputs/toxic_vectors/toxic_vectors.npz (503.7 MB)
```

---

## 第一步：SNIP 打分 — 安全神经元 S(q)（保存全部神经元）

在良性（benign）数据集上计算**所有神经元**的 SNIP 分数，保存完整结果，后续可灵活调整阈值。

**数据集说明**：
- 使用 `base_set_train.jsonl` + `attack_enhanced_set_train.jsonl` 组合
- 标签文件位于 `outputs/data_set_output/labels/`，由模型对这两部分数据进行评估
- 脚本会自动过滤出 `label == "Safe"` 的样本（即模型**成功拒绝**有害请求的情况）

**标签文件**：
| 数据集 | 标签文件 | Safe 样本数 | 总样本数 |
|--------|---------|-----------|---------|
| `attack_enhanced_set_train.jsonl` | `attack_enhanced_outputs.jsonl` | 4177 | 5000 |
| `base_set_train.jsonl` (分段) | `base_set_outputs_*.jsonl` | 20424 | 21317 |

**准备标签文件**（仅需执行一次）：

由于 `base_set_train.jsonl` 的标签被分成了 4 个文件，需要先合并：

```bash
# AutoDL / Linux 环境
cat /root/autodl-tmp/neurolens/outputs/data_set_output/labels/base_set_outputs_0_4999.jsonl \
    /root/autodl-tmp/neurolens/outputs/data_set_output/labels/base_set_outputs_5000_9999.jsonl \
    /root/autodl-tmp/neurolens/outputs/data_set_output/labels/base_set_outputs_10000_14999.jsonl \
    /root/autodl-tmp/neurolens/outputs/data_set_output/labels/base_set_outputs_15000_21316.jsonl \
    > /root/autodl-tmp/neurolens/outputs/data_set_output/labels/base_set_outputs_merged.jsonl
```

```powershell
# Windows PowerShell 环境
Get-Content outputs/data_set_output/labels/base_set_outputs_0_4999.jsonl, outputs/data_set_output/labels/base_set_outputs_5000_9999.jsonl, outputs/data_set_output/labels/base_set_outputs_10000_14999.jsonl, outputs/data_set_output/labels/base_set_outputs_15000_21316.jsonl | Set-Content outputs/data_set_output/labels/base_set_outputs_merged.jsonl
```

合并后会生成 `outputs/data_set_output/labels/base_set_outputs_merged.jsonl`（共 21317 条标签，`original_index` 从 0 到 21316）。

**算法原理**：$I_i(x) = |w_i \cdot \Delta \mathcal{L}(x)|$，衡量每个神经元参数对语言模型损失的影响程度。

```bash
# AutoDL / Linux 环境
python3 scripts/run_safety_identifier_salad.py \
    --model_path /root/autodl-tmp/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
    --dataset_path /root/autodl-tmp/neurolens/data/salad/raw/base_set_train.jsonl \
                   /root/autodl-tmp/neurolens/data/salad/raw/attack_enhanced_set_train.jsonl \
    --label_paths /root/autodl-tmp/neurolens/outputs/data_set_output/labels/base_set_outputs_merged.jsonl \
                  /root/autodl-tmp/neurolens/outputs/data_set_output/labels/attack_enhanced_outputs.jsonl \
    --source_type text \
    --output_path /root/autodl-tmp/neurolens/outputs/neurons/safety_all_neurons_scores.json \
    --safety_threshold_q -1 \
    --batch_size 4 \
    --num_samples 2000
```

```powershell
# Windows PowerShell 环境
python scripts/run_safety_identifier_salad.py `
    --model_path "D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct" `
    --dataset_path data/salad/raw/base_set_train.jsonl data/salad/raw/attack_enhanced_set_train.jsonl `
    --label_paths outputs/data_set_output/labels/base_set_outputs_merged.jsonl outputs/data_set_output/labels/attack_enhanced_outputs.jsonl `
    --source_type text `
    --output_path outputs/neurons/safety_all_neurons_scores.json `
    --safety_threshold_q -1 `
    --batch_size 4 `
    --num_samples 2000
```

> **注意**：每个数据集必须对应一个标签文件，`original_index` 索引从 0 开始，与原始数据集的行号一一对应。

**参数说明**：
- `--source_type text`：直接提取 `question` 字段
- `--label_paths`：与 `--dataset_path` 一一对应，只保留 `label == "Safe"` 的样本
- `--safety_threshold_q -1`：设置为 -1 表示**不应用阈值**，保存所有神经元的完整 SNIP 分数
- 后续可使用 `--safety_threshold_q 0.005` 等值灵活筛选不同比例

**输出文件**：`outputs/neurons/safety_all_neurons_scores.json`（包含所有神经元及其分数、排名、百分位）

---

## 第二步：SNIP 打分 — 效用神经元 U(p)（保存全部神经元）

在通用任务数据集上计算**所有神经元**的 SNIP 分数，保存完整结果。

```bash
# AutoDL / Linux 环境
python3 scripts/run_utility_identifier.py \
    --model_name_or_path /root/autodl-tmp/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
    --alpaca_path /root/autodl-tmp/neurolens/data/alpaca/alpaca_data.jsonl \
    --output_path /root/autodl-tmp/neurolens/outputs/neurons/utility_all_neurons_scores.json \
    --utility_threshold_p -1 \
    --batch_size 4 \
    --num_samples 1000
```

```powershell
# Windows PowerShell 环境
python scripts/run_utility_identifier.py `
    --model_name_or_path "D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct" `
    --alpaca_path data/alpaca/alpaca_data.jsonl `
    --output_path outputs/neurons/utility_all_neurons_scores.json `
    --utility_threshold_p -1 `
    --batch_size 4 `
    --num_samples 1000
```

**输出文件**：`outputs/neurons/utility_all_neurons_scores.json`（包含所有神经元及其分数、排名、百分位）

**参数说明**：
- `--utility_threshold_p -1`：设置为 -1 表示**不应用阈值**，保存所有神经元的完整 SNIP 分数
- `--utility_threshold_p 0.001`：选择 SNIP 分数排名前 0.1% 的神经元（p=0.1%）

> **关键原则**：安全神经元和效用神经元必须使用**不同的数据集**，否则两者重叠率会过高（>50%），导致 $D(p,q)$ 失去意义。

---

## 第三步：筛选神经元 + 计算专属安全神经元 D(p,q)

从完整神经元分数中根据阈值筛选，然后计算专属安全神经元。

> **提示**：如果只想先保存所有神经元分数（不筛选），可以跳过阈值的指定，脚本会自动使用所有神经元。

### 方式一：一键完成（推荐）

直接传入所有神经元文件 + 阈值参数，一步完成筛选 + 计算：

```bash
# AutoDL / Linux 环境
python3 scripts/compute_dedicated_safety_neurons.py \
    --safety_all_neurons_path /root/autodl-tmp/neurolens/outputs/neurons/safety_all_neurons_scores.json \
    --utility_all_neurons_path /root/autodl-tmp/neurolens/outputs/neurons/utility_all_neurons_scores.json \
    --safety_threshold_q 0.005 \
    --utility_threshold_p 0.001 \
    --output_path /root/autodl-tmp/neurolens/outputs/neurons/dedicated_safety_neurons.json
```

```powershell
# Windows PowerShell 环境
python scripts/compute_dedicated_safety_neurons.py `
    --safety_all_neurons_path outputs/neurons/safety_all_neurons_scores.json `
    --utility_all_neurons_path outputs/neurons/utility_all_neurons_scores.json `
    --safety_threshold_q 0.005 `
    --utility_threshold_p 0.001 `
    --output_path outputs/neurons/dedicated_safety_neurons.json
```

### 方式二：先筛选，再计算

如果想分别控制筛选过程，可以使用辅助脚本：

**3.2.1 筛选安全神经元**

```bash
# AutoDL / Linux 环境
python3 scripts/select_neurons_by_threshold.py \
    --input_path /root/autodl-tmp/neurolens/outputs/neurons/safety_all_neurons_scores.json \
    --output_path /root/autodl-tmp/neurolens/outputs/neurons/safety_neurons_q0.005.json \
    --threshold_q 0.005
```

```powershell
# Windows PowerShell 环境
python scripts/select_neurons_by_threshold.py `
    --input_path outputs/neurons/safety_all_neurons_scores.json `
    --output_path outputs/neurons/safety_neurons_q0.005.json `
    --threshold_q 0.005
```

**3.2.2 筛选效用神经元**

```bash
# AutoDL / Linux 环境
python3 scripts/select_neurons_by_threshold.py \
    --input_path /root/autodl-tmp/neurolens/outputs/neurons/utility_all_neurons_scores.json \
    --output_path /root/autodl-tmp/neurolens/outputs/neurons/utility_neurons_p0.001.json \
    --threshold_q 0.001
```

```powershell
# Windows PowerShell 环境
python scripts/select_neurons_by_threshold.py `
    --input_path outputs/neurons/utility_all_neurons_scores.json `
    --output_path outputs/neurons/utility_neurons_p0.001.json `
    --threshold_q 0.001
```

**3.2.3 计算专属安全神经元**

```bash
# AutoDL / Linux 环境
python3 scripts/compute_dedicated_safety_neurons.py \
    --safety_neurons_path /root/autodl-tmp/neurolens/outputs/neurons/safety_neurons_q0.005.json \
    --utility_neurons_path /root/autodl-tmp/neurolens/outputs/neurons/utility_neurons_p0.001.json \
    --output_path /root/autodl-tmp/neurolens/outputs/neurons/dedicated_safety_neurons.json
```

```powershell
# Windows PowerShell 环境
python scripts/compute_dedicated_safety_neurons.py `
    --safety_neurons_path outputs/neurons/safety_neurons_q0.005.json `
    --utility_neurons_path outputs/neurons/utility_neurons_p0.001.json `
    --output_path outputs/neurons/dedicated_safety_neurons.json
```

### 常用阈值组合

| q (安全) | p (效用) | 说明 |
|---------|---------|------|
| `0.001` | `0.0005` | 严格筛选，神经元数量较少 |
| `0.005` | `0.001` | 平衡筛选（推荐默认） |
| `0.01` | `0.005` | 宽松筛选，神经元数量较多 |

**输出文件**：
- `outputs/neurons/safety_all_neurons_scores.json`：所有安全神经元完整分数
- `outputs/neurons/safety_neurons_q0.005.json`：筛选后的安全神经元（q=0.5%）
- `outputs/neurons/utility_all_neurons_scores.json`：所有效用神经元完整分数
- `outputs/neurons/utility_neurons_p0.001.json`：筛选后的效用神经元（p=0.1%）
- `outputs/neurons/dedicated_safety_neurons.json`：专属安全神经元 D(p,q)

---

## 第四步：参数对齐 $S_i^k$

计算每个神经元参数方向与毒性向量的余弦相似度。

**算法原理（对应论文5.4节）**：

$$S_i^k = \frac{w_{\text{down},i}^k \cdot w_{\text{toxic}}^k}{\|w_{\text{down},i}^k\| \cdot \|w_{\text{toxic}}^k\|}$$

- $w_{\text{down},i}^k$：第 $k$ 层 MLP `down_proj` 的第 $i$ 行（对应第 $i$ 个神经元）
- $w_{\text{toxic}}^k$：第 $k$ 层的毒性向量（来自线性探针）
- $S_i^k > 0$（S+）：神经元参数方向与毒性方向对齐 → 促进有害内容生成
- $S_i^k \leq 0$（S-）：神经元参数方向与毒性方向相反 → 有助于防御

```bash
# AutoDL / Linux 环境
python3 scripts/run_parameter_alignment.py \
    --model-path /root/autodl-tmp/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
    --toxic-vectors-path /root/autodl-tmp/neurolens/outputs/toxic_vectors/toxic_vectors.npz \
    --target-neurons-path /root/autodl-tmp/neurolens/outputs/neurons/dedicated_safety_neurons.json \
    --output-path /root/autodl-tmp/neurolens/outputs/neurons \
    --output-filename parameter_alignment.json \
    --load-in-4bit \
    --clear-cache
```

```powershell
# Windows PowerShell 环境
python scripts/run_parameter_alignment.py `
    --model-path "D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct" `
    --toxic-vectors-path outputs/toxic_vectors/toxic_vectors.npz `
    --target-neurons-path outputs/neurons/dedicated_safety_neurons.json `
    --output-path outputs/neurons `
    --output-filename parameter_alignment.json `
    --load-in-4bit `
    --clear-cache
```

**输出文件**：`outputs/neurons/parameter_alignment.json`

**显存不足时的优化选项**：

```bash
# AutoDL / Linux 环境
# 4-bit 量化（最节省显存，推荐）
python3 scripts/run_parameter_alignment.py \
    --model-path /root/autodl-tmp/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
    --toxic-vectors-path /root/autodl-tmp/neurolens/outputs/toxic_vectors/toxic_vectors.npz \
    --output-path /root/autodl-tmp/neurolens/outputs/neurons \
    --load-in-4bit \
    --clear-cache

# 8-bit 量化（精度更高，但显存占用更大）
python3 scripts/run_parameter_alignment.py \
    --model-path /root/autodl-tmp/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
    --toxic-vectors-path /root/autodl-tmp/neurolens/outputs/toxic_vectors/toxic_vectors.npz \
    --output-path /root/autodl-tmp/neurolens/outputs/neurons \
    --load-in-8bit \
    --clear-cache
```

```powershell
# Windows PowerShell 环境
# 4-bit 量化（最节省显存，推荐）
python scripts/run_parameter_alignment.py `
    --model-path "D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct" `
    --toxic-vectors-path outputs/toxic_vectors/toxic_vectors.npz `
    --output-path outputs/neurons `
    --load-in-4bit `
    --clear-cache

# 8-bit 量化（精度更高，但显存占用更大）
python scripts/run_parameter_alignment.py `
    --model-path "D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct" `
    --toxic-vectors-path outputs/toxic_vectors/toxic_vectors.npz `
    --output-path outputs/neurons `
    --load-in-8bit `
    --clear-cache
```

---

## 第五步：激活投影 $A_i^k$

计算神经元在 jailbreak 样本上的激活投影，分别统计成功和失败样本的激活分布。

**算法原理（对应论文5.4节）**：

$$A_i^k = a_{\text{down},i}^k \cdot \frac{w_{\text{toxic}}^k}{\|w_{\text{toxic}}^k\|}$$

其中 $a_{\text{down},i}^k$ 是第 $k$ 层第 $i$ 个神经元在最后一个 token 位置的激活向量。

**数据集字段要求**：

第五步同时需要文本字段和 jailbreak 成功/失败标签。

| 原始数据集 | 原始文本字段 | jailbreak 标签来源 | 标签字段映射规则 |
|-----------|------------|------------------|----------------|
| `base_set_train.jsonl` | `question` | `outputs/data_set_output/labels/base_set_outputs_*.jsonl`（4 个分片） | `Safe`/`Controversial` → `False`（失败），`Unsafe` → `True`（成功） |
| `attack_enhanced_set_train.jsonl` | `augq` | `outputs/data_set_output/labels/attack_enhanced_outputs.jsonl` | 同上 |

> 原始数据集的字段命名与脚本期望不完全一致，且缺少 jailbreak 标签字段，需通过预处理脚本统一转换。

**数据预处理**（将两个数据集关联标签后合并）：

> 预处理脚本会将 `base_set_train.jsonl` 和 `attack_enhanced_set_train.jsonl` 与对应标签文件按 `original_index` 合并，统一输出 `text` 和 `jailbreak_success` 布尔字段。

```bash
# AutoDL / Linux 环境
python3 scripts/preprocess_activation_dataset.py \
    --base-set /root/autodl-tmp/neurolens/data/salad/raw/base_set_train.jsonl \
    --base-labels /root/autodl-tmp/neurolens/outputs/data_set_output/labels/base_set_outputs_0_4999.jsonl \
                   /root/autodl-tmp/neurolens/outputs/data_set_output/labels/base_set_outputs_5000_9999.jsonl \
                   /root/autodl-tmp/neurolens/outputs/data_set_output/labels/base_set_outputs_10000_14999.jsonl \
                   /root/autodl-tmp/neurolens/outputs/data_set_output/labels/base_set_outputs_15000_21316.jsonl \
    --attack-enhanced-set /root/autodl-tmp/neurolens/data/salad/raw/attack_enhanced_set_train.jsonl \
    --attack-labels /root/autodl-tmp/neurolens/outputs/data_set_output/labels/attack_enhanced_outputs.jsonl \
    --output /root/autodl-tmp/neurolens/data/salad/raw/activation_projection_dataset.jsonl
```

```powershell
# Windows PowerShell 环境
python scripts/preprocess_activation_dataset.py `
    --base-set data/salad/raw/base_set_train.jsonl `
    --base-labels outputs/data_set_output/labels/base_set_outputs_0_4999.jsonl `
                 outputs/data_set_output/labels/base_set_outputs_5000_9999.jsonl `
                 outputs/data_set_output/labels/base_set_outputs_10000_14999.jsonl `
                 outputs/data_set_output/labels/base_set_outputs_15000_21316.jsonl `
    --attack-enhanced-set data/salad/raw/attack_enhanced_set_train.jsonl `
    --attack-labels outputs/data_set_output/labels/attack_enhanced_outputs.jsonl `
    --output data/salad/raw/activation_projection_dataset.jsonl
```

预处理后会生成合并数据集 `activation_projection_dataset.jsonl`（约 26316 条样本），各数据集的标签分布如下：

| 数据集 | Safe / Controversial（失败） | Unsafe（成功） | 合计 |
|--------|--------------------------|------------|------|
| `base_set_train.jsonl` | 20817 | 500 | 21317 |
| `attack_enhanced_set_train.jsonl` | 4330 | 669 | 4999 |
| **合并合计** | **25147** | **1169** | **26316** |

**运行第五步**：

```bash
# AutoDL / Linux 环境
python3 scripts/run_activation_projection.py \
    --model-path /root/autodl-tmp/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
    --toxic-vectors-path /root/autodl-tmp/neurolens/outputs/toxic_vectors/toxic_vectors.npz \
    --dataset-path /root/autodl-tmp/neurolens/data/salad/raw/activation_projection_dataset.jsonl \
    --target-neurons-path /root/autodl-tmp/neurolens/outputs/neurons/dedicated_safety_neurons.json \
    --output-path /root/autodl-tmp/neurolens/outputs/neurons \
    --output-filename activation_projection.json \
    --batch-size 4 \
    --num-samples 500 \
    --load-in-4bit \
    --clear-cache
```

```powershell
# Windows PowerShell 环境
python scripts/run_activation_projection.py `
    --model-path "D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct" `
    --toxic-vectors-path outputs/toxic_vectors/toxic_vectors.npz `
    --dataset-path data/salad/raw/activation_projection_dataset.jsonl `
    --target-neurons-path outputs/neurons/dedicated_safety_neurons.json `
    --output-path outputs/neurons `
    --output-filename activation_projection.json `
    --batch-size 4 `
    --num-samples 500 `
    --load-in-4bit `
    --clear-cache
```

**输出文件**：`outputs/neurons/activation_projection.json`

**参数说明**：
- `dataset-path`：使用预处理后的 `activation_projection_dataset.jsonl`（26316 行），包含 `text` 和 `jailbreak_success` 字段
- `num-samples`：分别限制成功和失败样本的数量，确保两种类型的样本都能被充分分析
- $A_i^k > 0$（A+）：神经元激活在毒性方向有正投影 → 促进有害内容生成
- $A_i^k \leq 0$（A-）：神经元激活在毒性方向有负投影 → 抑制有害内容

**显存不足时的优化选项**：

```bash
# AutoDL / Linux 环境
python3 scripts/run_activation_projection.py \
    --model-path /root/autodl-tmp/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
    --toxic-vectors-path /root/autodl-tmp/neurolens/outputs/toxic_vectors/toxic_vectors.npz \
    --dataset-path /root/autodl-tmp/neurolens/data/salad/raw/activation_projection_dataset.jsonl \
    --target-neurons-path /root/autodl-tmp/neurolens/outputs/neurons/dedicated_safety_neurons.json \
    --output-path /root/autodl-tmp/neurolens/outputs/neurons \
    --batch-size 2 \
    --max-length 1024 \
    --num-samples 300 \
    --load-in-4bit \
    --clear-cache
```

```powershell
# Windows PowerShell 环境
python scripts/run_activation_projection.py `
    --model-path "D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct" `
    --toxic-vectors-path outputs/toxic_vectors/toxic_vectors.npz `
    --dataset-path data/salad/raw/activation_projection_dataset.jsonl `
    --target-neurons-path outputs/neurons/dedicated_safety_neurons.json `
    --output-path outputs/neurons `
    --batch-size 2 `
    --max-length 1024 `
    --num-samples 300 `
    --load-in-4bit `
    --clear-cache
```

---

## 第六步：四象限分类

基于参数对齐 $S_i^k$ 和激活投影 $A_i^k$ 的符号，将神经元分为四个功能象限。

**四象限定义（对应论文5.4节）**：

| 象限 | 参数对齐 | 激活投影 | 含义 | 风险等级 |
|------|----------|----------|------|----------|
| **S+A+** | S+ (正) | A+ (正) | 毒性特征增强 — 参数和激活都促进毒性 | 最危险 |
| **S-A+** | S- (负) | A+ (正) | 良性特征抑制 — 参数抑制毒性但激活促进毒性 | 中等 |
| **S+A-** | S+ (正) | A- (负) | 毒性特征抑制 — 参数促进毒性但激活抑制毒性（防御生效） | 较安全 |
| **S-A-** | S- (负) | A- (负) | 良性特征增强 — 参数和激活都抑制毒性 | 最安全 |

```bash
# AutoDL / Linux 环境
python3 scripts/run_quadrant_classification.py \
    --parameter-alignment-path /root/autodl-tmp/neurolens/outputs/neurons/parameter_alignment.json \
    --activation-projection-path /root/autodl-tmp/neurolens/outputs/neurons/activation_projection.json \
    --output-path /root/autodl-tmp/neurolens/outputs/neurons \
    --output-filename quadrant_classification.json \
    --threshold-s 0.0 \
    --threshold-a 0.0 \
    --print-statistics
```

```powershell
# Windows PowerShell 环境
python scripts/run_quadrant_classification.py `
    --parameter-alignment-path outputs/neurons/parameter_alignment.json `
    --activation-projection-path outputs/neurons/activation_projection.json `
    --output-path outputs/neurons `
    --output-filename quadrant_classification.json `
    --threshold-s 0.0 `
    --threshold-a 0.0 `
    --print-statistics
```

**输出文件**：
- `outputs/neurons/quadrant_classification.json`：每个神经元的象限分类结果
- `outputs/neurons/quadrant_visualization.json`：可视化增强数据

**自定义阈值**（如果需要更严格的筛选）：

```bash
# AutoDL / Linux 环境
python3 scripts/run_quadrant_classification.py \
    --parameter-alignment-path /root/autodl-tmp/neurolens/outputs/neurons/parameter_alignment.json \
    --activation-projection-path /root/autodl-tmp/neurolens/outputs/neurons/activation_projection.json \
    --output-path /root/autodl-tmp/neurolens/outputs/neurons \
    --threshold-s 0.1 \
    --threshold-a 0.05
```

```powershell
# Windows PowerShell 环境
python scripts/run_quadrant_classification.py `
    --parameter-alignment-path outputs/neurons/parameter_alignment.json `
    --activation-projection-path outputs/neurons/activation_projection.json `
    --output-path outputs/neurons `
    --threshold-s 0.1 `
    --threshold-a 0.05
```

**过滤特定象限**：

```bash
# AutoDL / Linux 环境
python3 scripts/run_quadrant_classification.py \
    --parameter-alignment-path /root/autodl-tmp/neurolens/outputs/neurons/parameter_alignment.json \
    --activation-projection-path /root/autodl-tmp/neurolens/outputs/neurons/activation_projection.json \
    --output-path /root/autodl-tmp/neurolens/outputs/neurons \
    --filter-quadrants S+A+ S-A+
```

```powershell
# Windows PowerShell 环境
python scripts/run_quadrant_classification.py `
    --parameter-alignment-path outputs/neurons/parameter_alignment.json `
    --activation-projection-path outputs/neurons/activation_projection.json `
    --output-path outputs/neurons `
    --filter-quadrants S+A+ S-A+
```

---

## 第七步：梯度依赖分析 $G_{i,j}$

分析神经元之间的因果影响强度，追踪上游神经元对目标神经元的影响。

**算法原理（对应论文5.4节）**：

$$G_{i,j} = \left| \frac{\partial a_{\text{down},i}^k}{\partial w_{\text{down},j}^{k-1}} \right|$$

- $a_{\text{down},i}^k$：第 $k$ 层 `down_proj` 第 $i$ 个神经元的激活值
- $w_{\text{down},j}^{k-1}$：第 $k-1$ 层 `down_proj` 第 $j$ 个神经元的权重参数
- 物理意义：衡量第 $k-1$ 层的第 $j$ 个神经元对第 $k$ 层第 $i$ 个神经元的影响强度

**数据集字段要求**：

第七步仅需文本字段，不需要 jailbreak 成功/失败标签。但原始数据集中的字段命名与脚本期望不完全一致，需要通过预处理脚本统一转换。

支持的文本字段（按优先级）：`text` > `question` > `prompt` > `input`

| 原始数据集 | 原始文本字段 | 预处理后字段 | 脚本支持状态 |
|-----------|------------|------------|------------|
| `base_set_train.jsonl` | `question` | `text` | ✅ 直接支持 |
| `attack_enhanced_set_train.jsonl` | `augq` | `text` | ❌ 需预处理 |
| `base_evaluation.jsonl` | `text` | `text` | ✅ 直接支持 |

**数据预处理**（将两个数据集字段统一为 `text` 后合并）：

> 由于 `attack_enhanced_set_train.jsonl` 的文本字段为 `augq`（而非脚本支持的 `text`/`question`/`prompt`），需要先运行预处理脚本进行字段映射和合并。

```bash
# AutoDL / Linux 环境
python3 scripts/preprocess_gradient_dataset.py \
    --base-set /root/autodl-tmp/neurolens/data/salad/raw/base_set_train.jsonl \
    --attack-enhanced-set /root/autodl-tmp/neurolens/data/salad/raw/attack_enhanced_set_train.jsonl \
    --output /root/autodl-tmp/neurolens/data/salad/raw/gradient_dependency_dataset.jsonl
```

```powershell
# Windows PowerShell 环境
python scripts/preprocess_gradient_dataset.py `
    --base-set data/salad/raw/base_set_train.jsonl `
    --attack-enhanced-set data/salad/raw/attack_enhanced_set_train.jsonl `
    --output data/salad/raw/gradient_dependency_dataset.jsonl
```

预处理脚本会：
1. 读取 `base_set_train.jsonl`，将 `question` 字段映射为 `text`
2. 读取 `attack_enhanced_set_train.jsonl`，将 `augq` 字段映射为 `text`
3. 合并输出为 `gradient_dependency_dataset.jsonl`（共 26318 条样本）

**运行第七步**：

```bash
# AutoDL / Linux 环境
python3 scripts/run_gradient_dependency.py \
    --model-path /root/autodl-tmp/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
    --dataset-path /root/autodl-tmp/neurolens/data/salad/raw/gradient_dependency_dataset.jsonl \
    --target-neurons-path /root/autodl-tmp/neurolens/outputs/neurons/dedicated_safety_neurons.json \
    --output-path /root/autodl-tmp/neurolens/outputs/neurons/gradient_dependency \
    --batch-size 2 \
    --num-samples 200 \
    --top-k 0.1 \
    --max-length 512 \
    --clear-cache
```

```powershell
# Windows PowerShell 环境
python scripts/run_gradient_dependency.py `
    --model-path "D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct" `
    --dataset-path data/salad/raw/gradient_dependency_dataset.jsonl `
    --target-neurons-path outputs/neurons/dedicated_safety_neurons.json `
    --output-path outputs/neurons/gradient_dependency `
    --batch-size 2 `
    --num-samples 200 `
    --top-k 0.1 `
    --max-length 512 `
    --clear-cache
```

**输出文件**：
- `outputs/neurons/gradient_dependency/gradient_dependency.json`：每个目标神经元及其上游依赖关系
- `outputs/neurons/gradient_dependency/gradient_dependency_visualization.json`：可视化数据

**参数说明**：
- `top-k 0.1`：保留前 10% 最强的梯度关联
- `num-samples`：使用的样本数量
- `max-length 512`：最大序列长度

**低显存 GPU 推荐配置**：

```bash
# AutoDL / Linux 环境
python3 scripts/run_gradient_dependency.py \
    --model-path /root/autodl-tmp/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
    --dataset-path /root/autodl-tmp/neurolens/data/salad/raw/gradient_dependency_dataset.jsonl \
    --target-neurons-path /root/autodl-tmp/neurolens/outputs/neurons/dedicated_safety_neurons.json \
    --output-path /root/autodl-tmp/neurolens/outputs/neurons/gradient_dependency \
    --batch-size 2 \
    --num-samples 100 \
    --max-length 512 \
    --clear-cache
```

```powershell
# Windows PowerShell 环境
python scripts/run_gradient_dependency.py `
    --model-path "D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct" `
    --dataset-path data/salad/raw/gradient_dependency_dataset.jsonl `
    --target-neurons-path outputs/neurons/dedicated_safety_neurons.json `
    --output-path outputs/neurons/gradient_dependency `
    --batch-size 2 `
    --num-samples 100 `
    --max-length 512 `
    --clear-cache
```

---

## 附录：关键参数说明

### 阈值参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--safety_threshold_q` | `-1` | 保存所有神经元（第一步时使用 `-1` 表示不筛选） |
| `--safety_threshold_q` | `0.001` | 选择 top 0.1% 安全神经元 |
| `--safety_threshold_q` | `0.005` | 选择 top 0.5% 安全神经元 |
| `--safety_threshold_q` | `0.01` | 选择 top 1% 安全神经元 |
| `--utility_threshold_p` | `-1` | 保存所有神经元（第二步时使用 `-1` 表示不筛选） |
| `--utility_threshold_p` | `0.0005` | 选择 top 0.05% 效用神经元 |
| `--utility_threshold_p` | `0.001` | 选择 top 0.1% 效用神经元 |
| `--threshold-s` | `0.0` | 参数对齐阈值（用于象限分类，step 6） |
| `--threshold-a` | `0.0` | 激活投影阈值（用于象限分类，step 6） |

> **注意**：`compute_dedicated_safety_neurons.py` 中的阈值参数使用下划线格式（`--safety_threshold_q`），而 `select_neurons_by_threshold.py` 中使用 `--threshold_q`。

### 显存优化参数

| 参数 | 说明 |
|------|------|
| `--load-in-4bit` | 4-bit 量化，最节省显存（推荐） |
| `--load-in-8bit` | 8-bit 量化，精度更高但占用更多显存 |
| `--clear-cache` | 加载模型前清理 GPU 缓存 |
| `--batch_size` | 批大小，显存不足时减小到 2 |

### 数据集说明

| 数据集 | AutoDL 路径 | Windows 路径 | 行数 | 用途 |
|--------|-------------|--------------|------|------|
| base_evaluation.jsonl | `/root/autodl-tmp/neurolens/data/salad/raw/` | `data/salad/raw/` | 30359 | 备选 jailbreak 测试集（可直接使用，原生含 `guard.jailbreak_success` 标签） |
| alpaca_data.jsonl | `/root/autodl-tmp/neurolens/data/alpaca/` | `data/alpaca/` | ~52K | 效用神经元识别（通用任务数据集） |
| base_set_train.jsonl | `/root/autodl-tmp/neurolens/data/salad/raw/` | `data/salad/raw/` | 21317 | 安全神经元识别（需配合外部标签过滤 Safe 样本） |
| attack_enhanced_set_train.jsonl | `/root/autodl-tmp/neurolens/data/salad/raw/` | `data/salad/raw/` | 5001 | 安全神经元识别（需配合外部标签过滤 Safe 样本） |
| activation_projection_dataset.jsonl | `/root/autodl-tmp/neurolens/data/salad/raw/` | `data/salad/raw/` | 26316 | 激活投影（base_set_train + attack_enhanced_set 预处理合并，统一字段为 text + jailbreak_success） |
| gradient_dependency_dataset.jsonl | `/root/autodl-tmp/neurolens/data/salad/raw/` | `data/salad/raw/` | 26318 | 梯度依赖分析（base_set_train + attack_enhanced_set 预处理合并，统一字段为 text） |
| defense_enhanced_set_train.jsonl | `/root/autodl-tmp/neurolens/data/salad/raw/` | `data/salad/raw/` | 201 | 备选安全数据集（使用 daugq 字段，样本量少） |

### 输出文件对应关系

| 步骤 | 输出文件 | 描述 |
|------|----------|------|
| 前置准备 | `outputs/toxic_vectors/toxic_vectors.npz` | 各层毒性向量 |
| 第一步 | `outputs/neurons/safety_all_neurons_scores.json` | 所有安全神经元完整 SNIP 分数 |
| 第一步 | `outputs/neurons/safety_neurons_q0.005.json` | 筛选后的安全神经元（q=0.5%） |
| 第二步 | `outputs/neurons/utility_all_neurons_scores.json` | 所有效用神经元完整 SNIP 分数 |
| 第二步 | `outputs/neurons/utility_neurons_p0.001.json` | 筛选后的效用神经元（p=0.1%） |
| 第三步 | `outputs/neurons/dedicated_safety_neurons.json` | 专属安全神经元 D(p,q) |
| 第四步 | `outputs/neurons/parameter_alignment.json` | 参数对齐 S_i^k |
| 第五步 | `outputs/neurons/activation_projection.json` | 激活投影 A_i^k |
| 第六步 | `outputs/neurons/quadrant_classification.json` | 四象限分类结果 |
| 第七步 | `outputs/neurons/gradient_dependency/gradient_dependency.json` | 梯度依赖关系 G_{i,j} |

### 完整流水线（可选）

如果不想分步运行，也可以使用端到端一键运行脚本：

```bash
# AutoDL / Linux 环境
python3 scripts/run_neurobreak_pipeline.py \
    --model-path /root/autodl-tmp/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
    --salad-data /root/autodl-tmp/neurolens/data/salad \
    --alpaca-data /root/autodl-tmp/neurolens/data/alpaca/alpaca_data.jsonl \
    --output /root/autodl-tmp/neurolens/outputs/neurobreak_pipeline \
    --bf16 \
    --num-snip-samples 1000 \
    --safety_threshold_q 0.005 \
    --utility_threshold_p 0.001

# 从指定阶段断点续传
python3 scripts/run_neurobreak_pipeline.py \
    --model-path /root/autodl-tmp/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
    --salad-data /root/autodl-tmp/neurolens/data/salad \
    --alpaca-data /root/autodl-tmp/neurolens/data/alpaca/alpaca_data.jsonl \
    --output /root/autodl-tmp/neurolens/outputs/neurobreak_pipeline \
    --from-phase 4
```

```powershell
# Windows PowerShell 环境
python scripts/run_neurobreak_pipeline.py `
    --model-path "D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct" `
    --salad-data data/salad `
    --alpaca-data data/alpaca/alpaca_data.jsonl `
    --output outputs/neurobreak_pipeline `
    --bf16 `
    --num-snip-samples 1000 `
    --safety_threshold_q 0.005 `
    --utility_threshold_p 0.001

# 从指定阶段断点续传
python scripts/run_neurobreak_pipeline.py `
    --model-path "D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct" `
    --salad-data data/salad `
    --alpaca-data data/alpaca/alpaca_data.jsonl `
    --output outputs/neurobreak_pipeline `
    --from-phase 4
```
