# NeuroLens `engine/neurons` 模块详解

> 面向计算机专业学生，深度解析基于 NeuroBreak 论文的安全神经元定位系统

---

## 目录

1. [整体功能概述](#1-整体功能概述)
2. [模块架构与依赖关系](#2-模块架构与依赖关系)
3. [各模块详细解析](#3-各模块详细解析)
   - 3.1 `data_loaders.py` — 数据加载器
   - 3.2 `snip_scorer.py` — SNIP 分数计算
   - 3.3 `safety_identifier.py` — 安全神经元识别
   - 3.4 `utility_identifier.py` — 效用神经元识别
   - 3.5 `salad_safety_dataset.py` — SALAD 安全数据集
   - 3.6 `parameter_alignment.py` — 参数对齐分析
   - 3.7 `activation_projection.py` — 激活投影分析
   - 3.8 `quadrant_classification.py` — 功能象限分类
   - 3.9 `gradient_dependency.py` — 梯度依赖分析
4. [数据流与输入输出总览](#4-数据流与输入输出总览)
5. [核心算法公式汇总](#5-核心算法公式汇总)
6. [使用流程示例](#6-使用流程示例)

---

## 1. 整体功能概述

`engine/neurons` 模块实现了 NeuroBreak 论文中的**安全神经元定位（Safety Neuron Localization）** 核心功能。其目标是：

> 在大语言模型（LLM）中，精确定位哪些神经元（或称 "hidden state dimensions / MLP hidden units"）负责安全对齐（safety alignment）行为，从而为模型的安全编辑（safety editing）提供依据。

### 研究背景

- 大语言模型在预训练中会从海量数据中学习有害内容模式
- 安全对齐（RLHF/RAFT 等）并不能彻底消除这些有害模式，只在表层做了抑制
- **关键洞察**：有害内容模式以神经元的激活形式存在于模型的隐藏层中
- **解决思路**：找到这些"问题神经元"，通过参数编辑技术（如 ROME/EMMET）将其修改为安全行为

### 核心任务

本模块围绕以下步骤构建完整的分析流程：

```
原始LLM权重
    │
    ├─[Step 1] SNIP分数计算 ──────→ 找出对损失影响最大的神经元
    │
    ├─[Step 2] 安全神经元识别 S(q) ─→ 在良性数据集上识别安全相关神经元
    │
    ├─[Step 3] 效用神经元识别 U(p) ─→ 在通用任务数据集上识别功能神经元
    │
    ├─[Step 4] 专属安全神经元 D(p,q) = S(q) \ U(p) ─→ 去除功能重叠部分
    │
    ├─[Step 5] 参数对齐 S_i^k ──────────→ 神经元权重方向与毒性向量对齐度
    │
    ├─[Step 6] 激活投影 A_i^k ──────────→ 神经元激活在毒性方向的投影
    │
    ├─[Step 7] 功能象限分类 ──────────────→ 四象限: S+A+, S-A+, S+A-, S-A-
    │
    └─[Step 8] 梯度依赖分析 ──────────────→ 神经元间因果关系图
```

---

## 2. 模块架构与依赖关系

```
__init__.py                 # 对外统一导出所有公开API

data_loaders.py             # 底层数据加载（其他模块的输入源）
     │
     ├── HiddenStateDataset       # 加载 .hs.npy 隐藏态文件
     ├── ProbeModel              # 探针模型类（Linear(hidden_dim, 2)）
     └── load_probe_toxic_vectors()  # 加载探针输出的毒性向量

snip_scorer.py              # 核心打分引擎
     │
     ├── compute_snip_scores()        # 基础 SNIP（单样本批次）
     ├── compute_snip_scores_batch()   # 便捷包装（直接传入文本列表）
     ├── rank_and_annotate_snip_scores()   # 排序 + 加排名/百分位
     ├── select_top_percent_neurons()  # 按比例截断
     ├── load_probe_toxic_vectors_from_snip()  # 加载探针毒性向量
     ├── get_recommended_analysis_layers()  # 根据 CV 准确率推荐分析层
     ├── compute_probe_guided_snip_scores()  # 探针增强版 SNIP
     ├── compute_layer_specific_snip_scores() # 分层 SNIP
     └── get_probe_quality_report()   # 探针质量报告

safety_identifier.py         # 安全神经元识别
     │
     ├── identify_safety_neurons()        # 识别 S(q)
     ├── default_safety_loss_fn()        # 安全场景的 CE Loss
     ├── get_dedicated_safety_neurons()  # 计算 D(p,q) = S(q) \ U(p)
     ├── identify_safety_neurons_with_probe_guidance()  # 探针增强版
     ├── get_layer_weights_for_safety()  # 层质量权重
     └── select_safety_neurons_by_layer_quality()  # 按层质量筛选

utility_identifier.py        # 效用神经元识别
     │
     ├── identify_utility_neurons()   # 识别 U(p)
     ├── default_utility_loss_fn()   # 效用场景的 CE Loss（与安全相同）
     └── AlpacaJsonlDataset          # Alpaca 数据集读取工具

salad_safety_dataset.py      # SALAD 数据集支持
     │
     ├── SaladSafetyDataset           # 从SALAD中提取安全样本
     └── CombinedSaladSafetyDataset   # 组合多个 SALAD 子集

parameter_alignment.py       # 参数对齐分析（对应论文5.4节 S_i^k）
     │
     ├── compute_parameter_alignment()     # 计算余弦相似度
     ├── save_parameter_alignment()        # 保存结果
     ├── load_toxic_vectors_for_parameter_alignment()  # 加载毒性向量
     └── select_optimal_layers_for_alignment()  # 选择最优分析层

activation_projection.py     # 激活投影分析（对应论文5.4节 A_i^k）
     │
     ├── compute_activation_projection()   # 计算激活投影
     ├── load_toxic_vectors_from_probes() # 加载探针毒性向量
     └── get_best_toxic_vector_for_activation()  # 获取最优毒性向量

quadrant_classification.py   # 功能象限分类
     │
     ├── classify_neuron_quadrants()      # 四象限分类
     ├── get_quadrant_statistics()        # 统计信息
     ├── save_quadrant_classification()   # 保存结果
     ├── filter_neurons_by_quadrant()     # 按象限过滤
     ├── load_layer_quality_from_probes() # 加载层质量
     ├── prepare_quadrant_visualization_data()  # 可视化数据准备
     └── save_quadrant_visualization_data()   # 保存可视化数据

gradient_dependency.py       # 梯度依赖分析
     │
     ├── compute_gradient_dependency()    # 计算 G_{i,j}
     └── visualize_gradient_dependency()  # 可视化
```

---

## 3. 各模块详细解析

### 3.1 `data_loaders.py` — 数据加载器

#### 核心类

**`HiddenStateDataset`**

这是一个 PyTorch `Dataset` 实现，专门从 `outputs/hidden_states/` 目录加载预提取的隐藏态向量。

- **输入文件格式**：
  - `*.hs.npy` — NumPy 压缩文件，形状为 `(num_samples, num_layers, hidden_dim)`，通常如 `(N, 32, 4096)`
  - `*.idx.npy` — 每个隐藏态对应的原始索引映射
  - `labels/*.jsonl` — Qwen 标注的标签文件（Safe / Unsafe / Controversial）

- **输出数据格式**：
  - `__getitem__(idx)` 返回 `(torch.Tensor, label)`：
    - `torch.Tensor` 形状为 `(num_layers, hidden_dim)` 或指定 `target_layer` 时为 `(hidden_dim,)`
    - `label` 为 `0`（Safe）、`1`（Unsafe）或 `None`（无标签）

- **关键设计**：使用 memory-mapped file（mmap）只读加载，节省内存

```python
dataset = HiddenStateDataset("outputs/hidden_states", labels_dir="outputs/data_set_output/labels")
hs, label = dataset[0]  # hs: (32, 4096), label: 0/1
```

**`ProbeModel`**

一个简单的线性探针模型，结构为 `Linear(hidden_dim, 2)`：

- `weight[0]` = Safe 类的权重向量
- `weight[1]` = Unsafe 类的权重向量
- **毒性向量**（tox_vec）= `weight[1] - weight[0]`

```python
model = ProbeModel.load_from_checkpoint("outputs/probes/layer_0/probe.pt")
tox_vec = model.tox_vec()  # (hidden_dim,)
```

#### 关键函数

**`load_probe_toxic_vectors(probes_dir, num_layers=None)`**

从探针输出目录加载各层的毒性向量和元数据。**支持三种目录格式**：

| 格式 | 路径 | 加载内容 |
|------|------|----------|
| 新格式（engine/probes） | `outputs/probes/model_id/layer_XX/` | `probe.pt` + `toxic_vector.npz` + `metrics.json` |
| 旧格式（scripts） | `outputs/linear_probes/layers/layerXX/` | `layerXX.pt` + `metrics.json` |
| 聚合格式 | `outputs/toxicity_vectors/all_layers_toxicity_vectors.json` | JSON 中的向量数组 |

- **返回**：`Tuple[Dict[int, np.ndarray], Dict[int, Dict]]` = `(vectors, metadata)`
  - `vectors[layer_idx]` = 形状为 `(hidden_dim,)` 的 `np.ndarray`
  - `metadata[layer_idx]` = `{'cv_accuracy', 'std', 'val_roc_auc', ...}`

---

### 3.2 `snip_scorer.py` — SNIP 分数计算

这是整个模块的**核心打分引擎**，实现了 SNIP（Single-shot Network Pruning based on Connection Sensitivity）方法。

#### 核心公式（对应论文）

对于每个神经元 $i$，SNIP 重要性分数定义为：

$$I_i(x) = \left| w_i \cdot \Delta \mathcal{L}(x) \right|$$

其中：
- $w_i$ — 该神经元对应的参数向量
- $\Delta \mathcal{L}(x)$ — 该参数上的损失梯度

#### 两种结构化打分目标

**`score_target="down_proj_out"`（默认）**
- 将 MLP `down_proj` 的**每一行**（维度为 `out_features`）视为一个神经元
- 每一行有 `in_features` 个权重参数，组成向量 $w_i$
- 分数：$I_j(x) = \left| \sum_i w_{j,i} \cdot g_{j,i} \right|$（按行求点积再取绝对值）

**`score_target="mlp_intermediate"`**
- 将 FFN 中间维度（`intermediate_size`）视为神经元
- 每个神经元跨越 `gate_proj`、`up_proj`、`down_proj` 三个投影层

#### 关键函数

**`compute_snip_scores(model, tokenizer, dataset, device, loss_fn, ...)`**

```
输入:
  model       - LLM 模型（需支持 .model.layers 结构，即 Llama 架构）
  tokenizer   - AutoTokenizer
  dataset     - PyTorch Dataset，支持格式：
                {"text": "..."}                        → 全序列 LM loss
                {"prompt": "...", "response": "..."}  → 仅 response token 计算 loss
                {"input": {"prompt": "..."}, "output": "..."}  → Alpaca 格式
  device      - torch.device
  loss_fn     - 自定义损失函数，默认使用 CE Loss
  batch_size  - 批大小（默认 8）
  num_samples - 限制样本数（None=全部）
  score_target - "down_proj_out" 或 "mlp_intermediate"

输出:
  Dict[(layer_idx, neuron_idx), snip_score]
  示例: {(0, 1023): 0.342, (5, 512): 0.287, ...}
```

**`rank_and_annotate_snip_scores(snip_scores)`**

在 SNIP 分数基础上添加排名和百分位信息：

```python
输入:  {(0, 0): 0.1, (0, 1): 0.9, (1, 0): 0.5}
输出:  {(0, 1): {"score": 0.9, "rank": 1, "percentile": 33.3, ...},
        (1, 0): {"score": 0.5, "rank": 2, "percentile": 66.7, ...},
        (0, 0): {"score": 0.1, "rank": 3, "percentile": 100.0, ...}}
```

**`select_top_percent_neurons(scored_neurons, top_percent)`**

按比例截断，保留前 N% 神经元：

```python
输入: scored_neurons（带 rank 信息）, top_percent=0.005（0.5%）
输出: 前 0.5% 神经元字典
```

#### 显存优化策略

默认 `restrict_grad_to_mlp=True`，仅对 MLP 权重开启梯度计算，显著降低显存占用。

---

### 3.3 `safety_identifier.py` — 安全神经元识别

#### 核心概念

**安全神经元候选集** $S(q)$：在良性（benign）数据集上 SNIP 分数排名前 $q\%$ 的神经元。

- **推荐数据集**：Stanford Alpaca（52K 安全指令-响应对）
- **替代数据集**：SALAD 安全子集（`data/salad/`）
- **⚠️ 关键原则**：必须与效用神经元使用**不同的数据集**

#### 关键函数

**`identify_safety_neurons(model, tokenizer, benign_dataset, device, safety_threshold_q=0.005, ...)`**

```
输入:
  benign_dataset    - 安全参考数据集（Alpaca 或 SALAD）
  safety_threshold_q - q 值，默认 0.005（前 0.5%）

输出:
  Dict[(layer_idx, neuron_idx), {
      "score": float,       # SNIP 分数
      "rank": int,          # 全局排名（1=最高）
      "percentile": float,  # 百分位
      "layer": int,
      "neuron": int
  }]
```

**`get_dedicated_safety_neurons(safety_neurons, utility_neurons)`**

计算**专属安全神经元**：

$$D(p,q) = S(q) \setminus U(p)$$

即从安全候选集中排除效用神经元，保留真正"只负责安全、不负责通用任务"的神经元。

```python
输入: safety_neurons（S(q)）, utility_neurons（U(p)）
输出: 专属安全神经元字典

⚠️ 如果两者重叠率 > 50%，会输出警告（建议换用不同数据集）
```

---

### 3.4 `utility_identifier.py` — 效用神经元识别

#### 核心概念

**效用神经元集合** $U(p)$：在通用任务数据集上 SNIP 分数排名前 $p\%$ 的神经元。

- **推荐数据集**：Stanford Alpaca（`data/alpaca/alpaca_data.jsonl`）
- **也可使用**：CSQA / PIQA / RACE / MMLU 等通用任务数据集

#### 关键函数

**`identify_utility_neurons(model, tokenizer, utility_dataset, device, utility_threshold_p=0.001, ...)`**

```
输入:
  utility_dataset    - 通用任务数据集（JSONL 格式）
  utility_threshold_p - p 值，默认 0.001（前 0.1%）

输出: 格式同 identify_safety_neurons()
```

**`AlpacaJsonlDataset`**

读取 Alpaca JSONL 数据集的工具类：

```python
# 支持的格式：
{"input": {"prompt": "..."}, "output": "..."}
{"prompt": "...", "response": "..."}
{"prompt": "...", "output": "..."}
{"input": "...", "output": "..."}
{"text": "..."}
```

---

### 3.5 `salad_safety_dataset.py` — SALAD 安全数据集

#### 概述

SALAD 是一个多层次安全数据集，本模块从中提取**安全样本**用于安全神经元识别。**支持三种来源**：

| 数据源 | 提取规则 | 字段 |
|--------|----------|------|
| `defense_enhanced_set_train.jsonl` | 使用 `daugq`（防御增强问题） | `{"text": daugq}` |
| `mcq_set_train.jsonl` | 仅选 `gt == "A"`（安全答案） | `{"text": baseq}` |
| `base_evaluation.jsonl` | 仅选 `guard.verdict == "allow"` | `{"input": {...}, "output": "..."}` |

**`CombinedSaladSafetyDataset`** — 组合多个 SALAD 子集，并统一转换为 `{"text": "..."}` 格式，便于 SNIP 打分。

---

### 3.6 `parameter_alignment.py` — 参数对齐分析

#### 核心公式（对应论文5.4节 $S_i^k$）

$$S_i^k = \frac{w_{\text{down},i}^k \cdot w_{\text{toxic}}^k}{\|w_{\text{down},i}^k\| \cdot \|w_{\text{toxic}}^k\|}$$

其中：
- $w_{\text{down},i}^k \in \mathbb{R}^d$ — 第 $k$ 层 MLP `down_proj` 的第 $i$ 行（对应第 $i$ 个神经元）
- $w_{\text{toxic}}^k$ — 第 $k$ 层的**毒性向量**（来自线性探针：$w_{\text{toxic}} = w_{\text{unsafe}} - w_{\text{safe}}$）

**语义解释**：
- $S_i^k > 0$（S+）：神经元参数方向与毒性方向对齐 → 促进有害内容生成
- $S_i^k \leq 0$（S-）：神经元参数方向与毒性方向相反 → 有助于防御

#### 投影方法

由于 `down_proj` 权重形状为 `(hidden_dim, intermediate_size)`，需要将中间维度投影到 `hidden_dim` 空间。代码实现了两种方法：

| 方法 | 描述 | 精度 |
|------|------|------|
| `up_proj_transpose`（默认） | 使用 $up\_proj^T$ 投影，保持 MLP 语义结构 | 高（推荐） |
| `truncate` | 简单截取前 `hidden_dim` 个维度 | 低（丢失 71% 维度） |

#### 关键函数

**`compute_parameter_alignment(model, toxic_vectors, target_neurons=None, ...)`**

```
输入:
  model             - LLM 模型
  toxic_vectors     - 毒性向量，支持三种输入：
                       1. Dict[int, np.ndarray]  — 预加载的向量
                       2. str/Path（目录）         — 自动从 outputs/ 加载
                       3. .npz 文件路径
  target_neurons    - 目标神经元子集（None=分析所有神经元）
  projection_method - "up_proj_transpose" 或 "truncate"
  prefer_layer      - 优先使用的层

输出:
  Dict[(layer_idx, neuron_idx), {
      "cosine_similarity": float,   # [-1, 1] 余弦相似度
      "alignment_type": "S+" | "S-", # 对齐类型
      "neuron_weight_norm": float,
      "toxic_vector_norm": float,
      "toxic_layer": int,
      "probe_cv_accuracy": float,
  }]
```

---

### 3.7 `activation_projection.py` — 激活投影分析

#### 核心公式（对应论文5.4节 $A_i^k$）

$$A_i^k = a_{\text{down},i}^k \cdot \frac{w_{\text{toxic}}^k}{\|w_{\text{toxic}}^k\|}$$

其中 $a_{\text{down},i}^k \in \mathbb{R}^d$ 是第 $k$ 层第 $i$ 个神经元在最后一个 token 位置的激活向量。

#### 激活投影计算细节

MLP 的计算流程：
```
gate_proj(x)  → gate (intermediate_size)
up_proj(x)    → up   (intermediate_size)
intermediate  = SiLU(gate) * up  (intermediate_size)
down_proj(intermediate) → output (hidden_dim)
```

激活投影使用 `up_proj` 的转置将中间激活投影到 `hidden_dim` 空间：
```
a_down = intermediate @ up_proj^T  → (hidden_dim,)
A_i^k = a_down[i] · w_toxic_normalized
```

#### 成功 vs 失败 jailbreak 分离

根据论文要求，分别统计：
- **成功样本**（jailbreak 生效）：平均激活投影 `successful_mean`
- **失败样本**（jailbreak 被拦截）：平均激活投影 `failed_mean`

判断成功的字段依次检查：
1. `jailbreak_success`（顶层）
2. `asr_label == 1`（顶层）
3. `guard.jailbreak_success`（嵌套）
4. `guard.asr_label == 1`（嵌套）
5. `inference.guard.jailbreak_success`（更深嵌套）

#### 关键函数

**`compute_activation_projection(model, tokenizer, dataset, toxic_vectors, target_neurons, device, ...)`**

```
输入:
  dataset  - jailbreak 数据集，每个样本包含：
              - 文本字段（text/prompt/input）
              - 成功标志（jailbreak_success/asr_label/success）
  其他参数同 parameter_alignment

输出:
  Dict[(layer_idx, neuron_idx), {
      "successful_mean": float,      # 成功样本平均激活投影
      "failed_mean": float,          # 失败样本平均激活投影
      "successful_std": float,
      "failed_std": float,
      "successful_count": int,
      "failed_count": int,
      "activation_projection": float,  # = successful_mean（用于象限分类）
      "activation_diff": float,        # successful_mean - failed_mean
      "toxic_layer": int,
      "probe_cv_accuracy": float,
  }]
```

---

### 3.8 `quadrant_classification.py` — 功能象限分类

#### 四象限定义（对应论文5.4节）

基于参数对齐 $S_i^k$ 和激活投影 $A_i^k$ 的符号，将神经元分为四个象限：

| 象限 | 参数对齐 | 激活投影 | 含义 | 风险等级 |
|------|----------|----------|------|----------|
| **S+A+** | S+ (正) | A+ (正) | 毒性特征增强 — 参数和激活都促进毒性 | 🔴 最危险 |
| **S-A+** | S- (负) | A+ (正) | 良性特征抑制 — 参数抑制毒性但激活促进毒性 | 🟠 中等 |
| **S+A-** | S+ (正) | A- (负) | 毒性特征抑制 — 参数促进毒性但激活抑制毒性（防御生效） | 🟢 较安全 |
| **S-A-** | S- (负) | A- (负) | 良性特征增强 — 参数和激活都抑制毒性 | 🟢 最安全 |

#### 层质量分组

根据探针 CV 准确率将层分为五个等级：

| 等级 | CV 准确率 | 说明 |
|------|-----------|------|
| S | ≥ 93% | 最可靠的分析层 |
| A | 91%–93% | 高可靠 |
| B | 89%–91% | 中等可靠 |
| C | 85%–89% | 较低可靠 |
| D | < 85% | 不可靠 |

#### 关键函数

**`classify_neuron_quadrants(parameter_alignment, activation_projection, threshold_s=0.0, threshold_a=0.0, ...)`**

```
输入:
  parameter_alignment - 来自 parameter_alignment.py 的结果
  activation_projection - 来自 activation_projection.py 的结果
  threshold_s - 参数对齐阈值（默认0.0）
  threshold_a - 激活投影阈值（默认0.0）

输出:
  Dict[(layer_idx, neuron_idx), {
      "quadrant": "S+A+" | "S-A+" | "S+A-" | "S-A-",
      "alignment": float,          # cosine_similarity
      "activation_projection": float,
      "activation_diff": float,
      "successful_mean": float,
      "failed_mean": float,
      "alignment_type": "S+" | "S-",
      "activation_type": "A+" | "A-",
      "toxic_layer": int,
      "probe_cv_accuracy": float,
      "layer_group": str,          # S/A/B/C/D
  }]
```

---

### 3.9 `gradient_dependency.py` — 梯度依赖分析

#### 核心公式（对应论文5.4节 $G_{i,j}$）

$$G_{i,j} = \left| \frac{\partial a_{\text{down},i}^k}{\partial w_{\text{down},j}^{k-1}} \right|$$

其中：
- $a_{\text{down},i}^k$ — 第 $k$ 层 `down_proj` 第 $i$ 个神经元的激活值
- $w_{\text{down},j}^{k-1}$ — 第 $k-1$ 层 `down_proj` 第 $j$ 个神经元的权重参数

**物理意义**：衡量第 $k-1$ 层的第 $j$ 个神经元对第 $k$ 层第 $i$ 个神经元的影响强度（因果关系）。

#### 实现方法

1. 在目标神经元（`down_proj` 第 $i$ 个）的激活值上做反向传播
2. 捕获第 $k-1$ 层 `down_proj` 权重的梯度
3. 取梯度张量的 L2 范数作为关联强度
4. 对所有样本累加，最后按比例保留 top-k% 最强关联

#### 关键函数

**`compute_gradient_dependency(model, tokenizer, dataset, target_neurons, device, top_k=0.1, ...)`**

```
输入:
  target_neurons - 目标神经元集合（来自安全/效用识别步骤）
  top_k          - 保留前 k% 的最强关联（默认 0.1 = 10%）
  其他参数同 activation_projection

输出:
  Dict[(layer_idx, neuron_idx), {
      "upstream_neurons": List[Tuple[int, int]],  # 上游神经元列表
      "gradient_strengths": List[float],          # 对应的梯度强度
      "mean_gradient_strength": float,
      "max_gradient_strength": float,
  }]
```

---

## 4. 数据流与输入输出总览

### 4.1 完整数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                      数据准备阶段                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LLM 模型 (Qwen2.5-7B-Instruct 等)                               │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────┐    ┌─────────────────┐                       │
│  │ scripts/        │    │ engine/probes/  │                       │
│  │ extract_hidden_  │    │ linear_probe_   │                       │
│  │ states.py       │    │ balanced.py     │                       │
│  └────────┬────────┘    └────────┬────────┘                       │
│           │                      │                                 │
│           ▼                      ▼                                 │
│  outputs/hidden_states/    outputs/probes/model_id/               │
│  ├── *.hs.npy            │   ├── layer_XX/probe.pt                │
│  │   shape=(N,32,4096)    │   │   shape=(2, hidden_dim)            │
│  ├── *.idx.npy            │   ├── layer_XX/toxic_vector.npz        │
│  └── labels/*.jsonl       │   │   w_toxic: (hidden_dim,)           │
│      Safe/Unsafe          │   └── layer_XX/metrics.json           │
│                            │       val_acc, std, roc_auc          │
│                            ▼                                       │
│                      outputs/linear_probes/layers/                │
│                      └── layerXX/layerXX.pt (旧格式)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      分析执行阶段                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  data/alpaca/alpaca_data.jsonl ──────┐                           │
│  data/salad/*.jsonl              ────┼──→ identify_safety_neurons()│
│  CSQA/PIQA/RACE 数据              ───┴──→ identify_utility_neurons()│
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐         │
│  │              compute_snip_scores()                   │         │
│  │         SNIP 分数计算（核心打分引擎）                  │         │
│  │                                                       │         │
│  │  输入: LLM + tokenizer + Dataset + CE Loss            │         │
│  │  过程: 前向传播 → 计算 loss → 反向传播 → 捕获梯度      │         │
│  │  输出: {(layer, neuron): snip_score}                 │         │
│  └──────────────────────────────────────────────────────┘         │
│                           │                                       │
│          ┌────────────────┼────────────────┐                     │
│          ▼                ▼                ▼                     │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────┐        │
│  │ identify_      │  │ parameter_   │  │ activation_      │        │
│  │ safety_neurons │  │ alignment()  │  │ projection()     │        │
│  │ S(q)           │  │ S_i^k        │  │ A_i^k            │        │
│  └───────┬───────┘  └──────┬───────┘  └────────┬─────────┘        │
│          │                  │                   │                 │
│          ▼                  └─────────┬───────────┘                 │
│  ┌───────────────┐                  ▼                              │
│  │ get_dedicated_│          ┌──────────────────────┐             │
│  │ safety_neurons│          │ classify_neuron_      │             │
│  │ D(p,q)=S\U    │          │ quadrants()          │             │
│  └───────────────┘          │ 四象限分类结果        │             │
│                              └──────────┬───────────┘             │
│                                         │                         │
│                                         ▼                         │
│                              ┌──────────────────────┐            │
│                              │ gradient_dependency()│            │
│                              │ 梯度依赖关系图 G_{i,j}│            │
│                              └──────────────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      输出结果文件                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  outputs/neurons/                                               │
│  ├── snip_scores.json         # 原始 SNIP 分数                   │
│  ├── safety_neurons.json     # S(q) 安全神经元候选集             │
│  ├── utility_neurons.json    # U(p) 效用神经元候选集             │
│  ├── dedicated_safety.json    # D(p,q) = S(q) \ U(p)            │
│  ├── parameter_alignment.json # S_i^k 参数对齐结果               │
│  ├── activation_projection.json  # A_i^k 激活投影结果           │
│  ├── quadrant_classification.json # 四象限分类结果               │
│  ├── quadrant_visualization.json  # 可视化增强数据               │
│  └── gradient_dependency.json    # G_{i,j} 梯度依赖图           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 各文件输入输出对照表

| .py 文件 | 输入文件 / 数据 | 输出文件 / 数据 |
|----------|----------------|----------------|
| `data_loaders.py` | `outputs/hidden_states/*.hs.npy`<br>`outputs/data_set_output/labels/*.jsonl`<br>`outputs/probes/*/layer_XX/probe.pt` | `HiddenStateDataset` 对象<br>`ProbeModel` 对象<br>`Dict[layer, vector]` |
| `snip_scorer.py` | LLM 模型 + tokenizer + Dataset | `Dict[(layer, neuron), score]` |
| `safety_identifier.py` | LLM + tokenizer + Alpaca/SALAD Dataset | `Dict[(layer, neuron), info]`（S(q)） |
| `utility_identifier.py` | LLM + tokenizer + Alpaca JSONL | `Dict[(layer, neuron), info]`（U(p)） |
| `salad_safety_dataset.py` | `data/salad/raw/*.jsonl` | `SaladSafetyDataset` / `CombinedSaladSafetyDataset` 对象 |
| `parameter_alignment.py` | LLM 权重 + `toxic_vector.npz` 或探针目录 | `parameter_alignment.json` |
| `activation_projection.py` | LLM 权重 + tokenizer + jailbreak Dataset + 毒性向量 | `activation_projection.json` |
| `quadrant_classification.py` | `parameter_alignment.json` + `activation_projection.json` | `quadrant_classification.json`<br>`quadrant_visualization.json` |
| `gradient_dependency.py` | LLM 权重 + tokenizer + Dataset + 目标神经元 | `gradient_dependency.json` |

### 4.3 数据格式详解

#### 隐藏态文件 `.hs.npy`

```
形状: (num_samples, num_layers, hidden_dim)
    = (N, 32, 4096)   # Qwen2.5-7B 示例

dtype: float32
存储: NumPy .npy 压缩格式（memory-mapped 加载）
```

#### 毒性向量 `toxic_vector.npz`

```
w_toxic: np.ndarray, shape=(hidden_dim,) = (4096,)
    dtype: float32
    计算: w_toxic = probe.weight[1] - probe.weight[0]
          = Unsafe 类的权重 - Safe 类的权重

可选字段:
    b: float  (偏置项, 通常为 0.0)
    w_toxic_normalized: 归一化版本
```

#### 探针 `probe.pt`

```
checkpoint 包含:
    linear.weight:  shape=(2, hidden_dim) = (2, 4096)
        [0] = Safe 类的权重向量
        [1] = Unsafe 类的权重向量
    linear.bias:    shape=(2,)
```

#### `metrics.json`

```json
// 新格式（engine/probes/）
{
  "val_acc": 0.9376,
  "std_val_acc": 0.0048,
  "val_roc_auc": 0.98,
  "val_pr_auc": 0.95
}

// 旧格式（scripts/train_linear_probe_labels.py）
{
  "avg_val_acc": 0.92,
  "std_val_acc": 0.01,
  "avg_metrics": {
    "avg_val_acc": 0.92,
    "avg_val_s_acc": 0.95,
    "avg_val_h_acc": 0.89
  }
}
```

---

## 5. 核心算法公式汇总

| 公式 | 名称 | 所在模块 | 含义 |
|------|------|----------|------|
| $I_i(x) = \left| w_i \cdot \Delta \mathcal{L}(x) \right|$ | SNIP 分数 | 神经元对损失的敏感度 |
| $S_i^k = \frac{w_{\text{down},i}^k \cdot w_{\text{toxic}}^k}{\|w_{\text{down},i}^k\| \cdot \|w_{\text{toxic}}^k\|}$ | 参数对齐 | 神经元权重与毒性向量的余弦相似度 |
| $A_i^k = a_{\text{down},i}^k \cdot \frac{w_{\text{toxic}}^k}{\|w_{\text{toxic}}^k\|}$ | 激活投影 | 神经元激活在毒性方向的投影 |
| $G_{i,j} = \left| \frac{\partial a_{\text{down},i}^k}{\partial w_{\text{down},j}^{k-1}} \right|$ | 梯度依赖 | 神经元间的因果影响强度 |
| $w_{\text{toxic}} = w_{\text{unsafe}} - w_{\text{safe}}$ | 毒性向量 | 探针二分类权重差值 |
| $S(q) = \{ i \mid I_i^s \text{ 为 } I^s \text{ 中 top } q\% \}$ | 安全神经元候选集 | 良性数据上的高 SNIP 神经元 |
| $U(p) = \{ i \mid I_i^u \text{ 为 } I^u \text{ 中 top } p\% \}$ | 效用神经元集合 | 通用任务上的高 SNIP 神经元 |
| $D(p,q) = S(q) \setminus U(p)$ | 专属安全神经元 | 安全专用、不参与通用任务的神经元 |

---

## 6. 使用流程示例

### 完整分析流程

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine.neurons import (
    identify_safety_neurons, identify_utility_neurons,
    get_dedicated_safety_neurons,
    compute_parameter_alignment, save_parameter_alignment,
    compute_activation_projection,
    classify_neuron_quadrants, save_quadrant_classification,
    compute_gradient_dependency, visualize_gradient_dependency,
)

# 1. 加载模型
model_id = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 安全神经元识别 S(q)
from engine.neurons import AlpacaJsonlDataset
safety_ds = AlpacaJsonlDataset("data/alpaca/alpaca_data.jsonl")
safety_neurons = identify_safety_neurons(
    model, tokenizer, safety_ds, device,
    safety_threshold_q=0.005,  # 前 0.5%
    batch_size=4, num_samples=1000
)

# 3. 效用神经元识别 U(p)
utility_neurons = identify_utility_neurons(
    model, tokenizer, "data/alpaca/alpaca_data.jsonl", device,
    utility_threshold_p=0.001,  # 前 0.1%
    batch_size=4, num_samples=1000
)

# 4. 专属安全神经元 D(p,q) = S(q) \ U(p)
dedicated_safety = get_dedicated_safety_neurons(safety_neurons, utility_neurons)

# 5. 参数对齐 S_i^k
param_align = compute_parameter_alignment(
    model,
    toxic_vectors="outputs/probes/Qwen2.5-7B-Instruct/",
    target_neurons=dedicated_safety,
    projection_method="up_proj_transpose"
)
save_parameter_alignment(param_align, "outputs/neurons/")

# 6. 激活投影 A_i^k
from engine.neurons import HiddenStateDataset
jailbreak_ds = HiddenStateDataset("outputs/hidden_states")
activation_proj = compute_activation_projection(
    model, tokenizer, jailbreak_ds,
    toxic_vectors="outputs/probes/Qwen2.5-7B-Instruct/",
    target_neurons=dedicated_safety,
    device=device
)

# 7. 功能象限分类
quadrants = classify_neuron_quadrants(param_align, activation_proj)
save_quadrant_classification(quadrants, "outputs/neurons/")

# 8. 梯度依赖分析
grad_dep = compute_gradient_dependency(
    model, tokenizer, jailbreak_ds,
    target_neurons=dedicated_safety,
    device=device, top_k=0.1
)
visualize_gradient_dependency(grad_dep, "outputs/neurons/gradient_dependency.json")
```

---

## 附录：术语对照表

| 英文术语 | 中文术语 | 说明 |
|----------|----------|------|
| Safety Neuron | 安全神经元 | 对抗有害内容生成的关键神经元 |
| Utility Neuron | 效用神经元 | 负责模型通用功能（问答、推理等）的神经元 |
| SNIP (Single-shot Network Pruning) | 单次网络剪枝 | 基于连接敏感度的剪枝方法 |
| Toxicity Vector | 毒性向量 | 探针中 Unsafe 与 Safe 权重的差值 |
| Parameter Alignment $S_i^k$ | 参数对齐 | 神经元权重与毒性向量的余弦相似度 |
| Activation Projection $A_i^k$ | 激活投影 | 神经元激活在毒性方向的投影 |
| Gradient Dependency $G_{i,j}$ | 梯度依赖 | 神经元间的因果影响强度 |
| Benign Dataset | 良性数据集 | 不含有害内容的参考数据集 |
| Jailbreak |越狱攻击 | 绕过LLM安全限制的技术 |
| down_proj / up_proj / gate_proj | MLP 投影层 | FFN 中的三个线性变换层 |
| hidden_dim | 隐藏维度 | 模型每层输出的向量维度 |
| intermediate_size | 中间维度 | MLP 中间激活的维度（通常为 4×hidden_dim） |
