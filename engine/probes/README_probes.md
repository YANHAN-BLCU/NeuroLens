# engine/probes 模块文档

> 面向计算机专业学生 | 线性探针（Linear Probe）分类器

---

## 一、整体功能概述

`engine/probes` 模块实现了一套**线性探针**（Linear Probe）分类系统，用于检测大语言模型（LLM）各层隐藏状态中是否包含"有害语义"。

### 核心思想

线性探针是一种极简的诊断工具：给定 LLM 对某文本生成的隐藏状态向量 **h**（维度 $d$），用线性分类器判断该文本是否包含有害意图。

分类公式：

$$
P(\text{有害} \mid \mathbf{h}) = \softmax(\mathbf{W}^\top \mathbf{h} + \mathbf{b})_1
$$

即：隐藏状态向量经过一个线性层（权重 $\mathbf{W} \in \mathbb{R}^{d \times 2}$，偏置 $\mathbf{b} \in \mathbb{R}^2$），再经 softmax 输出二分类概率。

### 模块文件

| 文件 | 用途 |
|------|------|
| `engine/probes/__init__.py` | 包入口，统一导出 API（实际从 `linear_probe_balanced` 导入） |
| `engine/probes/linear_probe_balanced.py` | 对外默认 API：`LinearProbe`、`extract_hidden_states`、`load_probe` 等（与 `scripts/train_probes_balanced.py` 对接） |
| `engine/probes/linear_probe.py` | 扩展训练管线：`HiddenStateDataset`、`train_layer_probes`、`save_probes`（可写出 `toxic_vector.npz` 等；不由 `__init__.py` 默认导出） |

---

## 二、`linear_probe.py` 详解（分层训练管线）

### 2.1 `HiddenStateDataset`

```31:84:engine/probes/linear_probe.py
class HiddenStateDataset(Dataset):
    def __init__(self, hidden_states: List[np.ndarray], labels: List[int], preload_to_tensor: bool = True):
        self.labels = labels
        if preload_to_tensor:
            self.hidden_states = [
                torch.from_numpy(hs).float() if isinstance(hs, np.ndarray) else torch.FloatTensor(hs)
                for hs in hidden_states
            ]
            self.labels_tensor = torch.LongTensor(labels)
        else:
            self.hidden_states = hidden_states
            self.labels_tensor = None
```

PyTorch `Dataset` 封装类，将 NumPy 隐藏状态数组列表包装为可迭代的数据集，供 `DataLoader` 使用。

- **输入格式**：每个元素为 `(hidden_dim,)` 的 1D NumPy 数组 + 对应标签（`0`=安全，`1`=有害）
- **优化**：`preload_to_tensor=True` 时在构造阶段就转为 Tensor，避免 `__getitem__` 每次重复转换
- **输出格式**：每次 `__getitem__` 返回 `{"hidden_states": Tensor, "label": LongTensor}`

---

### 2.2 `LinearProbe`（训练用，含多种初始化与 `get_toxic_vector`）

```87:155:engine/probes/linear_probe.py
class LinearProbe(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0, init_method: str = "xavier"):
        self.linear = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self._init_weights(init_method)
```

线性分类模型，结构为：

```
Dropout(可选) → Linear(hidden_dim, 2) → softmax
```

关键方法：

| 方法 | 说明 |
|------|------|
| `forward(x)` | 输入 `(batch, hidden_dim)`，返回 `(batch, 2)` logits |
| `predict_proba(x)` | 返回 `(batch, 2)` 概率分布 |
| `predict(x)` | 返回 `(batch,)` 类别索引 |
| `get_toxic_vector()` | 取出有害类权重向量 `w_toxic` 和偏置 `b`（用于计算内积分数） |

权重初始化支持 Xavier / Kaiming / Normal 三种方法，默认为 Xavier uniform。

---

### 2.3 `extract_hidden_states`

```203:293:engine/probes/linear_probe.py
def extract_hidden_states(
    model, tokenizer, texts, device,
    max_length: int = 512, batch_size: int = 8,
    pooling_method: str = "mean",
) -> List[np.ndarray]:
```

**功能**：将原始文本喂入 LLM，提取模型每一层的隐藏状态。

**处理流程**：

1. 批量分词（padding + truncation）
2. 调用 `model(output_hidden_states=True)`，获取所有层的隐藏状态
3. 对每一样本，遍历每一层，应用池化方法得到该层的固定维向量

**三种池化策略**：

| 方法 | 说明 |
|------|------|
| `"mean"` | 对所有非 padding token 取**平均池化**（默认） |
| `"cls"` | 取序列**首 token**（CLS） |
| `"last_token"` | 取**最后一个非 padding token** |

**返回值**：列表，每个元素形状为 `(num_layers, hidden_dim)`，即该文本在所有层的池化隐藏状态拼接成一个矩阵。

> 注：`linear_probe.py` 中 `extract_hidden_states` 默认池化为 `"mean"`；`linear_probe_balanced.py` 中同名函数默认为 `"last_token"`（与 `train_probes_balanced.py` 一致）。

---

### 2.4 `get_layer_training_config`

```296:362:engine/probes/linear_probe.py
def get_layer_training_config(layer_idx, num_layers, base_epochs=50, base_lr=0.002, base_dropout=0.1):
```

针对不同深度层自动适配训练超参数，体现"浅层需要更多训练，深层语义更清晰"的策略：

| 层范围 | 类型 | Epochs | 学习率 | Dropout | 准确率要求 |
|--------|------|--------|--------|---------|-----------|
| 0 ~ 5 | 浅层 | ×1.5 | ×1.3 | ×0.6 | ≥ 76% |
| 6 ~ 14 | 中层 | ×1.0 | ×1.0 | ×1.2 | ≥ 85% |
| 15 ~ 27 | 深层 | ×1.0 | ×1.0 | ×1.2 | ≥ 90% |
| 28 | 峰值层 | ×1.2 | ×1.1 | ×1.2 | ≥ 93% |
| 29+ | 最深层 | ×1.0 | ×1.0 | ×1.2 | ≥ 90% |

---

### 2.5 `train_layer_probes`

```365:933:engine/probes/linear_probe.py
def train_layer_probes(
    hidden_states, labels, num_layers, hidden_dim,
    train_indices, val_indices, device,
    ...
) -> Dict[int, Dict]:
```

核心训练函数，逐层训练线性探针。

**数据准备**：
- 从 `hidden_states` 列表中按索引切分出训练集、验证集
- 支持**过采样**（`use_oversample`）：对有害类样本有放回抽样，使安全:有害 ≈ 目标比例
- 支持**类别权重**（`use_class_weight`）：按逆频率加权 `CrossEntropyLoss`，平衡少数类

**训练循环关键设计**：

1. **重试机制**：若某层训练 `min_epochs_before_check=10` 轮后未达标，最多重试 2 次，每次调整超参数（学习率 +10%，Epochs +15%，Dropout -10%）
2. **早停**：
   - 达标后连续 10 轮无提升 → 停止
   - 未达标但连续 20 轮无提升 → 停止（防过拟合）
3. **最佳模型选择**：全程追踪验证集最高准确率对应的模型参数
4. **混合精度训练**：CUDA 可用时自动启用 `torch.amp.autocast` 加速
5. **学习率调度**：`ReduceLROnPlateau`（patience=8，factor=0.5）

**返回值**：每层的训练结果字典：

```python
{
    layer_idx: {
        "model": LinearProbe,           # 已加载最佳参数的模型
        "metrics": {
            "train_acc": ...,           # 最后一轮训练准确率
            "train_acc_best": ...,      # 最佳训练准确率
            "val_acc": ...,              # 最佳验证准确率（核心指标）
            "val_roc_auc": ...,
            "val_pr_auc": ...,
            "test_acc": ...,
            "meets_requirement": bool,
        },
        "toxic_vector": {
            "w_toxic": np.ndarray,       # L2归一化的有害方向向量
            "b": float,
        },
        "training_history": {            # 每轮的指标记录
            "epochs": [...],
            "train_acc": [...],
            "val_acc": [...],
            "train_loss": [...],
            "val_loss": [...],
            "learning_rate": [...],
        }
    }
}
```

---

### 2.6 `save_probes`

```936:1105:engine/probes/linear_probe.py
def save_probes(results, output_dir, model_id="llama-3-8b", filter_threshold=0.75):
```

将训练结果保存到磁盘。每层的输出目录结构：

```
output_dir/
└── probes/
    └── <model_id>/
        ├── summary.json                 # 全局汇总（有效/无效层列表）
        ├── training_log.json            # 完整训练日志
        └── layer_<i>/
            ├── probe.pt                 # 模型权重（torch.save(state_dict)）
            ├── toxic_vector.npz         # 毒性向量（w_toxic, b，含归一化版本）
            ├── metrics.json             # 该层各项指标
            └── training_history.json    # 训练曲线数据
```

**自动过滤**：准确率 < `filter_threshold`（默认75%）的层被标记为 `is_invalid`，但仍保存。

---

## 三、`linear_probe_balanced.py` 详解（推理与加载）

这是训练完成后用于**加载和使用探针**的轻量模块（亦为 `engine.probes` 包默认导出），不包含 `train_layer_probes` 等重型训练循环。

### 3.1 `LinearProbe`（简化版）

```69:126:engine/probes/linear_probe_balanced.py
class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        self.linear = nn.Linear(input_dim, 2)
        nn.init.xavier_uniform_(self.linear.weight)
```

相比训练版更简洁，无 Xavier/Kaiming/Normal 切换，固定用 Xavier；额外提供了 `get_safe_vector()` 方法用于取安全方向向量。

### 3.2 `load_probe` / `load_all_probes`

```132:201:engine/probes/linear_probe_balanced.py
def load_probe(layer_dir, device=None, dropout=0.0):
    state = torch.load(layer_dir / "probe.pt", map_location="cpu")
    input_dim = state["linear.weight"].shape[1]
    probe = LinearProbe(input_dim, dropout=dropout)
    probe.load_state_dict(state)
    probe.to(device).eval()
    # 加载 preprocessor.pkl（StandardScaler）
    scaler = None
    if (layer_dir / "preprocessor.pkl").exists():
        with open(layer_dir / "preprocessor.pkl", "rb") as f:
            scaler = pickle.load(f).get("scaler")
    return probe, scaler
```

从训练产物目录加载一个层的探针模型（`eval` 模式）和可选的预处理器（StandardScaler）。

### 3.3 `extract_hidden_states`（本文件实现）

默认池化策略为 `"last_token"`（最后一个非 padding token）。其余逻辑与 `linear_probe.py` 中版本类似，仅默认参数不同。

---

## 四、数据流总览

```
原始数据文件 (.jsonl)
    ├── text 字段：文本内容
    └── guard.asr_label 字段：标签 (0=安全, 1=有害)
         ↓
┌─────────────────────────────────────────────┐
│  extract_hidden_states()                    │
│  输入: texts (List[str])                    │
│  模型: HuggingFace LLM (如 LLaMA-3-8B)      │
│  输出: hidden_states (List[np.ndarray])    │
│        每个元素形状: (num_layers, hidden_dim)│
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│  train_layer_probes()                       │
│  输入: hidden_states + labels + 划分索引    │
│  处理: 逐层训练 LinearProbe                  │
│  输出: results Dict[layer_idx → metrics]   │
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│  save_probes()                              │
│  输出目录: outputs/probes/<model_id>/       │
│           ├── summary.json                  │
│           ├── training_log.json             │
│           └── layer_<i>/                    │
│               ├── probe.pt                  │
│               ├── toxic_vector.npz          │
│               ├── metrics.json              │
│               └── training_history.json     │
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│  generate_probe_report.py (scripts/)        │
│  加载上述输出目录                             │
│  输出: 验证报告 + 6 张可视化图表              │
└─────────────────────────────────────────────┘
```

---

## 五、输入 / 输出文件对照表

**每个文件的独立对照表** — 以下各 `.py` 文件各自一张两列表格，列出：**读入**哪些路径 / 资源 → **写出**哪些路径 / 结果。

（默认路径以仓库内 `scripts/train_probes_balanced.py` 的 argparse 为准；其它脚本可通过参数覆盖。）

### `engine/probes/__init__.py`

| `.py 文件` | 读入 → 写出 |
|:--|:--|
| `engine/probes/__init__.py` | 无独立磁盘 I/O → 仅从 `linear_probe_balanced` 重导出 `LinearProbe`、`extract_hidden_states`、`load_probe` 等 API |

### `engine/probes/linear_probe_balanced.py`

| `.py 文件` | 读入 → 写出 |
|:--|:--|
| `engine/probes/linear_probe_balanced.py` | 读入 调用方传入的 `texts` + 已加载的 `model` / `tokenizer`（`extract_hidden_states`）；或 读入 `layer_{i}/probe.pt` + （可选）`preprocessor.pkl`（`load_probe` / `load_all_probes`）→ 写出 无默认落盘（函数返回内存中的 `np.ndarray` / `torch.Tensor` / 模型对象；写文件由调用脚本负责） |

### `engine/probes/linear_probe.py`

| `.py 文件` | 读入 → 写出 |
|:--|:--|
| `engine/probes/linear_probe.py` | 读入 内存中的 `hidden_states` / `labels` 等（`train_layer_probes`）；或 同上的文本 + LLM（`extract_hidden_states`，默认池化 `mean`）；`save_probes` 读入 内存中的 `results` 字典 → 写出 `{output_dir}/probes/{model_id}/layer_{i}/probe.pt` + `toxic_vector.npz` + `metrics.json` + `training_history.json` + `summary.json` + `training_log.json`（目录结构由 `save_probes` 参数决定） |

### `scripts/train_probes_balanced.py`

| `.py 文件` | 读入 → 写出 |
|:--|:--|
| `scripts/train_probes_balanced.py` | 读入 `logs/base_evaluation.jsonl`（默认 `--data_file`，可改为任意含 `text` + `guard.asr_label` 的 `.jsonl`）+ （若存在且启用缓存）`outputs/probes/hidden_states_cache.npz` + 经 `ModelManager` 加载的 `AutoModelForCausalLM` 与 tokenizer → 写出 `outputs/probes/hidden_states_cache.npz` + `config.json` + `summary.json` + `training_log.json` + `layer_{i}/probe.pt` + `preprocessor.pkl` + `metrics.json` + `training_history.json`（默认 `--output_dir` 为 `outputs/probes`） |

### `scripts/generate_probe_report.py`

| `.py 文件` | 读入 → 写出 |
|:--|:--|
| `scripts/generate_probe_report.py` | 读入 `outputs/probes/`（或 `--probes_dir`）下各 `layer_{i}/metrics.json` + `training_history.json` + （若存在）`config.json` / `summary.json` → 写出 `outputs/probes_reports/<timestamp>/validation_report.json` + `probe_validation_report.txt` + `fig1~6.png`（可用 `--output_dir` / `--no_plot` 调整） |

### `scripts/extract_toxic_vectors.py`

| `.py 文件` | 读入 → 写出 |
|:--|:--|
| `scripts/extract_toxic_vectors.py` | 读入 `outputs/probes/layer_{i}/probe.pt`（各层，目录可由 `--probes_dir` 指定）→ 写出 `toxic_vectors.npz`（默认路径可用 `--output` 指定）+ 控制台汇总信息 |

### `scripts/linear_probe_balanced.py`

| `.py 文件` | 读入 → 写出 |
|:--|:--|
| `scripts/linear_probe_balanced.py` | 与 `engine/probes/linear_probe_balanced.py` 同构的模块定义（便于路径兼容或被其它脚本引用）→ 无独立 CLI；具体读入写出由**导入该脚本的调用方**决定（逻辑上等价于 engine 版：`probe.pt` / `preprocessor.pkl` 与内存中的隐藏态） |

### 数据字段速查（与上表中的 `.jsonl` 配合）

| 字段 | 含义 |
|:--|:--|
| `text` | 输入给 LLM 的字符串 |
| `guard.asr_label` | `0` = 安全意图，`1` = 有害意图 |

---

## 六、关键设计要点（供理解面试题）

### 1. 为什么叫"探针"？

Linear Probe 不是训练 LLM 本身，而是在 LLM 训练完成后，在其隐藏状态上"插入"一个极简线性分类器，用来探测（probe）LLM 内部表征中是否已编码了某种语义。这是可解释性研究的经典方法。

### 2. 为什么每层单独训练一个探针？

Transformer 的不同层编码不同粒度的语义：
- **浅层**（0~5）：词法、语法编码，分类信号弱 → 需要更多训练、更宽松的准确率要求
- **中层**（6~14）：句法结构、短语语义
- **深层**（15+）：高层语义、意图判断，分类信号强 → 准确率要求更高
- **峰值层**（28）：通常包含最强的任务相关表征

### 3. 过采样 vs 类别权重

实际数据中安全样本远多于有害样本（类别不平衡）。两种解决思路：

- **过采样**（`use_oversample`）：复制少数类样本，使其与多数类接近目标比例
- **类别权重**（`use_class_weight`）：在损失函数中对少数类加大权重（逆频率）

两者可以同时使用。

### 4. 毒性向量（Toxic Vector）有什么用？

`get_toxic_vector()` 取出的是 Linear Probe 权重矩阵的第二行（有害类权重），即：

$$
\mathbf{w}_{\text{toxic}} = \mathbf{W}[1, :], \quad b_{\text{toxic}} = \mathbf{b}[1]
$$

可以用它直接对隐藏状态打分：$s = \mathbf{w}_{\text{toxic}}^\top \mathbf{h} + b_{\text{toxic}}$，无需经过完整 softmax，从而实现高效的在线有害内容检测。

### 5. L2 归一化毒性向量

保存时对 `w_toxic` 做 L2 归一化，使得不同层的向量可比（消除了量纲差异）。原始向量也同时保存在 `w_toxic_original` 中。
