# NeuroLens 项目进度更新（安全神经元分析流水线）

> **更新日期**：2026 年 4 月 4 日  
> **面向读者**：计算机专业本科生；术语尽量通俗，必要时附简短解释。

---

## 一、本次进度在做什么（总览）

团队在 **Meta-Llama-3-8B-Instruct** 上，围绕 **SALAD** 安全数据与 **Alpaca** 通用任务数据，完成了一条「**第0步毒性向量** → 标签合并 → 神经元打分 → 专用安全神经元 → 参数/激活分析 → 四象限 → 梯度依赖」的流水线。  
下面按**执行顺序**说明每一步**做了什么**、**生成了什么文件**、**文件干什么用**。

---

## 二、第0步：毒性向量提取与聚合（`toxic_vectors.npz`）

> **说明**：此步在流水线中最靠前，编号为 **第0步**；它依赖各层已训练好的**线性探针**权重（如 `outputs/linear_probes/layers/layer01/layer01.pt` … `layer32/layer32.pt`）。探针本身需先用带标签数据训练，本文不展开训练命令。

### 当前步骤做了什么

从每一层线性探针的权重里抽出 **毒性方向向量** \(W_{\text{toxic}}\)（与有害/安全两类相关的权重差），再**合并成一份**统一的 NumPy 压缩包。

- **常用定义**（二分类探针）：  
  \[
  W_{\text{toxic}} = W_{\text{toxic\_class}} - W_{\text{safe\_class}}
  \]  
  即「有害类权重 − 安全类权重」，表示在激活空间里**把状态往「更有害」一侧推**的一个方向（数学上是差向量，工程上常再归一化后用于投影）。

- **处理过程**：逐层读取 `layerXX/layerXX.pt` 中的探针权重，得到每层的 \(w_{\text{toxic}}\)，再堆叠/打包进单个 `.npz` 文件。

### 本次输出文件

| 文件路径 | 格式 | 规模（参考） |
|----------|------|----------------|
| `outputs/toxic_vectors/toxic_vectors.npz` | NumPy `.npz` | 约 1 MB 量级（视实现而定） |

**文件内部常见字段（与实现对齐，名称以实际 `npz` 为准）**：

| 键名 | 含义（典型） |
|------|----------------|
| `vectors` | 形状约 **(32, 4096)**，32 层毒性方向堆成的矩阵 |
| `biases` | 形状约 **(32,)**，每层探针偏置 |
| `layer_indices` | 形状约 **(32,)**，层号 1～32 |
| `layer_1` … `layer_32` | 单层向量，便于按层读取 |
| `meta` | 数据来源、模型、路径等元信息 |

> **4096** 与 **32 层**对应 Llama 类隐藏维与层数；若你用的模型不同，维度和层数会不同。

### 文件作用与实际用途

- **作用**：为整网提供**各层统一的「毒性方向」参考**，后续所有「和毒性方向有多对齐」的投影都依赖它。  
- **用途**：  
  - **参数对齐**（\(S_i^k\)）：神经元权重与 \(w_{\text{toxic}}\) 的余弦相似度等。  
  - **激活投影**（\(A_i^k\)）：前向激活在毒性方向上的投影。  
  - **四象限分类**：S/A 两维中，与毒性向量相关的量由此文件间接支撑。  
  - **梯度依赖等分析**：常与同一套「安全神经元 + 毒性语义」框架一起使用，保证各步方向一致。

---

## 三、标签合并（数据准备）

### 当前步骤做了什么

将 **4 个分片标签文件**按顺序**首尾相接**合并成一个 `.jsonl` 文件（常见做法是用 `cat` 等命令逐行拼接），保证 **0～21316** 索引对应的标签**不重复、不遗漏**，形成一份「全量 base 集标签」。

### 本次输出文件

| 文件路径（相对项目或 AutoDL 环境） | 格式 |
|-----------------------------------|------|
| `outputs/data_set_output/labels/base_set_outputs_merged.jsonl` | JSONL（每行一条 JSON） |

### 文件作用与实际用途

- **作用**：把分散的标签文件收成**一份统一标签源**，后续脚本只需读**一个路径**。  
- **用途**：训练探针、分析毒性向量、计算 SNIP 安全/效用分数等下游任务；例如训练探针时 `--data_file` 可直接指向该合并文件，而不用在命令里写多个分片路径。

---

## 四、安全神经元 SNIP 打分（`run_safety_identifier_salad.py`）

### 当前步骤做了什么

用 **SNIP（Sensitivity-based Pruning，基于敏感度的剪枝思路）** 给**每一个神经元**算一个**安全重要性分数**：

1. 加载 **LLaMA-3-8B** 与 **SALAD** 安全相关数据（本例为 **base_set + attack_enhanced_set**，约 **2000** 条样本）。  
2. 对每个神经元，近似衡量「若弱化/拿掉该神经元，**安全样本上的损失会怎么变**」——变化量即 **SNIP 分数**。  
3. 对所有神经元**全局排序**，并记录排名、百分位。

> **通俗理解**：分数高的神经元，对「模型在安全数据上表现好不好」更敏感，更可能是**安全相关**的神经元。

### 本次输出文件

目录示例：`/root/autodl-tmp/neurolens/outputs/neurons/`（本地可为 `outputs/neurons/`）。

| 文件名 | 说明 |
|--------|------|
| `safety_all_neurons_scores.json` | **全部神经元**的分数、排名、百分位（本流水线核心产物之一）。 |
| `safety_neurons.json` | 若设置阈值 `q`，会额外保存 **Top q%** 神经元；当 `--safety_threshold_q -1`（不设阈值）时，**通常只保存上面「全量分数」文件，不输出此筛选文件**。 |

### 文件作用与实际用途

- **作用**：得到一张「安全维度」的**全局排行榜**。  
- **用途**：用 `select_neurons_by_threshold.py` 等脚本按不同比例（如 top 0.5%）再筛选；也可作为后续**抑制/定向干预**某些神经元的候选依据。

---

## 五、效用神经元 SNIP 打分（`run_utility_identifier.py`）

### 当前步骤做了什么

同样用 **SNIP**，但在 **Alpaca 通用任务**数据上算「**效用重要性**」：

1. 加载 **LLaMA-3-8B** 与 **Alpaca**（本例约 **1000** 条样本）。  
2. 对每个神经元，衡量其对 **Alpaca 任务损失**的影响，得到 **效用 SNIP 分数**。  
3. 全局排序并记录排名、百分位。

### 本次输出文件

| 文件名 | 说明 |
|--------|------|
| `utility_all_neurons_scores.json` | **全部神经元**的效用分数、排名、百分位。 |
| `utility_neurons.json` | 设阈值 `p` 时保存 Top p%；`--utility_threshold_p -1` 时一般**只保留全量分数文件**。 |

### 文件作用与实际用途

- **作用**：得到「通用能力/效用」上的神经元重要性排序。  
- **用途**：与**安全分数**对比，做**安全 vs 效用**权衡：哪些神经元「又重要又危险」、哪些「专管安全不太影响通用能力」等，为后续**精准编辑模型**提供依据。

---

## 六、专用安全神经元（`compute_dedicated_safety_neurons.py`）

### 当前步骤做了什么

按论文思路计算**专用安全神经元集合**：

\[
D(p,q) = S(q) \setminus U(p)
\]

- **S(q)**：从 `safety_all_neurons_scores.json` 里按 **q=0.005**（top **0.5%**）取出安全高分神经元。  
- **U(p)**：从 `utility_all_neurons_scores.json` 里按 **p=0.001**（top **0.1%**）取出效用高分神经元。  
- **D(p,q)**：在 **S(q)** 里**删掉**同时属于 **U(p)** 的神经元，得到「更偏安全、不那么像通用能力核心」的集合。

### 本次输出文件

| 文件名 | 路径示例 |
|--------|----------|
| `dedicated_safety_neurons.json` | `outputs/neurons/` 或 AutoDL 下 `.../outputs/neurons/` |

**内容概要**：专用神经元列表（含分数、排名、百分位等）、**元数据**（来源文件、阈值）、**统计**（安全集大小、效用集大小、重叠数量、专用集占比等）。

### 文件作用与实际用途

- **作用**：挑出「**更专管安全**」的神经元子集，减少与「通用语言生成」高度重叠的神经元。  
- **用途**：作为后续 **参数对齐、激活投影、梯度依赖** 等分析的 **target 神经元列表**；也可用于设计**安全增强**（如抑制某些激活）时**降低误伤通用能力**的风险。重叠比例还能帮助判断「安全与效用有多可分」。

---

## 七、参数对齐（`run_parameter_alignment.py`）

### 当前步骤做了什么

对应论文 **5.4 节**思路：对每个**目标神经元**（如专用安全神经元），取 MLP 里与该神经元相关的 **W_down 行向量**，与预计算的**毒性向量** \(w_{\text{toxic}}\) 算 **余弦相似度** \(S_i^k\)，并按正负分为 **S+** / **S-**。

- **S+**：权重方向与毒性方向**更同向**（可理解为参数层面更「顺着有害方向」）。  
- **S-**：**更反向或弱对齐**（可理解为参数层面更「顶着有害方向」）。

### 本次输出文件

| 文件名 | 路径示例 |
|--------|----------|
| `parameter_alignment.json` | `outputs/neurons/` |

**内容概要**：每个神经元的余弦相似度、对齐类型（S+/S-）、统计信息（S+/S- 数量、最值、均值、中位数等）、Top/Bottom 列表等。

### 文件作用与实际用途

- **作用**：从**权重（参数）**角度描述神经元与「毒性方向」的关系。  
- **用途**：制定干预策略（例如更关注 S+ 是否需要抑制）、与激活结果一起做**四象限分类**；为**神经元消融（ablation）**等实验提供依据。

---

## 八、激活投影数据预处理（`preprocess_activation_dataset.py`）

### 当前步骤做了什么

把 SALAD 原始数据与**模型输出标签**对齐，转成 **`run_activation_projection.py`** 能直接读的 JSONL：

1. **base_set**：多个标签分片 + `base_set_train.jsonl`，用 **`original_index`** 对齐。  
2. **attack_enhanced_set**：`attack_enhanced_set_train.jsonl` + 对应标签文件。  
3. 标签映射为布尔字段 **`jailbreak_success`**：**Unsafe → True**（越狱成功），**Safe / Controversial → False**（越狱失败）。  
4. 统一出 **`text`** 字段（base 用 `question`，attack 用 `augq`），合并为一个大 JSONL。

### 本次输出文件

| 文件名 | 路径示例 |
|--------|----------|
| `activation_projection_dataset.jsonl` | `data/salad/raw/` |

### 文件作用与实际用途

- **作用**：带 **「是否越狱成功」标签** 的 prompt 级数据集。  
- **用途**：**激活投影分析**的输入；可对比成功/失败样本上激活差异，验证专用安全神经元是否与安全行为相关；也可用于训练**线性探针**以更新毒性方向等。

---

## 九、激活投影（`run_activation_projection.py`）

### 当前步骤做了什么

仍对应论文 **5.4 节**：在带标签的样本上跑模型，用 **Hook** 取出目标神经元在 MLP 中的**激活**，再与毒性向量做**点积投影** \(A_i^k\)：

- 分别统计 **越狱成功** 与 **越狱失败** 样本上的平均投影。  
- 计算 **`activation_diff`（成功 − 失败）** 等，用于对比两类输入下行为差异。

（本例常见配置：如 **500** 条样本、`dedicated_safety_neurons.json` 作为目标神经元等，具体以实际命令行为准。）

### 本次输出文件

| 文件名 | 路径示例 |
|--------|----------|
| `activation_projection.json` | `outputs/neurons/` |

### 文件作用与实际用途

- **作用**：从**运行时激活**角度刻画神经元与毒性方向的关系，并区分越狱成功/失败。  
- **用途**：验证专用安全神经元是否真的在「越狱成功」时表现不同；与参数对齐一起输入 **四象限分类**；**activation_diff** 大的神经元可作为干预优先级参考。

---

## 十、四象限分类（`run_quadrant_classification.py`）

### 当前步骤做了什么

读取 **`parameter_alignment.json`**（S 维度：余弦相似度）和 **`activation_projection.json`**（A 维度：激活投影，分类所用值与脚本实现一致，一般为用于象限的投影量）。

用阈值 **`threshold_s = 0.0`**、**`threshold_a = 0.0`** 划分：

| 维度 | 条件（示意） | 含义（直观） |
|------|----------------|--------------|
| **S+ / S-** | 余弦相似度 > 0 或 ≤ 0 | 参数与毒性方向同向 / 不同向 |
| **A+ / A-** | 激活投影 > 0 或 ≤ 0 | 投影偏「促进毒性侧」/ 另一侧 |

组合得到四个象限：**S+A+、S-A+、S+A-、S-A-**。可加 **`--print-statistics`** 打印各象限数量与占比等。

### 本次输出文件

| 文件名 | 路径示例 |
|--------|----------|
| `quadrant_classification.json` | `outputs/neurons/` |

### 文件作用与实际用途

- **作用**：给每个神经元贴「**参数 × 激活**」联合标签，便于分工干预。  
- **用途**：  
  - **S+A+**：参数与激活都偏危险侧 → 常作为**优先抑制**候选。  
  - **S-A-**：两侧都偏安全侧 → 常作为**优先保护**对象。  
  - **S+A- / S-A+**：参数与激活**不一致** → 可能反映安全机制**正在起作用**或**失效**，需结合实验细查。  
  不同象限可对应 **ablation、激活修补（activation patching）、FFN 权重修改**等不同策略。

---

## 十一、梯度分析数据预处理（`preprocess_gradient_dataset.py`）

### 当前步骤做了什么

将 **`base_set_train.jsonl`** 与 **`attack_enhanced_set_train.jsonl`** 转为统一格式：

- `question` / `augq` → 统一字段 **`text`**。  
- **不做标签合并**（与 `preprocess_activation_dataset.py` 不同），只拼成一个大 JSONL。

### 本次输出文件

| 文件名 | 路径示例 |
|--------|----------|
| `gradient_dependency_dataset.jsonl` | `data/salad/raw/` |

### 文件作用与实际用途

- **作用**：给 **`run_gradient_dependency.py`** 提供「纯文本 prompt」批量前向与反传用的数据。  
- **用途**：做**梯度依赖**分析；与参数对齐、激活投影交叉看，可从「优化/敏感路径」角度补充神经元功能画像。

---

## 十二、梯度依赖（`run_gradient_dependency.py`）

### 当前步骤做了什么

对应论文 **5.4 节**思路：以 **W_down 神经元**为锚点，通过反传估计**上一层相关权重**对当前神经元激活的敏感度，即形如：

\[
G_{i,j} = \frac{\partial a_{\text{down},i}^{k}}{\partial w_{\text{upstream},j}^{k-1}}
\]

本例常见设置：**200** 条样本、`--top-k 0.1` 表示每个目标神经元只保留**梯度强度最高的约 10%** 上游连接，得到**稀疏依赖图**。

### 本次输出文件

| 文件名 | 路径示例 |
|--------|----------|
| `gradient_dependency.json` | `outputs/neurons/gradient_dependency/` |
| `gradient_dependency_visualization.json` | 同目录（供可视化） |

**`gradient_dependency.json` 主要内容**：每个目标神经元的 **`upstream_neurons`**、**`gradient_strengths`**、**均值/最大梯度强度**、上游数量等。

### 文件作用与实际用途

- **作用**：描述「**谁通过梯度路径强烈影响谁**」，补充参数与激活之外的一条因果/敏感链条。  
- **用途**：梳理层间**信息依赖路径**；评估抑制某个神经元可能波及的上下游；与 **S（参数对齐）**、**A（激活投影）** 一起形成更完整的**安全相关机制地图**。

---

## 十三、流水线一览（便于对齐分工）

建议团队按依赖关系理解顺序（具体脚本名以仓库为准）：

```
【前置】训练各层线性探针（带标签数据）
  → outputs/linear_probes/layers/layer01/layer01.pt … layer32/layer32.pt

第0步 毒性向量聚合（从探针权重抽取并打包）
  → outputs/toxic_vectors/toxic_vectors.npz

标签合并
  → base_set_outputs_merged.jsonl

安全 SNIP（SALAD）
  → safety_all_neurons_scores.json

效用 SNIP（Alpaca）
  → utility_all_neurons_scores.json

专用安全神经元 D = S \ U
  → dedicated_safety_neurons.json

参数对齐（依赖 toxic_vectors.npz）
  → parameter_alignment.json

激活数据预处理
  → activation_projection_dataset.jsonl

激活投影（依赖 toxic_vectors.npz）
  → activation_projection.json

四象限分类（依赖 parameter + activation 两步结果）
  → quadrant_classification.json

梯度数据预处理
  → gradient_dependency_dataset.jsonl

梯度依赖
  → gradient_dependency.json（+ visualization）
```

> **说明**：**第0步**产出 **`toxic_vectors.npz`**，是参数对齐与激活投影的**共同方向基准**；探针训练为第0步的前置条件。若你方仓库路径统一为 `outputs/` 相对路径，可将文中 AutoDL 绝对路径视为同一逻辑位置的部署实例。

---

## 十四、步骤总表（输入 / 输出 / 用途）

| 步骤 | 脚本 / 环节 | 输入 | 输出 | 用途 |
|:---:|---|---|---|---|
| **前置** | 训练各层线性探针 | 带标签的 SALAD 数据（Safe / Unsafe） | `outputs/linear_probes/layers/layerXX/layerXX.pt`（32 个文件） | 为第 0 步提供探针权重 |
| **0** | 毒性向量提取与聚合 | 各层探针权重 `layerXX.pt` | `outputs/toxic_vectors/toxic_vectors.npz` | 为参数对齐和激活投影提供统一的"毒性方向"基准 |
| **标签合并** | 合并分片标签 | 4 个分片标签文件 | `outputs/data_set_output/labels/base_set_outputs_merged.jsonl` | 收口为单一标签源，后续训练探针、SNIP 等均引用此文件 |
| **安全 SNIP** | `run_safety_identifier_salad.py` | `toxic_vectors.npz` + SALAD 数据 + 合并标签 | `safety_all_neurons_scores.json` | 得到每个神经元对"安全样本损失"的敏感度排序 |
| **效用 SNIP** | `run_utility_identifier.py` | Alpaca 通用任务数据 | `utility_all_neurons_scores.json` | 得到每个神经元对"通用任务损失"的敏感度排序 |
| **专用安全神经元** | `compute_dedicated_safety_neurons.py` | `safety_all_neurons_scores.json`（Top q%）+ `utility_all_neurons_scores.json`（Top p%）| `dedicated_safety_neurons.json` | 筛选出"偏安全但不太影响通用能力"的神经元子集，作为后续分析的目标 |
| **参数对齐** | `run_parameter_alignment.py` | `toxic_vectors.npz` + `dedicated_safety_neurons.json` | `parameter_alignment.json` | 从权重角度量化每个目标神经元与毒性方向的余弦相似度（S 维度） |
| **激活数据预处理** | `preprocess_activation_dataset.py` | `base_set_train.jsonl` + 合并标签；`attack_enhanced_set_train.jsonl` + 合并标签 | `activation_projection_dataset.jsonl` | 生成含 `text` 与 `jailbreak_success`（True/False）标签的统一 JSONL，供激活投影使用 |
| **激活投影** | `run_activation_projection.py` | `toxic_vectors.npz` + `dedicated_safety_neurons.json` + `activation_projection_dataset.jsonl` | `activation_projection.json` | 从运行时激活角度量化目标神经元在越狱成功/失败样本上的投影差异（A 维度） |
| **四象限分类** | `run_quadrant_classification.py` | `parameter_alignment.json`（S 维度）+ `activation_projection.json`（A 维度）| `quadrant_classification.json` | 联合 S/A 两个维度将神经元划分为 S+A+ / S+A- / S-A+ / S-A- 四类，为干预策略提供优先级与方向 |
| **梯度数据预处理** | `preprocess_gradient_dataset.py` | `base_set_train.jsonl` + `attack_enhanced_set_train.jsonl` | `gradient_dependency_dataset.jsonl` | 生成纯文本 prompt 集合，供梯度依赖分析做批量前向与反传 |
| **梯度依赖** | `run_gradient_dependency.py` | `dedicated_safety_neurons.json` + `gradient_dependency_dataset.jsonl` | `gradient_dependency.json`（+ `gradient_dependency_visualization.json`）| 量化目标神经元与其上游神经元的梯度关联强度，构建稀疏的层间依赖图，为干预副作用评估提供依据 |

> **说明**：表格中的路径为 AutoDL 环境示例；本地仓库中对应逻辑路径均在 `outputs/` 目录树下，文件名保持一致。

---

## 十五、后续可做事项（可选）

1. 对 **S+A+**、**S-A-** 抽样做**人工或自动**行为验证。  
2. 用 **`gradient_dependency_visualization.json`** 做依赖图可视化。  
3. 基于象限结果设计**小范围消融实验**，观察安全指标变化。  
4. 将上述步骤封装为**一键脚本**，减少环境差异导致的路径错误。

---

*文档根据团队对话与 2026-04-04 提供的步骤说明整理；若参数（样本数、阈值、路径）与本地运行不一致，以实际命令与 `metadata` 为准。*
