# NeuroLens

大语言模型安全神经元分析与可视化研究平台，实现 NeuroBreak 论文的完整研究管线：从神经元识别、激活投影、象限分类，到目标安全微调（TSFT）与攻击成功率（ASR）评估。

## 功能特性

- **神经元分析**：识别安全神经元 S(q) 与效用神经元 U(p)，计算专用安全神经元 D(p,q) = S(q) \ U(p)
- **SNIP 评分**：基于 I = |w ⊙ ∇L| 的神经元重要性评分与离线选择
- **激活与对齐分析**：激活投影 A_i^k、参数对齐 S_i^k，支持跨层演化追踪
- **象限分类**：将神经元按安全性与激活度分为四象限（S+A+, S-A+, S+A-, S-A-）
- **线性探针**：训练分层线性探针，识别隐藏状态中的有害语义表征
- **目标安全微调（TSFT）**：定向增强安全神经元，VA-TSFT 进一步考虑脆弱性感知
- **SALAD-Bench 评估**：完整的越狱攻击成功率与效用保持度评估管线
- **可视化系统**：9 个交互式分析面板，支持任务触发、实时日志、数据探索

## 支持的模型

NeuroLens 使用 HuggingFace `transformers` 标准接口加载模型，原则上支持任何基于 `AutoModelForCausalLM` 的因果语言模型。

**官方测试模型**（开箱即用）：
- **目标 LLM**：`LLM-Research/Meta-Llama-3-8B-Instruct`（ModelScope）/ `meta-llama/Meta-Llama-3-8B-Instruct`（HuggingFace）
- **安全分类器**：`LLM-Research/Llama-Guard-3-8B`（ModelScope）/ `meta-llama/Llama-Guard-3-8B`（HuggingFace）

使用其他模型时，在 `engine/models.py` 中修改 `LLM_ID` / `GUARD_ID` 常量，或在脚本调用时传入 `--model_id` 参数。

## 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习 | PyTorch 2.6.0 + CUDA 12.4 |
| 模型库 | Transformers 4.46.3，BitsAndBytes（4-bit 量化） |
| 后端 API | FastAPI + Uvicorn |
| 前端可视化 | React 18 + TypeScript，ECharts / Plotly.js / D3.js，Zustand |
| 构建工具 | Vite |

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/YANHAN-BLCU/NeuroLens.git
cd NeuroLens
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备模型

将模型放置到 `ms_models/` 目录，或通过脚本下载：

```bash
# 下载官方测试模型（Llama-3-8B-Instruct + Llama-Guard-3-8B）
python scripts/data/download_models.py --all-8b

# 使用 ModelScope（中国大陆推荐）
export MODELSCOPE_TOKEN=your_token
python scripts/data/download_models.py --all-8b

# 使用 HuggingFace
export HF_TOKEN=your_token
python scripts/data/download_models.py --all-8b --source hf
```

使用自定义模型时，直接将模型目录放入 `ms_models/` 或设置绝对路径：

```bash
export NEUROLENS_MODEL_PATH=/path/to/your/model
```

### 4. 运行核心管线

#### 一键运行完整 NeuroBreak 管线

```bash
python scripts/pipeline/run_neurobreak_pipeline.py \
    --model_id LLM-Research/Meta-Llama-3-8B-Instruct \
    --data_dir data/salad/raw \
    --output_dir outputs
```

#### 分阶段运行

```bash
# Phase 1：识别安全神经元与效用神经元
python scripts/pipeline/run_safety_identifier_salad.py --output_dir outputs/neurons
python scripts/pipeline/run_utility_identifier.py --output_dir outputs/neurons

# Phase 2：激活投影与参数对齐
python scripts/pipeline/run_activation_projection.py --output_dir outputs/neurons
python scripts/pipeline/run_parameter_alignment.py --output_dir outputs/neurons

# Phase 3：象限分类与专用安全神经元计算
python scripts/pipeline/run_quadrant_classification.py --output_dir outputs/neurons
python scripts/pipeline/compute_dedicated_safety_neurons.py --output_dir outputs/neurons

# Phase 4（可选）：TSFT 微调
python scripts/finetuning/run_tsft_finetuning.py \
    --output_dir outputs/tsft_finetuning

# Phase 5：评估
python scripts/evaluation/run_evaluate_asr.py --output_dir outputs/asr
python scripts/evaluation/evaluate_utility.py --output_dir outputs/utility
```

### 5. 准备数据集

```bash
# SALAD-Bench（越狱攻击评估数据集）
python scripts/data/download_salad.py

# 效用基准（ARC、HellaSwag 等）
python scripts/data/download_utility_datasets.py

# 预提取隐藏状态缓存（加速后续探针训练）
python scripts/data/extract_hidden_states.py \
    --data_file data/salad/raw/base_evaluation.jsonl \
    --max_samples 8000 \
    --output outputs/hidden_states_cache/cache.npz
```

### 6. 训练线性探针

```bash
# 平衡线性探针（推荐）
python scripts/probes/linear_probe_balanced.py \
    --hidden_states_cache outputs/hidden_states_cache/cache.npz \
    --output_dir outputs/linear_probes

# 使用 SNIP 离线评分选择神经元
python scripts/probes/offline_snip_compute.py --output_dir outputs/snip
python scripts/probes/offline_snip_select.py \
    --snip_dir outputs/snip \
    --output_dir outputs/neurons
```

### 7. SALAD-Bench 评估

```bash
python scripts/evaluation/evaluate_salad_pipeline.py \
    --data_dir data/salad/raw \
    --output logs/salad_evaluation.jsonl \
    --config base_set \
    --max_samples 500
```

**支持的配置**：
- `base_set`：基础数据集（21,318 样本）
- `attack_enhanced_set`：攻击增强集（5,000 样本）
- `defense_enhanced_set`：防御增强集（200 样本）
- `mcq_set`：多选题集（3,840 样本）

### 8. 启动可视化系统

```bash
cd visualization/backend
uvicorn main:app --host 0.0.0.0 --port 6008
```

访问 `http://localhost:6008` 查看交互式仪表盘。

面板均内置默认数据，无需预先生成 `outputs/` 即可渲染预览；接入真实输出后自动切换为实验数据。

| 面板 | 路径 | 功能 |
|------|------|------|
| A 控制面板 | `/vis/panel_A_control.html` | 微调 / 管线 / ASR 任务触发，实时日志流 |
| B 指标 | `/vis/panel_B_metric.html` | ASR / Utility 核心指标雷达图 |
| C 表征 | `/vis/panel_C_representation.html` | 隐藏态 t-SNE/PCA 散点 |
| D 层演化 | `/vis/panel_D_layer.html` | 层间安全特征演化与梯度依赖 |
| E 神经元 | `/vis/panel_E_neuron.html` | 四象限神经元分类（D3 连接图） |
| F 热图 | `/vis/panel_F_heatmap.html` | 跨层相似度热力图 |
| G Sankey | `/vis/panel_G_sankey.html` | 攻击路径追溯桑基图 |
| H 小提琴 | `/vis/panel_H_violin.html` | 神经元激活分布 |
| K 实例 | `/vis/panel_K_instance.html` | 专用安全神经元与样本实例详情 |

**可选：React 开发模式**（仅二次开发时需要）

```bash
cd visualization/frontend
npm install
npm run dev  # http://localhost:6006，/api 请求代理到 :6008
```

## 项目结构

```
NeuroLens/
├── engine/                     核心库（被 scripts/ 和 visualization/ 共同调用）
│   ├── models.py               模型加载（AutoModelForCausalLM，支持 4-bit 量化）
│   ├── server.py               FastAPI 推理服务端点
│   ├── assessment/             评估管线（SALAD 评估、效用评估、报告生成）
│   ├── neurons/                神经元分析算法
│   │   ├── snip_scorer.py          SNIP 重要性评分
│   │   ├── safety_identifier.py    安全神经元识别 S(q)
│   │   ├── utility_identifier.py   效用神经元识别 U(p)
│   │   ├── activation_projection.py
│   │   ├── parameter_alignment.py
│   │   ├── quadrant_classification.py
│   │   └── gradient_dependency.py
│   ├── probes/                 线性探针（基础版 + 平衡准确率版）
│   └── fine_tuning/            TSFT / VA-TSFT 微调实现
│
├── scripts/                    研究脚本（按功能分 8 个子目录）
│   ├── pipeline/               核心管线入口（run_neurobreak_pipeline.py 等）
│   ├── evaluation/             评估脚本（SALAD、ASR、Utility）
│   ├── data/                   数据下载与预处理
│   ├── finetuning/             微调训练与 delta 应用
│   ├── probes/                 探针训练与 SNIP 离线计算
│   ├── reporting/              报告与可视化数据生成
│   ├── analysis/               结果统计分析
│   └── tools/                  工具脚本（GPU 诊断、模型检查等）
│
├── visualization/              可视化系统
│   ├── backend/                FastAPI 服务（主入口，port 6008）
│   │   ├── main.py             API + 静态文件托管 + subprocess task runner
│   │   ├── index.html          仪表盘主页（9 个 panel 的 iframe 容器）
│   │   └── vis/                panel_A ~ panel_K HTML 文件
│   └── frontend/               React 18 + TypeScript SPA（开发用，port 6006）
│       ├── src/components/     视图组件（与面板对应）
│       ├── src/services/       API 客户端
│       ├── src/store/          Zustand 全局状态
│       └── vite.config.ts      dev: port 6006，proxy /api → :6008
│
├── configs/runtime/            运行时配置（YAML、.env）
├── data/                       数据集（SALAD、Alpaca、Utility 基准）
├── outputs/                    实验产物（神经元 JSON、探针权重、ASR 日志等）
├── ms_models/                  预训练模型缓存
├── docs/                       项目文档
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.9+
- PyTorch 2.0+，CUDA 12.0+（推荐 NVIDIA GPU，16GB+ VRAM）
- Node.js 18+（仅可视化开发时需要）

## 开发

```bash
# 代码格式化
black .
isort .
ruff check .

# GPU 环境诊断
python scripts/tools/check_gpu.py

# 模型路径检查
python scripts/tools/check_models.py

# 快速推理测试
python scripts/tools/test_models.py
```

## 文档

- [模块 API 参考](docs/module_api_reference.md)
- [神经元分析教程](docs/engine_neurons_tutorial.md)
- [微调流程说明](docs/fine_tuning_tutorial.md)
- [SALAD 评估指南](docs/SALAD_EVALUATION_GUIDE.md)
- [探针训练逻辑](docs/探针训练逻辑.md)
- [outputs/ 产物说明](docs/outputs_summary.md)

## 致谢

- [NeuroBreak](https://arxiv.org/abs/2502.07407) — 本项目所实现的研究论文
- [SALAD-Bench](https://github.com/OpenSafetyLab/SALAD-Bench) — 越狱攻击评估数据集
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — 模型加载与推理
- [ModelScope](https://modelscope.cn/) — 中国大陆模型托管

## 许可证

本项目代码遵循 MIT 许可证。使用 Llama 系列模型需遵守 [Meta Llama 使用条款](https://ai.meta.com/llama/use-policy/)。

---

如有问题或建议，请通过 [GitHub Issues](https://github.com/YANHAN-BLCU/NeuroLens/issues) 联系。
