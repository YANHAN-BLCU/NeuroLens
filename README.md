# NeuroLens

一个基于 Llama 推理模型和 Llama Guard 安全审核的 AI 安全评估研究项目，专注于大语言模型安全防护能力的研究与评估。

## ✨ 功能特性

- 🤖 **智能推理**：基于 Meta Llama 3 8B 模型进行文本生成和对话
- 🛡️ **安全审核**：集成 Llama Guard 3 进行内容安全检测和过滤
- 🧪 **SALAD 评估**：支持 SALAD-Bench 数据集评估，测试模型安全防护能力
- 🔍 **线性探针训练**：训练分层线性探针，识别模型隐藏状态中的有害语义
- 📊 **可视化系统**：提供 6 个可视化视图（指标、表征、层、神经元、实例、控制面板）
- 📈 **结果分析**：提供完整的评估结果分析和报告生成
- 🐳 **Docker 支持**：完整的容器化部署方案，便于实验环境复现
- ⚙️ **灵活配置**：支持自定义模型参数、审核阈值和类别
- 📈 **数据划分策略**：支持多种数据划分方案（6:2:2、6.5:1:1.5等）和过采样

## 🏗️ 技术栈

### 核心框架
- **深度学习**：PyTorch 2.6.0 + CUDA 12.4
- **模型库**：Transformers 4.46.3
- **量化加速**：BitsAndBytes 4-bit 量化
- **模型管理**：ModelScope（推荐，中国大陆访问更快）或 HuggingFace Transformers

### 前端可视化
- **框架**：React 18 + TypeScript
- **UI 组件**：Material UI
- **图表库**：ECharts、Plotly.js、Cytoscape.js
- **状态管理**：Zustand
- **构建工具**：Vite

### 后端 API
- **框架**：FastAPI
- **数据处理**：NumPy、Pydantic

### 部署
- **容器化**：Docker + NVIDIA CUDA 12.4

## 📋 前提条件

- Python 3.9+
- CUDA 12.4+ (推荐，用于 GPU 加速，8B 模型需要)
- Docker (可选，用于容器化部署)
- Node.js 18+ (用于前端开发)
- ModelScope 账号（推荐，中国大陆访问更快）或 HuggingFace 账号，已申请模型访问权限
  - ModelScope: `LLM-Research/Meta-Llama-3-8B-Instruct` 与 `LLM-Research/Llama-Guard-3-8B`
  - HuggingFace: `meta-llama/Meta-Llama-3-8B-Instruct` 与 `meta-llama/Llama-Guard-3-8B`

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/YANHAN-BLCU/NeuroLens.git
cd NeuroBreak-Reproduction
```

### 2. 安装依赖

```bash
# Python 依赖
pip install -r requirements.txt

# 前端依赖 (可选，仅开发可视化)
cd visualization/frontend
npm install
```

### 3. 配置环境变量

创建 `.env` 文件（或设置环境变量）：

```bash
# ModelScope Token（推荐，中国大陆访问更快）
MODELSCOPE_TOKEN=your_modelscope_token_here

# 或者使用 HuggingFace Token
# HF_TOKEN=your_huggingface_token_here

# 模型路径（可选，默认使用 ModelScope/HuggingFace 缓存）
MODEL_CACHE_DIR=/path/to/models
```

### 4. 下载模型（可选）

使用提供的脚本下载模型：

```bash
# 下载默认的 8B 模型（使用 ModelScope）
python scripts/download_models.py --all-8b

# 设置 ModelScope token（如果需要）
export MODELSCOPE_TOKEN=your_token
python scripts/download_models.py --all-8b
```

### 5. 运行评估实验

#### SALAD-Bench 数据集评估

```bash
# 在 Docker 容器内运行
docker exec -it neurolens /bin/bash
cd /workspace
python scripts/evaluate_salad_pipeline.py \
    --data_dir /workspace/data/salad/raw \
    --output /workspace/logs/salad_evaluation.jsonl \
    --config base_set \
    --max_samples 100
```

或使用 PowerShell 脚本（Windows）：

```powershell
.\scripts\run_salad_evaluation.ps1 -Config base_set -MaxSamples 100
```

#### 线性探针训练

训练分层线性探针，识别模型隐藏状态中的有害语义：

```bash
# 基本训练（6.5:1:1.5 划分方案）
python scripts/train_linear_probes.py \
    --data_file data/salad/raw/base_evaluation.jsonl \
    --output_dir outputs/probes \
    --num_epochs 80 \
    --lr 3e-3

# 6:2:2 划分 + 训练集过采样（方案A：1:1.5比例）
python scripts/train_622_full.py \
    --data_file data/salad/raw/base_evaluation.jsonl \
    --output_dir outputs/probes_622_full \
    --oversample_probe_train \
    --num_epochs 80

# 6:2:2 划分 + 1.5:1 训练集比例 + 测试集评估
python scripts/train_test_1_5_1.py \
    --data_file data/salad/raw/base_evaluation.jsonl \
    --output_dir outputs/probes_test_1_5_1 \
    --num_epochs 80

# 使用预提取的隐藏态缓存（更快）
python scripts/extract_hidden_states.py \
    --data_file data/salad/raw/base_evaluation.jsonl \
    --max_samples 8000 \
    --output outputs/hidden_states_cache/cache.npz

python scripts/train_linear_probes.py \
    --hidden_states_cache outputs/hidden_states_cache/cache.npz \
    --output_dir outputs/probes
```

详细说明请参考 [探针训练逻辑](docs/探针训练逻辑.md)。

#### IO 测试

```bash
python scripts/run_io_tests.py
```

### 6. 启动可视化系统

```bash
# 启动后端 (端口 8000)
cd visualization/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 启动前端 (端口 5173)
cd visualization/frontend
npm run dev
```

访问 http://localhost:5173 查看可视化界面。

## 🐳 Docker 部署

### 构建镜像

```bash
docker build -t neurolens:v1 -f docker/Dockerfile .
```

### 运行容器

```bash
docker run -it --gpus all \
  -v /path/to/models:/workspace/ms_models \
  -e MODELSCOPE_TOKEN=your_token \
  neurolens:latest
```

**注意**：
- 模型路径已更新为 `/workspace/ms_models`，请确保正确挂载模型目录
- 推荐使用 ModelScope token（`MODELSCOPE_TOKEN`），中国大陆访问速度更快
- 如果使用 HuggingFace，可设置 `HF_TOKEN` 环境变量

详细部署指南请参考 [DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)。

## 📁 项目结构

```
NeuroLens/
├── engine/                 # 核心引擎
│   ├── models.py          # 模型管理模块
│   ├── assessment/        # 评估模块
│   │   ├── evaluate.py    # 评估逻辑
│   │   └── report.py     # 报告生成
│   ├── probes/           # 探针模块
│   │   └── linear_probe.py
│   └── engine/            # 引擎核心
├── scripts/               # 工具脚本
│   ├── download_models.py    # 模型下载脚本
│   ├── evaluate_salad_pipeline.py  # SALAD 评估脚本
│   ├── train_linear_probes.py      # 线性探针训练脚本
│   ├── train_622_full.py           # 6:2:2划分 + 过采样训练脚本
│   ├── train_test_1_5_1.py         # 6:2:2划分 + 测试集评估脚本
│   ├── extract_hidden_states.py    # 隐藏态提取脚本
│   ├── analyze_salad_results.py     # SALAD 结果分析脚本
│   └── ...
├── visualization/         # 可视化系统
│   ├── frontend/         # 前端应用
│   │   ├── src/
│   │   │   ├── components/  # 视图组件
│   │   │   │   ├── ControlPanel.tsx     # 控制面板
│   │   │   │   ├── MetricView.tsx       # 指标视图
│   │   │   │   ├── RepresentationView.tsx  # 表征视图
│   │   │   │   ├── LayerView.tsx        # 层视图
│   │   │   │   ├── NeuronView.tsx      # 神经元视图
│   │   │   │   └── InstanceView.tsx    # 实例视图
│   │   │   ├── services/    # API 服务
│   │   │   ├── store/       # 状态管理
│   │   │   └── types/       # 类型定义
│   │   └── package.json
│   ├── backend/          # 后端 API
│   │   ├── main.py
│   │   └── requirements.txt
│   └── README.md
├── docs/                 # 文档目录
│   ├── DEPLOYMENT_GUIDE.md       # 部署指南
│   ├── SALAD_EVALUATION_GUIDE.md # SALAD 评估指南
│   ├── SALAD_EVALUATION_ANALYSIS.md  # SALAD 评估分析报告
│   ├── 探针训练逻辑.md            # 线性探针训练详细说明
│   └── ...
├── data/                 # 数据目录
│   └── salad/            # SALAD-Bench 数据集
├── ms_models/            # 模型文件目录
│   ├── LLM-Research/
│   │   ├── Meta-Llama-3-8B-Instruct/
│   │   └── Llama-Guard-3-8B/
├── docker/               # Docker 配置
│   └── Dockerfile
├── configs/              # 配置文件
│   └── runtime/
├── logs/                 # 日志和评估结果
├── outputs/              # 训练输出
├── requirements.txt      # Python 依赖
└── README.md            # 本文件
```

## 📚 文档

- [部署指南](docs/DEPLOYMENT_GUIDE.md) - 详细部署说明
- [模型适配总结](docs/MODEL_ADAPTATION_SUMMARY.md) - 模型配置说明
- [SALAD 评估指南](docs/SALAD_EVALUATION_GUIDE.md) - SALAD-Bench 数据集评估指南
- [SALAD 评估分析](docs/SALAD_EVALUATION_ANALYSIS.md) - SALAD 评估结果分析报告
- [探针训练逻辑](docs/探针训练逻辑.md) - 线性探针训练详细说明
- [数据集划分用途详解](docs/数据集划分用途详解.md) - 数据划分策略说明

## 🧪 测试与评估

### IO 测试

```bash
python scripts/run_io_tests.py
```

### SALAD-Bench 评估

运行 SALAD-Bench 数据集评估（需要先下载数据集）：

```bash
# 在 Docker 容器内运行
docker exec -it neurolens /bin/bash
cd /workspace
python scripts/evaluate_salad_pipeline.py \
    --data_dir /workspace/data/salad/raw \
    --output /workspace/logs/salad_evaluation.jsonl \
    --config base_set \
    --max_samples 100
```

或使用 PowerShell 脚本（Windows）：

```powershell
.\scripts\run_salad_evaluation.ps1 -Config base_set -MaxSamples 100
```

**支持的配置**：
- `base_set`: 基础数据集（21,318 样本）
- `attack_enhanced_set`: 攻击增强集（5,000 样本）
- `defense_enhanced_set`: 防御增强集（200 样本）
- `mcq_set`: 多选题集（3,840 样本）

详细说明请参考 [SALAD 评估指南](docs/SALAD_EVALUATION_GUIDE.md)。

### 分析评估结果

分析 SALAD 评估结果：

```bash
python scripts/analyze_salad_results.py
```

## 🔧 开发

### 代码格式化

```bash
# Python
black .
isort .
ruff check .
```

### 检查模型

```bash
python scripts/check_models.py
```

## ⚠️ 注意事项

1. **模型访问权限**：需要申请 Meta Llama 和 Llama Guard 模型的访问权限
   - ModelScope（推荐）：访问 https://modelscope.cn 申请模型权限
   - HuggingFace：访问 https://huggingface.co 申请模型权限
2. **模型路径**：模型默认路径为 `/workspace/ms_models`（容器内）或 `ms_models/`（本地）
3. **模型下载**：推荐使用 ModelScope 下载模型，中国大陆访问速度更快
4. **GPU 推荐**：8B 模型需要 GPU 支持，建议使用 NVIDIA GPU（16GB+ VRAM）
5. **首次加载**：8B 模型首次加载需要较长时间，这是正常现象
6. **内存要求**：建议至少 32GB RAM，使用 GPU 时建议 16GB+ VRAM（8B 模型）
7. **网络要求**：首次运行需要下载模型（约 16GB+），使用 ModelScope 可加速下载
8. **SALAD 评估**：运行 SALAD 评估前需要先下载 SALAD-Bench 数据集

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目遵循相应的开源许可证。使用 Meta Llama 模型需要遵守 [Llama 使用条款](https://ai.meta.com/llama/use-policy/)。

## 🙏 致谢

- [Meta Llama](https://ai.meta.com/llama/) - 提供强大的语言模型
- [ModelScope](https://modelscope.cn/) - 模型托管平台（中国大陆推荐）
- [HuggingFace](https://huggingface.co/) - 模型托管和 Transformers 库
- [SALAD-Bench](https://github.com/facebookresearch/SALAD-Bench) - 安全评估数据集

## 📮 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**
