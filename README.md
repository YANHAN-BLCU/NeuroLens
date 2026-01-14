# NeuroLens

一个集成了 Llama 推理模型和 Llama Guard 安全审核的 AI 应用系统，提供完整的 Web 界面和 API 服务。

## ✨ 功能特性

- 🤖 **智能推理**：基于 Meta Llama 3 8B 模型进行文本生成和对话
- 🛡️ **安全审核**：集成 Llama Guard 3 进行内容安全检测和过滤
- 🧪 **SALAD 评估**：支持 SALAD-Bench 数据集评估，测试模型安全防护能力
- 🎨 **现代化前端**：基于 React + TypeScript + Vite 构建的响应式 Web 界面
- 🚀 **高性能后端**：FastAPI 提供 RESTful API 服务
- 🐳 **Docker 支持**：完整的容器化部署方案
- ⚙️ **灵活配置**：支持自定义模型参数、审核阈值和类别

## 🏗️ 技术栈

### 后端
- **框架**：FastAPI 0.115.4
- **深度学习**：PyTorch 2.6.0 + CUDA 11.8
- **模型库**：Transformers 4.46.3
- **加速**：Accelerate 1.1.1
- **服务器**：Uvicorn 0.32.0

### 前端
- **框架**：React 19.2.0 + TypeScript 5.9.3
- **构建工具**：Vite 7.2.4
- **样式**：Tailwind CSS 3.4.14
- **状态管理**：Zustand 5.0.8
- **数据获取**：TanStack Query 5.90.10
- **图表**：Recharts 3.5.0

### 部署
- **容器化**：Docker + NVIDIA CUDA 12.4
- **模型管理**：ModelScope（推荐，中国大陆访问更快）或 HuggingFace Transformers

## 📋 前提条件

- Python 3.9+
- Node.js 18+ (用于前端开发)
- CUDA 12.4+ (推荐，用于 GPU 加速，8B 模型需要)
- Docker (可选，用于容器化部署)
- ModelScope 账号（推荐，中国大陆访问更快）或 HuggingFace 账号，已申请模型访问权限
  - ModelScope: `LLM-Research/Meta-Llama-3-8B-Instruct` 与 `LLM-Research/Llama-Guard-3-8B`
  - HuggingFace: `meta-llama/Meta-Llama-3-8B-Instruct` 与 `meta-llama/Llama-Guard-3-8B`

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/YANHAN-BLCU/NeuroBreak-Reproduction-.git
cd NeuroBreak-Reproduction
```

### 2. 安装后端依赖

```bash
pip install -r requirements.txt
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

### 5. 启动后端服务

```bash
# 方法1: 使用启动脚本
python scripts/start_server.py

# 方法2: 使用 uvicorn
uvicorn engine.server:app --host 0.0.0.0 --port 8000 --reload
```

后端将在 `http://localhost:8000` 启动。

### 6. 启动前端（开发模式）

```bash
cd frontend
npm install
npm run dev
```

前端将在 `http://localhost:5173` 启动。

### 7. 构建前端（生产模式）

```bash
cd frontend
npm run build
```

构建产物将输出到 `frontend/dist/` 目录。

## 🐳 Docker 部署

### 构建镜像

```bash
docker build -t neurobreak:latest -f docker/Dockerfile .
```

### 运行容器

```bash
docker run -it --gpus all \
  -p 8000:8000 \
  -v /path/to/models:/cache \
  -v /path/to/workspace/hf_models:/workspace/hf_models \
  -e MODELSCOPE_TOKEN=your_token \
  neurobreak:latest
```

**注意**：
- 模型路径已更新为 `/workspace/ms_models`，请确保正确挂载模型目录
- 推荐使用 ModelScope token（`MODELSCOPE_TOKEN`），中国大陆访问速度更快
- 如果使用 HuggingFace，可设置 `HF_TOKEN` 环境变量

详细部署指南请参考 [DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)。

## 📁 项目结构

```
NeuroBreak-Reproduction/
├── engine/                 # 后端服务
│   ├── server.py          # FastAPI 应用主文件
│   ├── models.py          # 模型管理模块
│   └── README.md          # 后端文档
├── frontend/              # 前端应用
│   ├── src/
│   │   ├── components/    # React 组件
│   │   ├── lib/           # API 客户端
│   │   ├── store/         # 状态管理
│   │   └── types/         # TypeScript 类型定义
│   └── package.json
├── docs/                  # 文档目录
│   ├── QUICK_START.md     # 快速启动指南
│   ├── DEPLOYMENT_GUIDE.md # 部署指南
│   ├── SALAD_EVALUATION_GUIDE.md # SALAD 评估指南
│   ├── SALAD_EVALUATION_ANALYSIS.md # SALAD 评估分析报告
│   └── ...
├── scripts/               # 工具脚本
│   ├── start_server.py    # 服务器启动脚本
│   ├── download_models.py # 模型下载脚本
│   ├── evaluate_salad_pipeline.py # SALAD 评估脚本
│   ├── analyze_salad_results.py # SALAD 结果分析脚本
│   └── ...
├── hf_models/            # 模型文件目录
│   ├── Meta-Llama-3-8B-Instruct/
│   └── Llama-Guard-3-8B/
├── docker/                # Docker 配置
│   └── Dockerfile
├── requirements.txt       # Python 依赖
└── README.md             # 本文件
```

## 🔌 API 文档

### 健康检查

```bash
GET /health
```

### 推理 + 审核联合流程

```bash
POST /api/pipeline/run
Content-Type: application/json

{
  "prompt": "用户输入文本",
  "inferenceConfig": {
    "modelId": "LLM-Research/Meta-Llama-3-8B-Instruct",
    "temperature": 0.7,
    "topP": 0.9,
    "maxTokens": 512,
    "stream": false
  },
  "guardConfig": {
    "modelId": "LLM-Research/Llama-Guard-3-8B",
    "threshold": 0.7,
    "autoBlock": false,
    "categories": ["violence", "politics"]
  }
}
```

### 独立安全审核

```bash
POST /api/moderate
Content-Type: application/json

{
  "text": "待审核文本",
  "threshold": 0.7,
  "categories": ["violence", "politics"]
}
```

更多 API 详情请参考 [engine/README.md](engine/README.md)。

## 📚 文档

- [快速启动指南](docs/QUICK_START.md) - 快速上手指南
- [部署指南](docs/DEPLOYMENT_GUIDE.md) - 详细部署说明
- [模型适配总结](docs/MODEL_ADAPTATION_SUMMARY.md) - 模型配置说明
- [Docker 模型挂载](docs/DOCKER_MODEL_MOUNT.md) - Docker 模型管理
- [SALAD 评估指南](docs/SALAD_EVALUATION_GUIDE.md) - SALAD-Bench 数据集评估指南
- [SALAD 评估分析](docs/SALAD_EVALUATION_ANALYSIS.md) - SALAD 评估结果分析报告

## 🧪 测试

### IO 测试

运行 IO 测试：

```bash
python scripts/run_io_tests.py
```

### SALAD-Bench 评估

运行 SALAD-Bench 数据集评估（需要先下载数据集）：

```bash
# 在 Docker 容器内运行
docker exec -it neurobreak-container /bin/bash
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

# TypeScript/React
cd frontend
npm run lint
```

### 检查模型

```bash
python scripts/check_models.py
```

## ⚠️ 注意事项

1. **模型访问权限**：需要申请 Meta Llama 和 Llama Guard 模型的访问权限
   - ModelScope（推荐）：访问 https://modelscope.cn 申请模型权限
   - HuggingFace：访问 https://huggingface.co 申请模型权限
2. **模型路径**：模型默认路径为 `/workspace/ms_models`（容器内）或 `hf_models/`（本地）
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
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Python Web 框架
- [React](https://react.dev/) - 前端 UI 框架

## 📮 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**

