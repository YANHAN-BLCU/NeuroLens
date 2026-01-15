# NeuroLens Engine

模型管理模块，负责加载和使用 Llama 推理模型和 Llama Guard 审核模型。

## 文件结构

- `models.py` - 模型管理模块，负责加载和使用 Llama 推理模型和 Llama Guard 审核模型
- `__init__.py` - 包初始化文件
- `assessment/` - 评估模块，包含 SALAD 评估相关功能

## 核心功能

### ModelManager

`ModelManager` 是一个单例类，负责管理推理模型和安全审核模型：

- **推理模型**：基于 Meta Llama 3 8B 模型进行文本生成
- **安全审核模型**：基于 Llama Guard 3 8B 进行内容安全检测

### 主要方法

- `load_llm()` - 加载推理模型（懒加载）
- `load_guard()` - 加载安全审核模型（懒加载）
- `generate()` - 使用推理模型生成文本
- `moderate()` - 使用 Guard 模型审核文本

## 使用示例

```python
from engine.models import ModelManager

# 初始化模型管理器
model_manager = ModelManager()

# 生成文本
output_text, input_tokens, output_tokens, latency_ms = model_manager.generate(
    prompt="你好，请介绍一下人工智能",
    max_tokens=384,
    temperature=0.7,
    top_p=0.9,
)

# 审核文本
guard_result = model_manager.moderate(
    text=output_text,
    threshold=0.6,
)
```

## 环境要求

- Python 3.9+
- PyTorch（支持 CUDA 更佳）
- Transformers
- ModelScope 或 HuggingFace（用于模型下载）

所有依赖已在 `requirements.txt` 中列出。

## 模型配置

默认使用的模型：
- `LLM-Research/Meta-Llama-3-8B-Instruct` - 推理模型（ModelScope）
- `LLM-Research/Llama-Guard-3-8B` - 安全审核模型（ModelScope）

确保已配置 ModelScope Token 并申请了模型访问权限。

## 注意事项

- 模型加载需要一定时间，首次请求可能较慢
- 建议使用 GPU 加速推理（8B 模型需要 16GB+ VRAM）
- 模型会懒加载，只在首次使用时加载
- 支持 4-bit 量化以降低显存占用
