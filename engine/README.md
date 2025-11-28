# NeuroBreak Engine

后端推理和审核服务，基于 FastAPI 实现。

## 文件结构

- `server.py` - FastAPI 应用主文件，提供 API 端点
- `models.py` - 模型管理模块，负责加载和使用 Llama 推理模型和 Llama Guard 审核模型
- `__init__.py` - 包初始化文件

## API 端点

### POST `/api/pipeline/run`
执行推理 + Guard 联合流程

**请求体：**
```json
{
  "prompt": "用户输入文本",
  "context": "可选上下文",
  "systemPrompt": "可选系统提示",
  "inferenceConfig": {
    "modelId": "meta-llama/Llama-3.2-3B-Instruct",
    "temperature": 0.7,
    "topP": 0.9,
    "topK": 50,
    "maxTokens": 512,
    "repetitionPenalty": 1.1,
    "presencePenalty": 0.0,
    "frequencyPenalty": 0.0,
    "stopSequences": [],
    "stream": false
  },
  "guardConfig": {
    "modelId": "meta-llama/Llama-Guard-3-1B",
    "threshold": 0.5,
    "autoBlock": false,
    "categories": ["violence", "politics"]
  }
}
```

### POST `/api/moderate`
独立安全审核文本

**请求体：**
```json
{
  "text": "待审核文本",
  "threshold": 0.5,
  "categories": ["violence", "politics"]
}
```

## 启动方式

1. **使用 uvicorn（推荐）**
   ```bash
   uvicorn engine.server:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **使用启动脚本**
   ```bash
   python scripts/start_server.py
   ```

3. **直接运行模块**
   ```bash
   python -m engine.server
   ```

## 环境要求

- Python 3.9+
- PyTorch（支持 CUDA 更佳）
- Transformers
- FastAPI
- Uvicorn

所有依赖已在 `requirements.txt` 中列出。

## 模型加载

首次运行时，模型管理器会自动下载并加载：
- `meta-llama/Llama-3.2-3B-Instruct` - 推理模型
- `meta-llama/Llama-Guard-3-1B` - 安全审核模型

确保已配置 HuggingFace Token 并申请了模型访问权限。

## 注意事项

- 模型加载需要一定时间，首次请求可能较慢
- 建议使用 GPU 加速推理
- 模型会懒加载，只在首次使用时加载

