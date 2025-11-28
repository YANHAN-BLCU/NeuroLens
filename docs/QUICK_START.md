# 快速启动指南

## 前提条件

✅ 模型已挂载到容器内的 `/cache/` 目录
✅ 脚本已同步到容器内

## 启动步骤

### 1. 检查模型路径（可选）

```bash
docker exec -it neurobreak-container /bin/bash
python scripts/check_models.py
```

应该看到：
- ✓ 推理模型（容器内 - /cache）
- ✓ 安全分类器（容器内 - /cache）

### 2. 启动后端服务器

在容器内执行：

```bash
# 方法1: 使用启动脚本
python scripts/start_server.py

# 方法2: 使用 uvicorn
uvicorn engine.server:app --host 0.0.0.0 --port 8000

# 方法3: 后台运行
nohup uvicorn engine.server:app --host 0.0.0.0 --port 8000 > /tmp/server.log 2>&1 &
```

后端将在 `http://0.0.0.0:8000` 启动。

### 3. 验证后端运行

在另一个终端：

```bash
# 健康检查
curl http://localhost:8000/health

# 或访问
curl http://localhost:8000/
```

应该返回：
```json
{
  "status": "ok",
  "service": "NeuroBreak API",
  "version": "1.0.0"
}
```

### 4. 启动前端（如果需要）

前端通常需要单独启动，请参考 `frontend/README.md`。

## 测试 API

### 测试推理 + Guard 联合流程

```bash
curl -X POST http://localhost:8000/api/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请用中文解释什么是机器学习",
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
  }'
```

### 测试独立审核

```bash
curl -X POST http://localhost:8000/api/moderate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "这是一个测试文本",
    "threshold": 0.5,
    "categories": ["violence", "politics"]
  }'
```

## 常见问题

### Q: 后端启动失败
- 检查端口 8000 是否被占用
- 检查模型路径是否正确
- 查看错误日志

### Q: 模型加载很慢
- 首次加载需要时间，这是正常的
- 后续请求会使用已加载的模型（单例模式）

### Q: 如何查看后端日志
```bash
# 如果使用 nohup 后台运行
tail -f /tmp/server.log

# 如果直接运行，日志会直接输出到终端
```

## 下一步

- 查看 [模型适配总结](MODEL_ADAPTATION_SUMMARY.md) 了解详细配置
- 查看 [部署指南](DEPLOYMENT_GUIDE.md) 了解完整部署流程

