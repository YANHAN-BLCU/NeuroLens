# 模型适配总结

本文档总结了将本地模型（F盘）适配到前后端的完整过程。

## 已完成的修改

### 1. 测试脚本适配

#### `scripts/run_io_tests.py`
- ✅ 添加了本地模型路径检测功能
- ✅ 支持从 `/cache/` 路径加载模型（当前容器挂载点）
- ✅ 支持从 `/workspace/models/` 路径加载模型（备用路径）
- ✅ 修复了 `torch_dtype` deprecation 警告（使用 `dtype` 代替）

#### `scripts/check_models.py`
- ✅ 新增模型路径检查工具
- ✅ 自动检测多个可能的模型路径
- ✅ 显示详细的模型信息（大小、文件数等）

#### `scripts/download_models.py`
- ✅ 支持通过环境变量 `LOCAL_MODEL_PATH` 指定本地模型路径
- ✅ 如果模型已存在，自动跳过下载

### 2. 后端适配

#### `engine/models.py`
- ✅ 添加了 `get_model_path()` 函数，自动检测模型路径
- ✅ 修改 `load_llm()` 方法，使用本地模型路径
- ✅ 修改 `load_guard()` 方法，使用本地模型路径
- ✅ 修复了 `torch_dtype` deprecation 警告（使用 `dtype` 代替）
- ✅ 支持环境变量自定义模型路径

### 3. 文档更新

- ✅ `docs/DOCKER_MODEL_MOUNT.md` - Docker 模型挂载配置指南
- ✅ `docs/SYNC_FILES_TO_DOCKER.md` - 文件同步到 Docker 容器指南
- ✅ `docs/DEPLOYMENT_GUIDE.md` - 更新了 Windows 本地开发配置

### 4. 同步脚本

- ✅ `scripts/sync_to_docker.ps1` - Windows PowerShell 同步脚本
- ✅ `scripts/sync_to_docker.sh` - Linux/Mac Bash 同步脚本

## 模型路径配置

### 当前容器内的模型位置
- 推理模型: `/cache/meta-llama_Llama-3.2-3B-Instruct`
- 安全分类器: `/cache/meta-llama_Llama-Guard-3-1B`

### 路径检测优先级
1. `/cache/` - 当前容器的挂载点（优先）
2. `/workspace/models/` - 备用路径
3. HuggingFace ID - 如果本地路径不存在，回退到在线下载

### 环境变量支持
可以通过环境变量自定义模型路径：
```bash
export LLM_CONTAINER_PATH="/cache/meta-llama_Llama-3.2-3B-Instruct"
export GUARD_CONTAINER_PATH="/cache/meta-llama_Llama-Guard-3-1B"
```

## 验证测试

### 1. 检查模型路径
```bash
python scripts/check_models.py
```

### 2. 运行 I/O 测试
```bash
python scripts/run_io_tests.py
```

### 3. 测试后端模型加载
```bash
python -c "from engine.models import ModelManager; mm = ModelManager(); mm.load_llm(); mm.load_guard()"
```

### 4. 启动后端服务器
```bash
python scripts/start_server.py
# 或
uvicorn engine.server:app --host 0.0.0.0 --port 8000
```

## 前后端工作流程

1. **前端** → 发送请求到 `/api/pipeline/run`
2. **后端** → `ModelManager` 自动检测并使用本地模型路径
3. **模型加载** → 从 `/cache/` 加载模型（如果存在）
4. **推理执行** → 使用本地模型进行推理和审核
5. **返回结果** → 前端接收并显示结果

## 注意事项

1. **模型路径**: 确保模型在 `/cache/` 目录下，目录名格式为 `meta-llama_Llama-3.2-3B-Instruct`（使用下划线）
2. **文件同步**: 修改代码后，使用 `.\scripts\sync_to_docker.ps1` 同步到容器
3. **环境变量**: 如果模型路径不同，可以通过环境变量自定义
4. **首次加载**: 模型首次加载需要一些时间，后续请求会使用已加载的模型（单例模式）

## 故障排查

### 问题1: 后端仍从 HuggingFace 下载模型
- 检查模型路径是否正确
- 运行 `python scripts/check_models.py` 查看路径状态
- 检查环境变量是否设置正确

### 问题2: 模型加载失败
- 检查模型文件是否完整（需要 `config.json` 和 `.safetensors` 文件）
- 检查文件权限
- 查看后端日志中的错误信息

### 问题3: 前后端无法通信
- 检查后端是否正常启动（端口 8000）
- 检查 CORS 配置
- 检查前端 API 基础 URL 配置

## 下一步

现在前后端已经完全适配本地模型，可以：
1. 启动后端服务器
2. 启动前端应用
3. 在浏览器中测试完整的工作流程

