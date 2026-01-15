# 模型适配总结

本文档总结了将本地模型适配到评估系统的完整过程。

## 已完成的修改

### 1. 测试脚本适配

#### `scripts/run_io_tests.py`
- ✅ 添加了本地模型路径检测功能
- ✅ 支持从 `/workspace/ms_models` 路径加载模型
- ✅ 修复了 `torch_dtype` deprecation 警告（使用 `dtype` 代替）

#### `scripts/check_models.py`
- ✅ 新增模型路径检查工具
- ✅ 自动检测多个可能的模型路径
- ✅ 显示详细的模型信息（大小、文件数等）

#### `scripts/download_models.py`
- ✅ 支持通过环境变量 `LOCAL_MODEL_PATH` 指定本地模型路径
- ✅ 如果模型已存在，自动跳过下载
- ✅ 支持 ModelScope 和 HuggingFace

### 2. 模型管理适配

#### `engine/models.py`
- ✅ 添加了 `get_model_path()` 函数，自动检测模型路径
- ✅ 修改 `load_llm()` 方法，使用本地模型路径
- ✅ 修改 `load_guard()` 方法，使用本地模型路径
- ✅ 修复了 `torch_dtype` deprecation 警告（使用 `dtype` 代替）
- ✅ 支持环境变量自定义模型路径
- ✅ 支持 4-bit 量化以降低显存占用
- ✅ 改进的设备检测逻辑，支持量化模型

### 3. 文档更新

- ✅ `docs/DEPLOYMENT_GUIDE.md` - 更新了模型路径配置
- ✅ `docs/SYNC_FILES_TO_DOCKER.md` - 文件同步到 Docker 容器指南

### 4. 同步脚本

- ✅ `scripts/sync_to_docker.ps1` - Windows PowerShell 同步脚本
- ✅ `scripts/sync_to_docker.sh` - Linux/Mac Bash 同步脚本

## 模型路径配置

### 当前容器内的模型位置
- 推理模型: `/workspace/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct`
- 安全分类器: `/workspace/ms_models/LLM-Research/Llama-Guard-3-8B`

### 路径检测优先级
1. `/workspace/ms_models` - 工作空间模型目录（优先）
2. ModelScope 缓存目录 - `~/.cache/modelscope/hub`
3. HuggingFace 缓存目录 - `~/.cache/huggingface`
4. ModelScope ID - 如果本地路径不存在，回退到在线下载

### 环境变量支持
可以通过环境变量自定义模型路径：
```bash
export LLM_CONTAINER_PATH="/workspace/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
export GUARD_CONTAINER_PATH="/workspace/ms_models/LLM-Research/Llama-Guard-3-8B"
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

### 3. 测试模型加载
```bash
python -c "from engine.models import ModelManager; mm = ModelManager(); mm.load_llm(); mm.load_guard()"
```

## 评估工作流程

1. **模型加载** → `ModelManager` 自动检测并使用本地模型路径
2. **模型推理** → 从 `/workspace/ms_models` 加载模型（如果存在）
3. **评估执行** → 使用本地模型进行推理和审核
4. **结果输出** → 保存评估结果到日志文件

## 注意事项

1. **模型路径**: 确保模型在 `/workspace/ms_models` 目录下，目录结构为 `LLM-Research/Meta-Llama-3-8B-Instruct`
2. **文件同步**: 修改代码后，使用 `.\scripts\sync_to_docker.ps1` 同步到容器
3. **环境变量**: 如果模型路径不同，可以通过环境变量自定义
4. **首次加载**: 模型首次加载需要一些时间，后续请求会使用已加载的模型（单例模式）
5. **显存管理**: 8B 模型使用 4-bit 量化，降低显存占用

## 故障排查

### 问题1: 模型仍从在线下载
- 检查模型路径是否正确
- 运行 `python scripts/check_models.py` 查看路径状态
- 检查环境变量是否设置正确

### 问题2: 模型加载失败
- 检查模型文件是否完整（需要 `config.json` 和 `.safetensors` 文件）
- 检查文件权限
- 查看日志中的错误信息

### 问题3: 显存不足
- 确保已启用 4-bit 量化（默认启用）
- 检查 GPU 显存是否足够（建议 16GB+）
- 如果显存不足，Guard 模型会自动回退到 CPU

## 下一步

现在模型系统已经完全适配本地模型，可以：
1. 运行 SALAD-Bench 评估实验
2. 分析评估结果
3. 进行模型性能优化实验
