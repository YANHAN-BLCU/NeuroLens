# Docker 容器模型挂载配置指南

本文档说明如何将 Windows F 盘的本地模型挂载到 Docker 容器内，以便测试脚本能够使用本地模型。

## 模型路径配置

### Windows 本地路径（F盘）
- 推理模型: `F:/models/meta-llama_Llama-3.2-3B-Instruct`
- 安全分类器: `F:/models/meta-llama_Llama-Guard-3-1B`

### Docker 容器内路径
- 推理模型: `/workspace/models/meta-llama_Llama-3.2-3B-Instruct`
- 安全分类器: `/workspace/models/meta-llama_Llama-Guard-3-1B`

## Docker 容器启动命令（Windows）

在 Windows 上启动 Docker 容器时，需要将 F 盘挂载到容器内：

```bash
docker run --gpus all -it \
  --name neurobreak-container \
  -p 8000:8000 \
  -p 4173:4173 \
  -v E:\BLCU\DC25\part1\NeuroBreak-Reproduction:/workspace \
  -v F:/models:/workspace/models \
  -v /opt/nb-cache:/workspace/.cache \
  neurobreak:latest \
  /bin/bash
```

**关键参数说明：**
- `-v F:/models:/workspace/models`: 将 Windows F 盘的 `models` 目录挂载到容器内的 `/workspace/models`
- `-v E:\BLCU\DC25\part1\NeuroBreak-Reproduction:/workspace`: 将项目代码挂载到容器内

## 环境变量配置（可选）

如果需要自定义模型路径，可以在启动容器时设置环境变量：

```bash
docker run --gpus all -it \
  --name neurobreak-container \
  -p 8000:8000 \
  -p 4173:4173 \
  -e LLM_LOCAL_PATH="F:/models/meta-llama_Llama-3.2-3B-Instruct" \
  -e GUARD_LOCAL_PATH="F:/models/meta-llama_Llama-Guard-3-1B" \
  -e LLM_CONTAINER_PATH="/workspace/models/meta-llama_Llama-3.2-3B-Instruct" \
  -e GUARD_CONTAINER_PATH="/workspace/models/meta-llama_Llama-Guard-3-1B" \
  -v E:\BLCU\DC25\part1\NeuroBreak-Reproduction:/workspace \
  -v F:/models:/workspace/models \
  neurobreak:latest \
  /bin/bash
```

## 验证模型挂载

进入容器后，验证模型是否已正确挂载：

```bash
# 检查容器内的模型路径
ls -lh /workspace/models/

# 应该能看到：
# meta-llama_Llama-3.2-3B-Instruct/
# meta-llama_Llama-Guard-3-1B/

# 检查模型文件
ls -lh /workspace/models/meta-llama_Llama-3.2-3B-Instruct/
ls -lh /workspace/models/meta-llama_Llama-Guard-3-1B/
```

## 运行测试脚本

挂载完成后，在容器内运行测试脚本：

```bash
# 运行 I/O 测试
python scripts/run_io_tests.py
```

脚本会自动检测并使用挂载的本地模型路径。

## 注意事项

1. **路径格式**: Windows 路径使用正斜杠 `/` 或反斜杠 `\\`，但在 Docker 挂载时建议使用正斜杠
2. **权限问题**: 确保 Docker Desktop 有权限访问 F 盘（在 Docker Desktop 设置中配置共享驱动器）
3. **路径一致性**: 确保 F 盘上的模型目录结构与脚本中配置的路径一致
4. **模型命名**: 模型目录名应该是 `meta-llama_Llama-3.2-3B-Instruct` 和 `meta-llama_Llama-Guard-3-1B`（使用下划线替换斜杠）

## 故障排查

### 问题1: 容器内看不到模型文件
- 检查 Docker Desktop 是否已启用 F 盘共享
- 检查挂载命令中的路径是否正确
- 在容器内运行 `ls -la /workspace/models/` 查看挂载情况

### 问题2: 脚本仍从 HuggingFace 下载模型
- 检查模型路径是否正确
- 检查环境变量是否设置正确
- 查看脚本输出的 `[模型路径]` 日志信息

### 问题3: 权限错误
- 确保 Docker Desktop 有 F 盘访问权限
- 在 Windows 上检查 F 盘的共享和权限设置

