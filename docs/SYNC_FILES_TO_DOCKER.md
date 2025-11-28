# 同步文件到 Docker 容器指南

当修改了脚本文件后，需要确保文件同步到 Docker 容器内。有几种方法：

## 方法1: 使用 Volume 挂载（推荐，自动同步）

如果容器启动时使用了 `-v` 参数挂载了项目目录，文件会自动同步。

### 检查容器是否使用了挂载

```bash
# 在 Windows PowerShell 中检查容器挂载
docker inspect neurobreak-container | Select-String -Pattern "Mounts" -Context 0,20
```

或者：

```bash
# 在容器内检查
docker exec neurobreak-container ls -la /workspace/scripts/
```

### 如果容器没有使用挂载，需要重新启动容器

**停止并删除旧容器：**

```bash
docker stop neurobreak-container
docker rm neurobreak-container
```

**使用挂载重新启动容器：**

```bash
docker run --gpus all -it \
  --name neurobreak-container \
  -p 8000:8000 \
  -p 4173:4173 \
  -v E:\BLCU\DC25\part1\NeuroBreak-Reproduction:/workspace \
  -v F:/models:/workspace/models \
  neurobreak:latest \
  /bin/bash
```

这样，Windows 上的文件修改会自动反映到容器内。

## 方法2: 手动复制文件到容器（临时方案）

如果容器已经在运行且不想重启，可以手动复制文件：

### 复制单个文件

```bash
# 在 Windows PowerShell 中执行
docker cp scripts/run_io_tests.py neurobreak-container:/workspace/scripts/run_io_tests.py
docker cp scripts/download_models.py neurobreak-container:/workspace/scripts/download_models.py
docker cp scripts/check_models.py neurobreak-container:/workspace/scripts/check_models.py
```

### 复制整个 scripts 目录

```bash
# 在 Windows PowerShell 中执行
docker cp scripts neurobreak-container:/workspace/
```

### 验证文件已复制

```bash
# 进入容器
docker exec -it neurobreak-container /bin/bash

# 检查文件
ls -lh /workspace/scripts/run_io_tests.py
cat /workspace/scripts/run_io_tests.py | head -30
```

## 方法3: 使用 docker exec 直接编辑（不推荐）

如果只是小修改，可以在容器内直接编辑：

```bash
# 进入容器
docker exec -it neurobreak-container /bin/bash

# 安装编辑器（如果没有）
apt-get update && apt-get install -y vim nano

# 编辑文件
vim /workspace/scripts/run_io_tests.py
```

**注意**: 这种方法修改不会保存到 Windows 文件系统。

## 方法4: 使用 docker-compose（推荐用于生产环境）

创建 `docker-compose.yml` 文件：

```yaml
version: '3.8'

services:
  neurobreak:
    image: neurobreak:latest
    container_name: neurobreak-container
    runtime: nvidia
    ports:
      - "8000:8000"
      - "4173:4173"
    volumes:
      - E:\BLCU\DC25\part1\NeuroBreak-Reproduction:/workspace
      - F:/models:/workspace/models
    environment:
      - LLM_CONTAINER_PATH=/workspace/models/meta-llama_Llama-3.2-3B-Instruct
      - GUARD_CONTAINER_PATH=/workspace/models/meta-llama_Llama-Guard-3-1B
    stdin_open: true
    tty: true
    command: /bin/bash
```

然后使用：

```bash
# 启动容器
docker-compose up -d

# 进入容器
docker-compose exec neurobreak /bin/bash

# 停止容器
docker-compose down
```

## 快速检查脚本

在容器内运行检查脚本，确认文件已同步：

```bash
# 在容器内执行
python scripts/check_models.py

# 检查脚本内容
head -30 /workspace/scripts/run_io_tests.py
```

## 常见问题

### Q: 为什么修改了文件但容器内看不到？

A: 可能的原因：
1. 容器没有使用 volume 挂载
2. 挂载路径不正确
3. 文件权限问题

**解决方案**: 使用 `docker inspect` 检查挂载配置，或重新启动容器。

### Q: 如何确认文件已同步？

A: 在容器内检查文件修改时间：

```bash
# 在容器内
ls -lh /workspace/scripts/run_io_tests.py
stat /workspace/scripts/run_io_tests.py
```

### Q: 容器重启后文件丢失？

A: 如果容器没有使用 volume 挂载，重启后文件会丢失。确保使用 `-v` 参数挂载目录。

## 推荐工作流程

1. **开发时**: 使用 volume 挂载，文件自动同步
2. **测试时**: 在容器内运行测试脚本
3. **部署时**: 使用 docker-compose 管理容器

