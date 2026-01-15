# NeuroLens 部署与使用文档

本指南覆盖从准备代码、上传到计算容器、构建运行环境，到模型评估实验的完整流程。按顺序执行可以在 GPU 服务器上复现模型评估环境。

## 目录
1. [项目组件速览](#项目组件速览)
2. [前置条件](#前置条件)
3. [准备与上传项目文件](#准备与上传项目文件)
4. [构建 Docker 镜像](#构建-docker-镜像)
5. [启动容器并初始化环境](#启动容器并初始化环境)
6. [下载模型与健康检查](#下载模型与健康检查)
7. [运行评估实验](#运行评估实验)
8. [更新与维护](#更新与维护)
9. [常见问题排查](#常见问题排查)

## 项目组件速览

| 模块 | 位置 | 说明 |
| --- | --- | --- |
| 容器环境 | `docker/Dockerfile` | 基于 `nvidia/cuda:12.4.0-cudnn8-runtime-ubuntu22.04`，预装 PyTorch、Transformers 等依赖。 |
| 推理与审核脚本 | `scripts/download_models.py`、`scripts/run_io_tests.py` | 下载 Meta-Llama-3-8B-Instruct 与 Llama-Guard-3-8B，提供 I/O 冒烟测试。 |
| 评估脚本 | `scripts/evaluate_salad_pipeline.py` | SALAD-Bench 数据集评估脚本。 |
| 数据/缓存 | `data/`、`notebooks/` | 模型缓存、测试结果输出目录，可通过挂载映射到宿主机。 |

## 前置条件

- 一台具备 NVIDIA GPU 的 Linux 服务器（推荐 16GB+ 显存），已安装 Docker ≥ 24.0 与 `nvidia-container-toolkit`。
- 本地开发机安装 Git、`scp/rsync` 等上传工具。
- ModelScope 账号（推荐，中国大陆访问更快）或 HuggingFace 账号，已申请模型访问权限。
  - ModelScope: `LLM-Research/Meta-Llama-3-8B-Instruct` 与 `LLM-Research/Llama-Guard-3-8B`
  - HuggingFace: `meta-llama/Meta-Llama-3-8B-Instruct` 与 `meta-llama/Llama-Guard-3-8B`

## 准备与上传项目文件

1. **获取代码**  
   ```bash
   git clone https://github.com/YANHAN-BLCU/NeuroBreak-Reproduction-.git
   cd NeuroBreak-Reproduction
   ```

2. **打包（可选）**  
   ```bash
   tar czf neurobreak.tar.gz NeuroBreak-Reproduction
   ```

3. **上传到服务器**  
   ```bash
   scp -r NeuroBreak-Reproduction user@SERVER_IP:/opt/
   # 或者使用 rsync，便于后续增量同步
   rsync -av --progress NeuroBreak-Reproduction/ user@SERVER_IP:/opt/NeuroBreak-Reproduction
   ```

4. **服务器端校验**  
   ```bash
   ssh user@SERVER_IP
   cd /opt/NeuroBreak-Reproduction
   ls
   ```

## 构建 Docker 镜像

在服务器根目录执行：

```bash
cd /opt/NeuroBreak-Reproduction
docker build -t neurobreak:latest -f docker/Dockerfile .
```

> 提示：如需加速，可在构建命令前设置国内镜像源或传入 `--build-arg https_proxy`。

## 启动容器并初始化环境

建议将代码目录与缓存/数据目录通过 Volume 映射到宿主机，方便持久化：

### Linux 服务器

```bash
docker run --gpus all -it \
  --name neurobreak \
  -v /opt/NeuroBreak-Reproduction:/workspace \
  -v /opt/nb-cache:/workspace/.cache \
  -e MODELSCOPE_TOKEN=your_token \
  neurobreak:latest \
  /bin/bash
```

### Windows 本地开发

如果模型已下载到本地，需要将模型目录挂载到容器内：

```bash
docker run --gpus all -it \
  --name neurobreak-container \
  -v E:\BLCU\DC25\part1\NeuroBreak-Reproduction:/workspace \
  -v F:/models:/workspace/ms_models \
  -v /opt/nb-cache:/workspace/.cache \
  -e MODELSCOPE_TOKEN=your_token \
  neurobreak:latest \
  /bin/bash
```

**注意**: 
- 确保 Docker Desktop 已启用相应盘符的共享（Settings → Resources → File Sharing）
- 模型路径应为: `/workspace/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct` 和 `/workspace/ms_models/LLM-Research/Llama-Guard-3-8B`

进入容器后，默认工作目录为 `/workspace`。可以创建 `.env` 记录敏感配置（如 ModelScope Token）：

```bash
cat <<'EOF' > /workspace/.env
MODELSCOPE_TOKEN=your_modelscope_token
# 或者使用 HuggingFace Token（如果使用 HuggingFace）
# HF_TOKEN=hf_xxx
EOF
```

> `.env` 仅在容器内可读，确保不要提交到 Git。

## 下载模型与健康检查

### 方式1: 使用本地模型（推荐，如果模型已下载）

如果模型已下载到本地，启动容器时已挂载，直接运行测试：

```bash
python scripts/run_io_tests.py
# 结果输出 notebooks/io_test_results.json
```

脚本会自动检测并使用挂载的本地模型路径。

### 方式2: 在容器内下载模型

1. **设置 ModelScope Token（推荐，中国大陆访问更快）**
   ```bash
   export MODELSCOPE_TOKEN=your_modelscope_token
   ```
   
   或者在启动容器时设置：
   ```bash
   docker run --gpus all -it \
     -e MODELSCOPE_TOKEN=your_modelscope_token \
     ...
   ```

2. **下载模型（使用 ModelScope）**
   ```bash
   # 下载到容器内的缓存目录
   python scripts/download_models.py --all-8b \
     --output /workspace/ms_models
   ```
   
   **使用 HuggingFace（备选方案）**
   
   如果需要使用 HuggingFace，需要先登录：
   ```bash
   export HF_TOKEN=hf_xxx
   huggingface-cli login --token $HF_TOKEN
   ```

3. **运行 I/O 冒烟测试**
   ```bash
   python scripts/run_io_tests.py
   # 结果输出 notebooks/io_test_results.json
   ```

如需英文版或更多 prompt，可运行 `scripts/run_io_tests_2.py`。

## 运行评估实验

### SALAD-Bench 数据集评估

1. **下载 SALAD-Bench 数据集**（如果尚未下载）
   ```bash
   python scripts/download_salad.py
   ```

2. **运行评估**
   ```bash
   python scripts/evaluate_salad_pipeline.py \
     --data_dir /workspace/data/salad/raw \
     --output /workspace/logs/salad_evaluation.jsonl \
     --config base_set \
     --max_samples 100
   ```

3. **分析结果**
   ```bash
   python scripts/analyze_salad_results.py \
     --input /workspace/logs/salad_evaluation.jsonl
   ```

详细说明请参考 [SALAD 评估指南](SALAD_EVALUATION_GUIDE.md)。

## 更新与维护

- **更新代码**：在宿主机同步后，进入容器执行 `git pull`。
- **更新依赖**：通过 `pip install -r requirements.txt --upgrade`。
- **重启容器**：
  ```bash
  docker restart neurobreak
  docker exec -it neurobreak /bin/bash
  ```
- **备份数据**：定期复制 `/workspace/notebooks`、`/workspace/data`、`/workspace/logs` 至安全位置。

## 常见问题排查

- **ModelScope 下载报 401/403**：确认已申请模型权限，并在容器内 `export MODELSCOPE_TOKEN=...` 后重试。
- **HuggingFace 下载报 401/403**（如果使用 HuggingFace）：确认已申请模型权限，并在容器内 `export HF_TOKEN=...` 后重试。
- **显存不足**：在 `scripts/evaluate_salad_pipeline.py` 中调低 `max_tokens` 或使用 4-bit 量化（已默认启用）。
- **模型加载失败**：检查模型路径是否正确，运行 `python scripts/check_models.py` 查看路径状态。

---

完成以上步骤后，即可在容器内运行模型评估实验。若需进一步自动化（如 docker compose 或 CI/CD），可在此文档基础上扩展。祝实验顺利！
