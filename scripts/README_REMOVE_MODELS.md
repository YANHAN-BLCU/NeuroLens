# 删除模型脚本使用指南

本目录包含用于删除本地和 Docker 容器内模型文件的脚本。

## 脚本说明

### 1. `remove_models.ps1` - Windows 本地删除脚本

在 Windows 本地删除模型文件。

**使用方法：**
```powershell
# 交互式删除（会提示确认）
.\scripts\remove_models.ps1

# 直接删除（跳过确认）
.\scripts\remove_models.ps1 --confirm

# 删除指定模型
.\scripts\remove_models.ps1 --Models "meta-llama_Llama-3.2-3B-Instruct"
```

**搜索路径：**
- `F:/models/`
- `%USERPROFILE%\.cache\huggingface\models\`
- `%HF_HOME%\models\`
- `%TRANSFORMERS_CACHE%\models\`
- 项目内的 `hf_models/` 目录

**默认删除的模型：**
- `meta-llama_Llama-3.2-3B-Instruct`
- `meta-llama_Llama-Guard-3-1B`
- `Meta-Llama-3-8B-Instruct`
- `Llama-Guard-3-8B`

### 2. `remove_models.sh` - Docker 容器内删除脚本

在 Docker 容器内删除模型文件。

**使用方法：**

**在容器内执行：**
```bash
# 进入容器
docker exec -it <container_name> bash

# 在容器内执行
bash /workspace/scripts/remove_models.sh

# 直接删除（跳过确认）
bash /workspace/scripts/remove_models.sh --confirm
```

**从宿主机执行：**
```bash
docker exec -it <container_name> bash /workspace/scripts/remove_models.sh
```

**搜索路径：**
- `/cache/`
- `/workspace/hf_models/`
- `/workspace/models/`
- `$HOME/.cache/huggingface/models/`
- `$HF_HOME/models/`
- `$TRANSFORMERS_CACHE/models/`

### 3. `remove_models_from_container.ps1` - 从 Windows 主机删除容器内模型

从 Windows 主机在 Docker 容器内删除模型文件。

**使用方法：**
```powershell
# 使用默认容器名称
.\scripts\remove_models_from_container.ps1

# 指定容器名称
.\scripts\remove_models_from_container.ps1 -ContainerName "my-container"

# 直接删除（跳过确认）
.\scripts\remove_models_from_container.ps1 --confirm
```

## 快速使用

### 删除本地模型
```powershell
.\scripts\remove_models.ps1
```

### 删除容器内模型
```powershell
.\scripts\remove_models_from_container.ps1
```

### 同时删除本地和容器内模型
```powershell
# 先删除本地
.\scripts\remove_models.ps1 --confirm

# 再删除容器内
.\scripts\remove_models_from_container.ps1 --confirm
```

## 注意事项

1. **备份重要数据**：删除操作不可逆，请确保已备份重要模型文件。

2. **容器状态**：删除容器内模型时，容器需要处于运行状态。如果容器未运行，脚本会提示是否启动容器。

3. **权限问题**：确保有足够的权限删除模型文件。如果遇到权限问题，可能需要以管理员身份运行。

4. **磁盘空间**：删除模型后会释放大量磁盘空间，脚本会显示释放的空间大小。

5. **模型路径**：脚本会自动搜索多个可能的模型路径，如果模型在其他位置，可能需要手动删除。

## 示例输出

```
========================================
删除本地模型文件
========================================

查找模型: meta-llama_Llama-3.2-3B-Instruct
  找到: F:/models/meta-llama_Llama-3.2-3B-Instruct (大小: 6.5 GB)
  找到: hf_models\meta-llama_Llama-3.2-3B-Instruct (大小: 6.5 GB)

将要删除以下模型:
  - meta-llama_Llama-3.2-3B-Instruct: F:/models/meta-llama_Llama-3.2-3B-Instruct (6.5 GB)
  - meta-llama_Llama-Guard-3-1B: F:/models/meta-llama_Llama-Guard-3-1B (2.1 GB)

总计大小: 8.6 GB

确认删除以上模型? (yes/no): yes

开始删除...
删除: F:/models/meta-llama_Llama-3.2-3B-Instruct
  ✓ 已删除: meta-llama_Llama-3.2-3B-Instruct
删除: F:/models/meta-llama_Llama-Guard-3-1B
  ✓ 已删除: meta-llama_Llama-Guard-3-1B

========================================
删除完成
========================================
成功删除: 2 个模型
释放空间: 8.6 GB
```

