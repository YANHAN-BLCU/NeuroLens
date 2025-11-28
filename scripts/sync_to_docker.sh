#!/bin/bash
# Bash 脚本：同步文件到 Docker 容器
# 使用方法: bash scripts/sync_to_docker.sh

CONTAINER_NAME="${1:-neurobreak-container}"
WORKSPACE_PATH="/workspace"

echo "========================================"
echo "同步文件到 Docker 容器"
echo "========================================"
echo "容器名称: $CONTAINER_NAME"
echo "工作目录: $WORKSPACE_PATH"
echo ""

# 检查容器是否存在
if ! docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "错误: 容器 '$CONTAINER_NAME' 不存在！"
    echo "请先启动容器或检查容器名称。"
    exit 1
fi

# 检查容器是否运行
if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "警告: 容器 '$CONTAINER_NAME' 未运行，正在启动..."
    docker start $CONTAINER_NAME
    sleep 2
fi

echo "正在同步脚本文件..."

# 同步脚本文件
scripts=(
    "scripts/run_io_tests.py"
    "scripts/download_models.py"
    "scripts/check_models.py"
)

for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "  复制 $script ..."
        docker cp "$script" "${CONTAINER_NAME}:${WORKSPACE_PATH}/$script"
        if [ $? -eq 0 ]; then
            echo "  ✓ $script 已同步"
        else
            echo "  ✗ $script 同步失败"
        fi
    else
        echo "  ⚠ $script 不存在，跳过"
    fi
done

echo ""
echo "验证文件..."
docker exec $CONTAINER_NAME ls -lh "${WORKSPACE_PATH}/scripts/" | grep -E "run_io_tests|download_models|check_models"

echo ""
echo "========================================"
echo "同步完成！"
echo "========================================"
echo ""
echo "现在可以在容器内运行:"
echo "  docker exec -it $CONTAINER_NAME /bin/bash"
echo "  python scripts/run_io_tests.py"

