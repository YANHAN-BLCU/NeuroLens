#!/bin/bash
# Sync NeuroBreak-Reproduction files to Docker container
# Usage: ./scripts/sync_to_container.sh [container_name]

CONTAINER_NAME="${1:-neurobreak-container}"

echo "Syncing files to container: $CONTAINER_NAME"

# Check if container is running
if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container $CONTAINER_NAME is not running"
    exit 1
fi

# Key files and directories to sync
FILES=(
    "engine/assessment/evaluate.py"
    "engine/assessment/report.py"
    "scripts/download_salad.py"
    "scripts/eval_utility.py"
    "scripts/verify_salad.py"
    "scripts/download_models.py"
    "configs/runtime/salad.yaml"
)

DIRS=(
    "data/salad"
    "configs/runtime"
)

# Sync individual files
echo "Syncing files..."
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  Copying $file"
        docker cp "$file" "${CONTAINER_NAME}:/workspace/$file"
    else
        echo "  Warning: $file not found"
    fi
done

# Sync directories (if needed)
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "  Copying directory $dir"
        docker cp "$dir" "${CONTAINER_NAME}:/workspace/$(dirname $dir)/"
    fi
done

echo "Sync complete!"
echo "Verifying files in container..."

# Verify key files
docker exec "$CONTAINER_NAME" bash -c "
    echo 'Checking files...'
    test -f /workspace/engine/assessment/evaluate.py && echo '  ✓ evaluate.py' || echo '  ✗ evaluate.py missing'
    test -f /workspace/engine/assessment/report.py && echo '  ✓ report.py' || echo '  ✗ report.py missing'
    test -f /workspace/scripts/download_salad.py && echo '  ✓ download_salad.py' || echo '  ✗ download_salad.py missing'
    test -f /workspace/scripts/eval_utility.py && echo '  ✓ eval_utility.py' || echo '  ✗ eval_utility.py missing'
    test -f /workspace/configs/runtime/salad.yaml && echo '  ✓ salad.yaml' || echo '  ✗ salad.yaml missing'
"

