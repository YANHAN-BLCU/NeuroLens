#!/usr/bin/env python3
"""
快速检查模型路径是否正确挂载
"""

import os
from pathlib import Path

# 模型路径配置
LLM_LOCAL_PATH = os.getenv("LLM_LOCAL_PATH", "F:/models/meta-llama_Llama-3.2-3B-Instruct")
GUARD_LOCAL_PATH = os.getenv("GUARD_LOCAL_PATH", "F:/models/meta-llama_Llama-Guard-3-1B")
# 容器内路径：优先检查 /cache（当前挂载点），其次 /workspace/models
LLM_CONTAINER_PATH = os.getenv("LLM_CONTAINER_PATH", "/cache/meta-llama_Llama-3.2-3B-Instruct")
GUARD_CONTAINER_PATH = os.getenv("GUARD_CONTAINER_PATH", "/cache/meta-llama_Llama-Guard-3-1B")
# 备用路径
LLM_WORKSPACE_PATH = "/workspace/models/meta-llama_Llama-3.2-3B-Instruct"
GUARD_WORKSPACE_PATH = "/workspace/models/meta-llama_Llama-Guard-3-1B"

def check_model_path(path: str, name: str):
    """检查模型路径"""
    print(f"\n{'='*60}")
    print(f"检查 {name}")
    print(f"{'='*60}")
    print(f"路径: {path}")
    
    path_obj = Path(path)
    
    if path_obj.exists():
        print(f"✓ 路径存在")
        
        # 检查是否是目录
        if path_obj.is_dir():
            print(f"✓ 是目录")
            
            # 检查关键文件
            config_file = path_obj / "config.json"
            if config_file.exists():
                print(f"✓ config.json 存在")
            else:
                print(f"✗ config.json 不存在")
            
            # 列出目录内容
            items = list(path_obj.iterdir())
            print(f"  目录包含 {len(items)} 个项目")
            print(f"  前5项: {[item.name for item in items[:5]]}")
            
            # 检查模型文件
            safetensors = list(path_obj.glob("*.safetensors"))
            if safetensors:
                total_size = sum(f.stat().st_size for f in safetensors)
                total_size_gb = total_size / (1024 ** 3)
                print(f"✓ 找到 {len(safetensors)} 个模型文件，总大小: {total_size_gb:.2f} GB")
            else:
                print(f"⚠ 未找到 .safetensors 文件")
        else:
            print(f"✗ 不是目录")
    else:
        print(f"✗ 路径不存在")
        
        # 检查父目录
        parent = path_obj.parent
        if parent.exists():
            print(f"  父目录存在: {parent}")
            print(f"  父目录内容: {[item.name for item in parent.iterdir()][:10]}")
        else:
            print(f"  父目录也不存在: {parent}")

def main():
    print("模型路径检查工具")
    print("="*60)
    
    # 检查容器内路径（主要路径）
    check_model_path(LLM_CONTAINER_PATH, "推理模型（容器内 - /cache）")
    check_model_path(GUARD_CONTAINER_PATH, "安全分类器（容器内 - /cache）")
    
    # 检查备用路径
    check_model_path(LLM_WORKSPACE_PATH, "推理模型（容器内 - /workspace/models）")
    check_model_path(GUARD_WORKSPACE_PATH, "安全分类器（容器内 - /workspace/models）")
    
    # 检查本地路径（如果不在容器内）
    if not Path("/.dockerenv").exists() and not Path("/workspace").exists():
        check_model_path(LLM_LOCAL_PATH, "推理模型（本地）")
        check_model_path(GUARD_LOCAL_PATH, "安全分类器（本地）")
    
    # 检查环境变量
    print(f"\n{'='*60}")
    print("环境变量")
    print(f"{'='*60}")
    print(f"LLM_LOCAL_PATH: {os.getenv('LLM_LOCAL_PATH', '未设置')}")
    print(f"GUARD_LOCAL_PATH: {os.getenv('GUARD_LOCAL_PATH', '未设置')}")
    print(f"LLM_CONTAINER_PATH: {os.getenv('LLM_CONTAINER_PATH', '未设置')}")
    print(f"GUARD_CONTAINER_PATH: {os.getenv('GUARD_CONTAINER_PATH', '未设置')}")
    
    # 检查挂载点
    print(f"\n{'='*60}")
    print("挂载点检查")
    print(f"{'='*60}")
    
    # 检查 /cache/models
    cache_models = Path("/cache/models")
    if cache_models.exists():
        print(f"✓ /cache/models 存在")
        items = list(cache_models.iterdir())
        if items:
            print(f"  内容: {[item.name for item in items][:10]}")
        else:
            print(f"  目录为空")
    else:
        print(f"✗ /cache/models 不存在")
    
    # 检查 /workspace/models
    workspace_models = Path("/workspace/models")
    if workspace_models.exists():
        print(f"✓ /workspace/models 存在")
        items = list(workspace_models.iterdir())
        if items:
            print(f"  内容: {[item.name for item in items][:10]}")
        else:
            print(f"  目录为空")
    else:
        print(f"✗ /workspace/models 不存在")
        print(f"  提示: 如果模型在 F 盘，请确保启动容器时使用 -v F:/models:/workspace/models")

if __name__ == "__main__":
    main()

