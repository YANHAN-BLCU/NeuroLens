# scripts/check_cuda.py
import torch
import sys

print("=" * 60)
print("CUDA 环境诊断")
print("=" * 60)

# 检查 PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")

# 检查 CUDA 是否可用
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    显存: {props.total_memory / 1024**3:.2f} GB")
        print(f"    计算能力: {props.major}.{props.minor}")
else:
    print("\n❌ CUDA 不可用！")
    print("\n可能的原因：")
    print("1. 未安装 NVIDIA GPU 驱动")
    print("2. PyTorch 未安装 CUDA 版本")
    print("3. CUDA 版本不匹配")
    print("4. Docker 容器未正确配置 GPU 支持")
    print("5. 在 CPU-only 环境中运行")

# 检查环境变量
print("\n环境变量检查：")
import os
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "未设置")
print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

print("=" * 60)