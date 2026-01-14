"""
下载推理模型和安全分类器脚本（使用 ModelScope）

支持下载：
- 推理模型：
  - LLM-Research/Llama-3.2-3B-Instruct (默认)
  - LLM-Research/Meta-Llama-3-8B-Instruct
- 安全分类器：
  - LLM-Research/Llama-Guard-3-1B (默认)
  - LLM-Research/Llama-Guard-3-8B

使用方法：
    python scripts/download_models.py --model LLM-Research/Llama-3.2-3B-Instruct --classifier LLM-Research/Llama-Guard-3-1B
    python scripts/download_models.py --all  # 下载默认的两个模型
    python scripts/download_models.py --model-8b --classifier-8b  # 下载 8B 模型
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from modelscope import snapshot_download
    from modelscope.hub.api import HubApi
except ImportError:
    print("错误: 需要安装 modelscope")
    print("请运行: pip install modelscope")
    sys.exit(1)

# ModelScope 错误处理
try:
    from modelscope.hub.errors import NotExistError, RequestError
except ImportError:
    # 如果导入失败，使用通用异常
    NotExistError = FileNotFoundError
    RequestError = Exception


# 默认模型配置（ModelScope 格式）
DEFAULT_MODEL = "LLM-Research/Llama-3.2-3B-Instruct"
DEFAULT_CLASSIFIER = "LLM-Research/Llama-Guard-3-1B"

# 8B 模型配置
MODEL_8B = "LLM-Research/Meta-Llama-3-8B-Instruct"
CLASSIFIER_8B = "LLM-Research/Llama-Guard-3-8B"


def get_cache_dir() -> Path:
    """获取模型缓存目录"""
    # ModelScope 默认缓存目录
    cache_dir = os.getenv("MODELSCOPE_CACHE") or os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
    if cache_dir:
        return Path(cache_dir) / "models"
    # 默认使用用户目录下的缓存（ModelScope 兼容 HuggingFace 缓存路径）
    return Path.home() / ".cache" / "modelscope" / "hub"


def download_model(
    model_id: str,
    cache_dir: Path,
    token: Optional[str] = None,
    max_retries: int = 3,
    max_workers: int = 8,
) -> Path:
    """
    下载单个模型（使用 ModelScope）
    
    Args:
        model_id: ModelScope 模型 ID
        cache_dir: 缓存目录
        token: ModelScope token（可选，优先使用环境变量）
        max_retries: 最大重试次数
        max_workers: 最大并发下载数（ModelScope 暂不支持，保留参数）
    
    Returns:
        模型本地路径
    """
    print(f"\n{'='*60}")
    print(f"开始下载模型: {model_id}")
    print(f"{'='*60}")
    
    # 获取 token
    if not token:
        token = os.getenv("MODELSCOPE_TOKEN") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("HF_TOKEN")
    
    # 如果 token 存在，设置环境变量
    if token:
        os.environ["MODELSCOPE_TOKEN"] = token
        print(f"✓ 已设置 ModelScope token")
    
    # ModelScope 会自动处理缓存目录，但我们也可以指定本地目录
    # 构建本地目录路径
    local_dir = cache_dir / model_id.replace("/", "_")
    
    # 检查是否已存在
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"⚠ 模型目录已存在: {local_dir}")
        response = input("是否重新下载？(y/N): ").strip().lower()
        if response != 'y':
            print(f"✓ 跳过下载，使用现有模型: {local_dir}")
            return local_dir
    
    # 创建缓存目录
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载模型
    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"下载中... (尝试 {retry_count + 1}/{max_retries})")
            # ModelScope 的 snapshot_download 会自动处理缓存
            # 如果指定了 local_dir，会下载到该目录；否则使用 cache_dir
            model_path = snapshot_download(
                model_id=model_id,
                cache_dir=str(cache_dir),
                local_dir=str(local_dir) if local_dir != cache_dir / model_id.replace("/", "_") else None,
                revision="master",
            )
            # ModelScope 返回的路径可能是缓存路径，优先使用 local_dir
            if local_dir.exists() and any(local_dir.iterdir()):
                final_path = local_dir
            else:
                final_path = Path(model_path)
            print(f"✓ 模型下载完成: {final_path}")
            return final_path
            
        except Exception as e:
            error_str = str(e).lower()
            
            # 检查是否是模型不存在错误
            if "not found" in error_str or "not exist" in error_str or "404" in error_str:
                print(f"✗ 模型不存在: {model_id}")
                print(f"  提示: 请检查 ModelScope 上是否存在该模型")
                print(f"  访问 https://modelscope.cn/models 搜索模型")
                sys.exit(1)
            
            # 检查是否是认证错误
            if "401" in error_str or "unauthorized" in error_str or "token" in error_str or "permission" in error_str:
                print(f"✗ 认证失败: 请检查 ModelScope token")
                print(f"  提示: 在 ModelScope 官网获取 token")
                print(f"  设置环境变量: export MODELSCOPE_TOKEN=your_token")
                sys.exit(1)
            
            # 其他错误，重试
            retry_count += 1
            if retry_count < max_retries:
                print(f"⚠ 下载失败，将重试: {e}")
            else:
                print(f"✗ 下载失败，已达最大重试次数: {e}")
                print(f"  提示: 可能需要设置 ModelScope token")
                print(f"  设置环境变量: export MODELSCOPE_TOKEN=your_token")
                sys.exit(1)
    
    return local_dir


def verify_model(model_path: Path) -> bool:
    """
    验证模型完整性
    
    Args:
        model_path: 模型本地路径
    
    Returns:
        是否验证通过
    """
    if not model_path.exists():
        print(f"✗ 模型路径不存在: {model_path}")
        return False
    
    # 检查关键文件
    required_files = ["config.json"]
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠ 缺少关键文件: {', '.join(missing_files)}")
        return False
    
    # 统计文件大小
    total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    total_size_gb = total_size / (1024 ** 3)
    file_count = len(list(model_path.rglob("*")))
    
    print(f"✓ 模型验证通过")
    print(f"  路径: {model_path}")
    print(f"  文件数: {file_count}")
    print(f"  总大小: {total_size_gb:.2f} GB")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="下载推理模型和安全分类器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  # 下载默认的两个模型（3B）
  python {sys.argv[0]} --all
  
  # 下载 8B 的两个模型
  python {sys.argv[0]} --all-8b
  
  # 单独下载推理模型
  python {sys.argv[0]} --model {DEFAULT_MODEL}
  
  # 下载 8B 推理模型
  python {sys.argv[0]} --model-8b
  
  # 单独下载分类器
  python {sys.argv[0]} --classifier {DEFAULT_CLASSIFIER}
  
  # 下载 8B 分类器
  python {sys.argv[0]} --classifier-8b
  
  # 自定义模型和输出目录
  python {sys.argv[0]} --model {DEFAULT_MODEL} --classifier {DEFAULT_CLASSIFIER} --output /path/to/cache
  
  # 混合使用：3B 推理模型 + 8B 分类器
  python {sys.argv[0]} --model {DEFAULT_MODEL} --classifier-8b
  
  # 使用并发下载和验证
  python {sys.argv[0]} --all --max-workers 8 --verify
        """,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help=f"推理模型 ID (默认: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        help=f"安全分类器 ID (默认: {DEFAULT_CLASSIFIER})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="下载默认的两个模型（推理模型 + 安全分类器）",
    )
    parser.add_argument(
        "--model-8b",
        action="store_true",
        help=f"使用 8B 推理模型: {MODEL_8B}",
    )
    parser.add_argument(
        "--classifier-8b",
        action="store_true",
        help=f"使用 8B 安全分类器: {CLASSIFIER_8B}",
    )
    parser.add_argument(
        "--all-8b",
        action="store_true",
        help="下载 8B 的两个模型（8B 推理模型 + 8B 安全分类器）",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="模型缓存目录（默认: $MODELSCOPE_CACHE/models 或 ~/.cache/modelscope/hub）",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="ModelScope token（也可通过环境变量 MODELSCOPE_TOKEN 设置）",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="最大重试次数（默认: 3）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="最大并发下载数（默认: 8）",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="下载后验证模型完整性",
    )
    
    args = parser.parse_args()
    
    # 确定要下载的模型
    models_to_download = []
    
    # 处理 --all 和 --all-8b 选项
    if args.all:
        models_to_download = [DEFAULT_MODEL, DEFAULT_CLASSIFIER]
    elif args.all_8b:
        models_to_download = [MODEL_8B, CLASSIFIER_8B]
    else:
        # 处理单个模型选项
        if args.model_8b:
            models_to_download.append(MODEL_8B)
        elif args.model:
            models_to_download.append(args.model)
        
        if args.classifier_8b:
            models_to_download.append(CLASSIFIER_8B)
        elif args.classifier:
            models_to_download.append(args.classifier)
    
    if not models_to_download:
        print("错误: 请指定要下载的模型")
        print("使用 --all 下载默认模型，或使用 --model/--classifier 指定模型")
        parser.print_help()
        sys.exit(1)
    
    # 确定缓存目录
    if args.output:
        cache_dir = Path(args.output)
    else:
        cache_dir = get_cache_dir()
    
    print(f"模型缓存目录: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载模型
    downloaded_paths = []
    for model_id in models_to_download:
        try:
            model_path = download_model(
                model_id=model_id,
                cache_dir=cache_dir,
                token=args.token,
                max_retries=args.max_retries,
                max_workers=args.max_workers,
            )
            downloaded_paths.append((model_id, model_path))
            
            # 验证模型
            if args.verify:
                verify_model(model_path)
                
        except KeyboardInterrupt:
            print("\n\n用户中断下载")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ 下载 {model_id} 时出错: {e}")
            sys.exit(1)
    
    # 总结
    print(f"\n{'='*60}")
    print("下载完成总结")
    print(f"{'='*60}")
    for model_id, model_path in downloaded_paths:
        print(f"  {model_id}")
        print(f"    → {model_path}")
    print(f"\n所有模型已保存到: {cache_dir}")


if __name__ == "__main__":
    main()

