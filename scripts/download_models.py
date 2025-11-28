"""
下载推理模型和安全分类器脚本

支持下载：
- 推理模型：meta-llama/Llama-3.2-3B-Instruct
- 安全分类器：meta-llama/Llama-Guard-3-1B

使用方法：
    python scripts/download_models.py --model meta-llama/Llama-3.2-3B-Instruct --classifier meta-llama/Llama-Guard-3-1B
    python scripts/download_models.py --all  # 下载默认的两个模型
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download, login
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("错误: 需要安装 huggingface_hub")
    print("请运行: pip install huggingface_hub")
    sys.exit(1)


# 默认模型配置
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_CLASSIFIER = "meta-llama/Llama-Guard-3-1B"
DEFAULT_MIRROR_ENDPOINT = "https://hf-mirror.com"


def get_cache_dir() -> Path:
    """获取模型缓存目录"""
    # 优先使用环境变量指定的本地模型路径（F盘）
    local_model_path = os.getenv("LOCAL_MODEL_PATH")
    if local_model_path:
        return Path(local_model_path)
    
    cache_dir = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
    if cache_dir:
        return Path(cache_dir) / "models"
    # 默认使用用户目录下的缓存
    return Path.home() / ".cache" / "huggingface" / "models"


def configure_endpoint(use_official: bool = False, mirror: Optional[str] = None) -> str:
    """
    配置 HuggingFace 下载端点，默认优先选择国内镜像。
    """
    if use_official:
        target = mirror or os.getenv("HF_ENDPOINT") or "https://huggingface.co"
    else:
        target = (
            mirror
            or os.getenv("HF_MIRROR")
            or os.getenv("HF_ENDPOINT")
            or os.getenv("HUGGINGFACE_ENDPOINT")
            or DEFAULT_MIRROR_ENDPOINT
        )

    os.environ["HF_ENDPOINT"] = target
    return target


def download_model(
    model_id: str,
    cache_dir: Path,
    token: Optional[str] = None,
    max_retries: int = 3,
    max_workers: int = 8,
) -> Path:
    """
    下载单个模型
    
    Args:
        model_id: HuggingFace 模型 ID
        cache_dir: 缓存目录
        token: HuggingFace token（可选，优先使用环境变量）
        max_retries: 最大重试次数
        max_workers: 最大并发下载数
    
    Returns:
        模型本地路径
    """
    print(f"\n{'='*60}")
    print(f"开始下载模型: {model_id}")
    print(f"{'='*60}")
    
    # 获取 token
    if not token:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    # 如果 token 存在，尝试登录
    if token:
        try:
            login(token=token, add_to_git_credential=False)
            print(f"✓ 已使用 token 登录 HuggingFace")
        except Exception as e:
            print(f"⚠ 登录失败，将尝试匿名下载: {e}")
    
    # 构建本地目录路径
    local_dir = cache_dir / model_id.replace("/", "_")
    
    # 检查是否已存在
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"✓ 模型目录已存在: {local_dir}")
        print(f"  跳过下载，使用现有模型")
        return local_dir
    
    # 创建缓存目录
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载模型
    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"下载中... (尝试 {retry_count + 1}/{max_retries})")
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                token=token,
                max_workers=max_workers,
            )
            print(f"✓ 模型下载完成: {local_dir}")
            return local_dir
            
        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                print(f"✗ 认证失败: 请检查 HuggingFace token 和模型访问权限")
                print(f"  提示: 在 HuggingFace 官网申请 {model_id} 的访问权限")
                print(f"  然后设置环境变量: export HF_TOKEN=your_token")
                sys.exit(1)
            elif e.response.status_code == 404:
                print(f"✗ 模型不存在: {model_id}")
                sys.exit(1)
            else:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"⚠ 下载失败，将重试: {e}")
                else:
                    print(f"✗ 下载失败，已达最大重试次数: {e}")
                    sys.exit(1)
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"⚠ 下载出错，将重试: {e}")
            else:
                print(f"✗ 下载失败: {e}")
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
  # 下载默认的两个模型
  python {sys.argv[0]} --all
  
  # 单独下载推理模型
  python {sys.argv[0]} --model {DEFAULT_MODEL}
  
  # 单独下载分类器
  python {sys.argv[0]} --classifier {DEFAULT_CLASSIFIER}
  
  # 自定义模型和输出目录
  python {sys.argv[0]} --model {DEFAULT_MODEL} --classifier {DEFAULT_CLASSIFIER} --output /path/to/cache
  
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
        "--output",
        type=str,
        help="模型缓存目录（默认: $LOCAL_MODEL_PATH 或 $HF_HOME/models 或 ~/.cache/huggingface/models）",
    )
    parser.add_argument(
        "--mirror",
        type=str,
        help=f"自定义镜像端点（默认优先使用 {DEFAULT_MIRROR_ENDPOINT}）",
    )
    parser.add_argument(
        "--use-official",
        action="store_true",
        help="强制使用官方 huggingface.co 下载（不会自动切换镜像）",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace token（也可通过环境变量 HF_TOKEN 设置）",
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

    endpoint = configure_endpoint(use_official=args.use_official, mirror=args.mirror)
    print(f"HuggingFace 端点: {endpoint}")
    
    # 确定要下载的模型
    models_to_download = []
    if args.all:
        models_to_download = [DEFAULT_MODEL, DEFAULT_CLASSIFIER]
    else:
        if args.model:
            models_to_download.append(args.model)
        if args.classifier:
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


