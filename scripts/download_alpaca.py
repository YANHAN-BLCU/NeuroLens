#!/usr/bin/env python3
"""
下载 Stanford Alpaca 数据集

使用方法:
    python scripts/download_alpaca.py --output data/alpaca

数据集来源: https://github.com/tatsu-lab/stanford_alpaca
"""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """下载文件并显示进度"""
    print(f'正在从 {url} 下载...')
    
    try:
        req = Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        
        with urlopen(req) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f'\r进度: {percent:.1f}% ({downloaded}/{total_size} bytes)', end='', flush=True)
            
            print()  # 换行
            print(f'下载完成: {output_path}')
            return True
            
    except HTTPError as e:
        print(f'HTTP 错误 {e.code}: {e.reason}')
        return False
    except URLError as e:
        print(f'URL 错误: {e.reason}')
        return False
    except Exception as e:
        print(f'下载失败: {e}')
        return False


def convert_json_to_jsonl(json_path: Path, jsonl_path: Path):
    """将 JSON 格式转换为 JSONL 格式，适配代码库的数据格式要求"""
    print(f'正在转换格式: {json_path} -> {jsonl_path}')
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f'错误: JSON 文件应包含一个数组，但得到 {type(data)}')
            return False
        
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        
        converted_count = 0
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in data:
                # Stanford Alpaca 格式: {"instruction": "...", "input": "...", "output": "..."}
                # 转换为代码库需要的格式: {"input": {"prompt": "..."}}
                
                # 构建 prompt
                instruction = item.get('instruction', '')
                input_text = item.get('input', '')
                
                if input_text:
                    prompt = f"{instruction}\n\n{input_text}"
                else:
                    prompt = instruction
                
                # 创建符合代码库格式的 JSON 对象
                output_item = {
                    "input": {
                        "prompt": prompt
                    }
                }
                
                # 可选：保留原始字段
                if 'output' in item:
                    output_item['output'] = item['output']
                
                f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                converted_count += 1
        
        print(f'转换完成: {converted_count} 个样本')
        return True
        
    except json.JSONDecodeError as e:
        print(f'JSON 解析错误: {e}')
        return False
    except Exception as e:
        print(f'转换失败: {e}')
        return False


def main():
    parser = argparse.ArgumentParser(
        description='下载 Stanford Alpaca 数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载到默认位置 data/alpaca/
  python scripts/download_alpaca.py

  # 指定输出目录
  python scripts/download_alpaca.py --output /path/to/alpaca

  # 只下载，不转换格式
  python scripts/download_alpaca.py --no-convert

数据集来源: https://github.com/tatsu-lab/stanford_alpaca
        """
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/alpaca',
        help='输出目录（默认: data/alpaca）'
    )
    
    parser.add_argument(
        '--url',
        type=str,
        default='https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json',
        help='数据集下载 URL（默认: Stanford Alpaca 官方 URL）'
    )
    
    parser.add_argument(
        '--no-convert',
        action='store_true',
        help='只下载 JSON 文件，不转换为 JSONL 格式'
    )
    
    parser.add_argument(
        '--keep-json',
        action='store_true',
        help='保留原始 JSON 文件（默认会删除）'
    )
    
    parser.add_argument(
        '--yes',
        action='store_true',
        help='自动确认所有提示（适用于 Docker/CI 环境）'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    json_path = output_dir / 'alpaca_data.json'
    jsonl_path = output_dir / 'alpaca_data.jsonl'
    
    # 检查文件是否已存在
    if json_path.exists():
        print(f'文件已存在: {json_path}')
        print('跳过下载步骤')
    else:
        # 下载数据集
        if not download_file(args.url, json_path):
            print('下载失败，请检查网络连接或 URL')
            sys.exit(1)
    
    # 转换为 JSONL 格式（如果需要）
    if not args.no_convert:
        if jsonl_path.exists():
            print(f'JSONL 文件已存在: {jsonl_path}')
            if args.yes:
                # 非交互模式：自动重新转换
                print('自动重新转换（--yes 模式）')
                if not convert_json_to_jsonl(json_path, jsonl_path):
                    sys.exit(1)
            else:
                # 交互模式：询问用户
                try:
                    response = input('是否重新转换? (y/N): ').strip().lower()
                    if response != 'y':
                        print('跳过转换步骤')
                    else:
                        if not convert_json_to_jsonl(json_path, jsonl_path):
                            sys.exit(1)
                except (EOFError, KeyboardInterrupt):
                    # Docker 非交互式环境或 Ctrl+C
                    print('\n跳过转换步骤（非交互式环境）')
        else:
            if not convert_json_to_jsonl(json_path, jsonl_path):
                sys.exit(1)
        
        # 删除原始 JSON 文件（如果不需要保留）
        if not args.keep_json and json_path.exists():
            print(f'删除原始 JSON 文件: {json_path}')
            json_path.unlink()
    
    # 显示统计信息
    if jsonl_path.exists():
        print(f'\n数据集信息:')
        print(f'  文件路径: {jsonl_path}')
        
        # 统计样本数量
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            sample_count = sum(1 for _ in f)
        print(f'  样本数量: {sample_count:,}')
        
        # 文件大小
        file_size = jsonl_path.stat().st_size
        print(f'  文件大小: {file_size / 1024 / 1024:.2f} MB')
    
    print('\n下载完成！')
    print(f'\n使用示例:')
    print(f'  python scripts/run_snip_scorer.py \\')
    print(f'      --model-path /path/to/model \\')
    print(f'      --dataset-path {jsonl_path} \\')
    print(f'      --output-path /path/to/output \\')
    print(f'      --mode utility \\')
    print(f'      --batch-size 8 \\')
    print(f'      --num-samples 0')


if __name__ == '__main__':
    main()
