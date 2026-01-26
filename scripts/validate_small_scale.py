#!/usr/bin/env python3
"""
小规模验证监控脚本
监控容器内的小规模训练验证，并生成报告
"""

import subprocess
import time
import sys
from datetime import datetime
from pathlib import Path

def run_docker_cmd(cmd):
    """在容器中执行命令"""
    full_cmd = f"docker exec neurobreak-container {cmd}"
    try:
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[超时]"
    except Exception as e:
        return f"[错误: {str(e)}]"

def print_header(text):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def main():
    print_header("NeuroBreak 小规模验证监控")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 监控循环
    iteration = 0
    start_time = time.time()
    max_wait = 600  # 最多等待 10 分钟
    
    while time.time() - start_time < max_wait:
        iteration += 1
        elapsed = int(time.time() - start_time)
        
        print(f"\n[{elapsed}s] 检查进度 #{iteration}")
        print("-" * 60)
        
        # 获取日志最后部分
        log = run_docker_cmd("tail -20 /workspace/train_probes.log")
        
        # 检查关键进度标记
        if "进度条" in log or "提取隐藏状态" in log:
            print("✓ 隐藏状态提取进度条已启用")
        
        if "训练层探针" in log:
            print("✓ 层级训练进度条已启用")
        
        if "[Model] ✓ 模型已使用4-bit量化" in log:
            print("✓ 模型加载完成（4-bit 量化）")
        
        if "[Layer" in log and "准确率:" in log:
            print("✓ 层级训练进行中")
            # 提取最后的层信息
            for line in log.split('\n'):
                if "[Layer" in line and "准确率:" in line:
                    print(f"  {line.strip()}")
        
        # 检查是否完成
        if "Summary" in log or "已保存到:" in log:
            print("\n✓ 训练完成！")
            print_header("验证完成")
            print(log)
            break
        
        # 查看 GPU 状态
        gpu_info = run_docker_cmd("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits")
        if gpu_info and gpu_info != "[错误" and gpu_info != "[超时]":
            parts = gpu_info.split(',')
            if len(parts) >= 3:
                print(f"GPU: {parts[0].strip()}MB/{parts[1].strip()}MB, 利用率: {parts[2].strip()}%")
        
        # 检查进程
        ps_output = run_docker_cmd("ps aux | grep 'train_linear' | grep -v grep")
        if ps_output and "python" in ps_output:
            print(f"✓ 训练进程运行中")
        else:
            print(f"⚠ 训练进程未找到")
        
        # 等待后重新检查
        if iteration < 20:  # 前 20 次频繁检查
            time.sleep(5)
        else:  # 之后每 10 秒检查一次
            time.sleep(10)
    
    # 最终输出完整日志
    print_header("最终训练日志")
    final_log = run_docker_cmd("cat /workspace/train_probes.log")
    print(final_log)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ 监控已停止（用户中断）")
        sys.exit(0)

