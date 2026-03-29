#!/usr/bin/env python3
"""
启动后端服务器
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting NeuroBreak API server on {host}:{port}")
    print("Press Ctrl+C to stop")
    
    uvicorn.run(
        "engine.server:app",
        host=host,
        port=port,
        reload=True,  # 开发模式，自动重载
    )

