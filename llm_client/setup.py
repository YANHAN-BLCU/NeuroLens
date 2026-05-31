"""
LLM 多厂商客户端统一接口
支持中国主流大模型厂商，用于 LLM 安全评测
"""

from setuptools import setup, find_packages

setup(
    name="llm_client",
    version="1.0.0",
    description="多厂商大语言模型API调用系统，用于LLM安全评测",
    author="Neurolens",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "requests>=2.28.0",
    ],
)
