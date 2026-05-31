"""
LLM 多厂商客户端统一接口

支持中国主流大模型厂商，用于 LLM 安全评测
"""

from llm_client.core.base import LLMClient, LLMResponse, LLMRequest
from llm_client.core.factory import LLMClientFactory, create_client
from llm_client.api import LLMAPIService, get_api_service, generate_with_config

__all__ = [
    # 核心类
    "LLMClient",
    "LLMResponse",
    "LLMRequest",
    # 工厂
    "LLMClientFactory",
    "create_client",
    # API服务
    "LLMAPIService",
    "get_api_service",
    "generate_with_config",
]
