"""
LLM 客户端工厂模块
使用工厂模式创建不同厂商的客户端实例
"""

from typing import Optional
import logging

from llm_client.core.base import LLMClient
from llm_client.providers import (
    OpenAIClient,
    QwenClient,
    ERNIEClient,
    HunyuanClient,
    DeepSeekClient,
    MoonshotClient,
)

logger = logging.getLogger(__name__)


class LLMClientFactory:
    """
    LLM客户端工厂类

    提供统一的客户端创建接口，支持所有主流大模型厂商
    """

    _REGISTRY: dict = {
        "openai": OpenAIClient,
        "qwen": QwenClient,
        "ernie": ERNIEClient,
        "hunyuan": HunyuanClient,
        "deepseek": DeepSeekClient,
        "moonshot": MoonshotClient,
        # 别名支持
        "wenxin": ERNIEClient,
        "baidu": ERNIEClient,
        "ali": QwenClient,
        "alibaba": QwenClient,
        "tencent": HunyuanClient,
        "kimi": MoonshotClient,
    }

    @classmethod
    def register(cls, name: str, client_class: type) -> None:
        """
        注册新的客户端类型

        Args:
            name: 厂商标识符
            client_class: 客户端类
        """
        cls._REGISTRY[name.lower()] = client_class
        logger.info(f"已注册新厂商: {name} -> {client_class.__name__}")

    @classmethod
    def create(
        cls,
        provider: str,
        api_key: str,
        model_name: str,
        base_url: Optional[str] = None,
        timeout: int = 60
    ) -> LLMClient:
        """
        创建LLM客户端实例

        Args:
            provider: 厂商名称 (openai/qwen/ernie/hunyuan/deepseek/moonshot)
            api_key: API密钥
            model_name: 模型名称
            base_url: API基础地址（可选）
            timeout: 请求超时时间

        Returns:
            LLMClient: 客户端实例

        Raises:
            ValueError: 不支持的厂商

        Example:
            >>> client = LLMClientFactory.create(
            ...     provider="qwen",
            ...     api_key="sk-xxx",
            ...     model_name="qwen-turbo"
            ... )
        """
        provider_lower = provider.lower()
        if provider_lower not in cls._REGISTRY:
            available = ", ".join(cls._REGISTRY.keys())
            raise ValueError(
                f"不支持的厂商: {provider}\n"
                f"支持的厂商: {available}"
            )

        client_class = cls._REGISTRY[provider_lower]
        return client_class(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            timeout=timeout
        )

    @classmethod
    def list_providers(cls) -> list:
        """
        列出所有支持的厂商

        Returns:
            支持的厂商名称列表
        """
        return list(cls._REGISTRY.keys())

    @classmethod
    def get_provider_info(cls, provider: str) -> Optional[str]:
        """
        获取厂商信息

        Args:
            provider: 厂商名称

        Returns:
            厂商描述信息
        """
        info_map = {
            "openai": "OpenAI (GPT系列)",
            "qwen": "阿里云 通义千问",
            "ernie": "百度 文心一言",
            "hunyuan": "腾讯云 混元大模型",
            "deepseek": "DeepSeek",
            "moonshot": "Moonshot (Kimi)",
        }
        return info_map.get(provider.lower())


def create_client(provider: str, api_key: str, model_name: str, **kwargs) -> LLMClient:
    """
    便捷函数：创建LLM客户端

    工厂函数的简写形式

    Args:
        provider: 厂商名称
        api_key: API密钥
        model_name: 模型名称
        **kwargs: 其他参数（base_url, timeout）

    Returns:
        LLMClient: 客户端实例

    Example:
        >>> client = create_client("qwen", "sk-xxx", "qwen-turbo")
        >>> response = client.generate("你好")
    """
    return LLMClientFactory.create(
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        **kwargs
    )
