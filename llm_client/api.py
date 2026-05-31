"""
LLM 客户端统一接口模块
提供 REST API 风格的使用方式
"""

from typing import Any, Dict, Optional
import logging

from llm_client.core.base import LLMClient, LLMResponse
from llm_client.core.factory import LLMClientFactory, create_client

logger = logging.getLogger(__name__)


class LLMAPIService:
    """
    LLM API 服务类

    提供统一的 REST API 风格接口
    支持字典格式的输入输出
    """

    def __init__(self):
        self._client: Optional[LLMClient] = None

    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        初始化客户端

        Args:
            config: 配置字典，包含 provider, api_key, model

        Returns:
            bool: 初始化是否成功
        """
        try:
            provider = config.get("provider")
            api_key = config.get("api_key")
            model = config.get("model")

            if not all([provider, api_key, model]):
                logger.error("配置缺少必要字段: provider, api_key, model")
                return False

            self._client = create_client(
                provider=provider,
                api_key=api_key,
                model_name=model,
                base_url=config.get("base_url"),
                timeout=config.get("timeout", 60)
            )
            logger.info(f"已初始化 {provider} 客户端: {model}")
            return True
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return False

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        生成文本响应

        Args:
            prompt: 输入提示词
            **kwargs: 其他参数（temperature, max_tokens 等）

        Returns:
            Dict[str, Any]: 统一格式的响应字典
        """
        if not self._client:
            return {
                "response": "",
                "success": False,
                "error": "客户端未初始化，请先调用 initialize()"
            }

        response = self._client.generate(prompt, **kwargs)
        return response.to_dict()

    def batch_generate(
        self,
        prompts: list,
        **kwargs
    ) -> list:
        """
        批量生成

        Args:
            prompts: 提示词列表
            **kwargs: 其他参数

        Returns:
            list: 响应字典列表
        """
        return [self.generate(p, **kwargs) for p in prompts]


# 全局服务实例
_api_service: Optional[LLMAPIService] = None


def get_api_service() -> LLMAPIService:
    """
    获取全局 API 服务实例

    Returns:
        LLMAPIService: API服务实例
    """
    global _api_service
    if _api_service is None:
        _api_service = LLMAPIService()
    return _api_service


def generate_with_config(config: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    """
    根据配置生成响应（便捷函数）

    Args:
        config: 配置字典，包含 provider, api_key, model
        prompt: 提示词

    Returns:
        Dict[str, Any]: 响应字典
    """
    service = LLMAPIService()
    if not service.initialize(config):
        return {
            "response": "",
            "success": False,
            "error": "初始化失败"
        }
    return service.generate(prompt)
