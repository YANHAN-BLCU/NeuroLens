"""
LLM客户端统一抽象基类
提供所有厂商客户端的统一接口定义
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM调用统一响应格式"""
    response: str
    success: bool = True
    error: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "response": self.response,
            "success": self.success,
            "error": self.error,
            "provider": self.provider,
            "model": self.model
        }


@dataclass
class LLMRequest:
    """LLM调用请求格式"""
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    timeout: int = 60

    def to_payload(self) -> Dict[str, Any]:
        """转换为各厂商API的请求payload"""
        base_payload = {
            "prompt": self.prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }
        return base_payload


class LLMClient(ABC):
    """
    大语言模型客户端抽象基类

    所有厂商客户端必须继承此类并实现相应方法
    """

    PROVIDER_NAME: str = "base"

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: Optional[str] = None,
        timeout: int = 60
    ):
        """
        初始化LLM客户端

        Args:
            api_key: API密钥
            model_name: 模型名称
            base_url: API基础地址（可选）
            timeout: 请求超时时间（秒）
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self._validate_config()

    def _validate_config(self) -> None:
        """验证配置参数"""
        if not self.api_key:
            raise ValueError("API密钥不能为空")
        if not self.model_name:
            raise ValueError("模型名称不能为空")

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        生成文本响应（统一接口）

        Args:
            prompt: 输入提示词
            **kwargs: 其他可选参数

        Returns:
            LLMResponse: 统一格式的响应对象
        """
        pass

    @abstractmethod
    def _build_headers(self) -> Dict[str, str]:
        """构建请求头"""
        pass

    @abstractmethod
    def _build_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """构建请求体"""
        pass

    @abstractmethod
    def _parse_response(self, raw_response: Dict[str, Any]) -> str:
        """解析厂商API响应"""
        pass

    @abstractmethod
    def get_api_endpoint(self) -> str:
        """获取API端点"""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.model_name}>"
