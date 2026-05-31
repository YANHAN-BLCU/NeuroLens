"""
百度 文心一言 客户端实现
支持 ERNIE 系列模型
"""

from typing import Any, Dict, Optional
import requests

from llm_client.core.base import LLMClient, LLMResponse


class ERNIEClient(LLMClient):
    """
    百度 文心一言 客户端

    支持 ERNIE-Tiny、ERNIE-3.5、ERNIE-4.0 等模型
    API文档: https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html
    """

    PROVIDER_NAME = "ernie"
    DEFAULT_BASE_URL = "https://qianfan.baidubce.com/v2/chat/completions"

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: Optional[str] = None,
        timeout: int = 60
    ):
        super().__init__(api_key, model_name, base_url, timeout)
        self._base_url = base_url or self.DEFAULT_BASE_URL

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _build_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2048)
        top_p = kwargs.get("top_p", 1.0)

        return {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }

    def _parse_response(self, raw_response: Dict[str, Any]) -> str:
        choices = raw_response.get("choices", [])
        if choices and len(choices) > 0:
            return choices[0].get("message", {}).get("content", "")
        return ""

    def get_api_endpoint(self) -> str:
        return self._base_url

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        try:
            endpoint = self.get_api_endpoint()
            headers = self._build_headers()
            payload = self._build_payload(prompt, **kwargs)

            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            raw = response.json()
            result = self._parse_response(raw)

            return LLMResponse(
                response=result,
                success=True,
                provider=self.PROVIDER_NAME,
                model=self.model_name,
                raw_response=raw
            )
        except requests.RequestException as e:
            return LLMResponse(
                response="",
                success=False,
                error=str(e),
                provider=self.PROVIDER_NAME,
                model=self.model_name
            )
