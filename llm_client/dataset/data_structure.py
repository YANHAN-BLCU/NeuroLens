"""
测试数据集数据结构
用于 LLM 越狱评测
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import copy


@dataclass
class JailbreakContent:
    """
    越狱内容结构

    Attributes:
        text: 越狱提示文本
        enhanced_text: 增强版越狱文本（可选）
    """
    text: str
    enhanced_text: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JailbreakContent":
        """从字典创建 JailbreakContent"""
        return cls(
            text=data.get("text", ""),
            enhanced_text=data.get("enhanced_text") or data.get("enhancedText")
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "text": self.text,
            "enhanced_text": self.enhanced_text
        }


@dataclass
class MethodInfo:
    """
    攻击方法信息

    Attributes:
        name: 方法名称（如 role_play, dac, etc.）
        category: 方法类别
        description: 方法描述
        parameters: 方法参数
    """
    name: str
    category: Optional[str] = None
    description: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MethodInfo":
        """从字典创建 MethodInfo"""
        return cls(
            name=data.get("name", "unknown"),
            category=data.get("category"),
            description=data.get("description"),
            parameters=data.get("parameters", {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "parameters": self.parameters
        }


@dataclass
class TestSample:
    """
    测试样本结构

    Attributes:
        index: 样本索引
        base_jailbreak: 基础越狱内容
        method: 攻击方法信息
        enhanced_jailbreak: 增强版越狱内容（可选）
    """
    index: int
    base_jailbreak: JailbreakContent
    method: MethodInfo
    enhanced_jailbreak: Optional[JailbreakContent] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestSample":
        """
        从字典创建 TestSample

        Args:
            data: 原始数据字典

        Returns:
            TestSample 实例
        """
        # 解析 Base Jailbreak
        base_data = data.get("Base Jailbreak", {})
        base_jailbreak = JailbreakContent.from_dict(base_data)

        # 解析 Method
        method_data = data.get("Method", {})
        method = MethodInfo.from_dict(method_data)

        # 解析 Enhanced Jailbreak（可选）
        enhanced_data = data.get("Enhanced Jailbreak", {})
        enhanced_jailbreak = None
        if enhanced_data:
            enhanced_jailbreak = JailbreakContent.from_dict(enhanced_data)

        return cls(
            index=data.get("Index", 0),
            base_jailbreak=base_jailbreak,
            method=method,
            enhanced_jailbreak=enhanced_jailbreak
        )

    def get_prompt(self, use_enhanced: bool = True) -> str:
        """
        获取越狱提示文本

        Args:
            use_enhanced: 是否优先使用增强版（默认 True）

        Returns:
            越狱提示文本
        """
        if use_enhanced and self.enhanced_jailbreak:
            return self.enhanced_jailbreak.enhanced_text or self.enhanced_jailbreak.text
        return self.base_jailbreak.text

    def to_output_dict(self, use_enhanced: bool = True) -> Dict[str, Any]:
        """
        转换为统一输出格式

        Args:
            use_enhanced: 是否优先使用增强版

        Returns:
            统一格式字典
        """
        return {
            "id": self.index,
            "prompt": self.get_prompt(use_enhanced),
            "method": self.method.name
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为完整字典"""
        result = {
            "Index": self.index,
            "Base Jailbreak": self.base_jailbreak.to_dict(),
            "Method": self.method.to_dict()
        }
        if self.enhanced_jailbreak:
            result["Enhanced Jailbreak"] = self.enhanced_jailbreak.to_dict()
        return result


@dataclass
class TestDataset:
    """
    测试数据集结构

    Attributes:
        name: 数据集名称
        description: 数据集描述
        samples: 样本列表
        metadata: 元数据
    """
    name: str = "unnamed"
    description: str = ""
    samples: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TestSample:
        return self.samples[index]

    def __iter__(self):
        return iter(self.samples)

    def filter_by_method(self, method_name: str) -> "TestDataset":
        """按方法名称过滤"""
        filtered = [s for s in self.samples if s.method.name == method_name]
        return TestDataset(
            name=self.name,
            description=self.description,
            samples=filtered,
            metadata=self.metadata
        )

    def get_methods(self) -> list:
        """获取所有方法名称"""
        return list(set(s.method.name for s in self.samples))
