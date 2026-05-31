"""
测试数据解析系统
用于 LLM 越狱评测

支持：
- JSON 文件加载
- CSV 文件加载（JailBench 格式）
- 随机抽样
- 统一输出格式
- 与 TestRunner 无缝集成
"""

from .data_structure import (
    JailbreakContent,
    MethodInfo,
    TestSample,
    TestDataset
)
from .loader import DatasetLoader
from .csv_loader import CSVLoader
from .test_runner import TestRunner, TestResult, OutputResult, Safeguard
from .evaluator import Evaluator, EvaluationResult, judge_jailbroken, evaluate
from .safeguard import apply_safeguard, detect_jailbreak, SafeguardPrompt
from .result_formatter import ResultFormatter, FormattedResult, format_and_save

__all__ = [
    "JailbreakContent",
    "MethodInfo",
    "TestSample",
    "TestDataset",
    "DatasetLoader",
    "CSVLoader",
    "TestRunner",
    "TestResult",
    "OutputResult",
    "Safeguard",
    "Evaluator",
    "EvaluationResult",
    "judge_jailbroken",
    "evaluate",
    "apply_safeguard",
    "detect_jailbreak",
    "SafeguardPrompt",
    "ResultFormatter",
    "FormattedResult",
    "format_and_save",
]
