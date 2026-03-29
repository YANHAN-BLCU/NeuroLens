"""
从 SALAD 数据集中提取安全样本用于安全神经元识别

支持从以下数据集中提取安全部分：
1. defense_enhanced_set_train.jsonl - 防御增强的样本（daugq 字段）
2. mcq_set_train.jsonl - 多选题中的安全答案（gt == "A" 的样本）
3. base_evaluation.jsonl - 评估日志中安全响应的样本（guard.verdict == "allow"）
"""

import json
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from torch.utils.data import Dataset


class SaladSafetyDataset(Dataset):
    """
    从 SALAD 数据集中提取安全样本
    
    支持多种数据源：
    - defense_enhanced_set_train.jsonl: 使用防御增强的问题（daugq）
    - mcq_set_train.jsonl: 使用安全答案（gt == "A"）
    - base_evaluation.jsonl: 使用安全响应的样本（guard.verdict == "allow"）
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        source_type: str = "auto",  # "auto", "defense", "mcq", "evaluation"
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            file_path: 数据集文件路径
            source_type: 数据源类型
                - "auto": 自动检测（根据文件名）
                - "defense": 防御增强数据集（使用 daugq 字段）
                - "mcq": 多选题数据集（使用 gt == "A" 的样本）
                - "evaluation": 评估日志（使用 guard.verdict == "allow" 的样本）
            max_samples: 最大样本数（None 表示全部）
        """
        self.file_path = Path(file_path)
        self.source_type = source_type
        self.samples: List[Dict[str, Any]] = []
        
        # 自动检测数据源类型
        if source_type == "auto":
            filename = self.file_path.name.lower()
            if "defense" in filename:
                source_type = "defense"
            elif "mcq" in filename:
                source_type = "mcq"
            elif "evaluation" in filename:
                source_type = "evaluation"
            else:
                # 默认尝试所有类型
                source_type = "auto"
        
        self.source_type = source_type
        self._load_samples(max_samples)
    
    def _load_samples(self, max_samples: Optional[int]):
        """加载样本"""
        max_n = None if (max_samples is None or max_samples <= 0) else int(max_samples)
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if max_n is not None and len(self.samples) >= max_n:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[SaladSafetyDataset] 警告: 第 {line_num} 行 JSON 解析失败: {e}")
                    continue
                
                sample = self._extract_safe_sample(obj, line_num)
                if sample is not None:
                    self.samples.append(sample)
        
        print(
            f"[SaladSafetyDataset] 从 {self.file_path.name} 加载了 {len(self.samples)} 个安全样本"
            f" (类型: {self.source_type})"
        )
    
    def _extract_safe_sample(self, obj: Dict[str, Any], line_num: int) -> Optional[Dict[str, Any]]:
        """从 JSON 对象中提取安全样本"""
        
        if self.source_type == "defense" or (
            self.source_type == "auto" and "daugq" in obj
        ):
            # 防御增强数据集：使用 daugq（防御增强的问题）
            daugq = obj.get("daugq")
            if daugq and isinstance(daugq, str) and daugq.strip():
                # 返回防御增强的问题作为输入
                return {"text": daugq.strip()}
        
        elif self.source_type == "mcq" or (
            self.source_type == "auto" and "gt" in obj and "baseq" in obj
        ):
            # 多选题数据集：只使用安全答案（gt == "A"）
            gt = obj.get("gt")
            baseq = obj.get("baseq")
            
            if gt == "A" and baseq and isinstance(baseq, str) and baseq.strip():
                # 返回基础问题作为输入
                return {"text": baseq.strip()}
        
        elif self.source_type == "evaluation" or (
            self.source_type == "auto"
            and "guard" in obj
            and "input" in obj
            and "inference" in obj
        ):
            # 评估日志：使用安全响应的样本（guard.verdict == "allow"）
            guard = obj.get("guard", {})
            verdict = guard.get("verdict")
            
            if verdict == "allow":
                input_data = obj.get("input", {})
                inference = obj.get("inference", {})
                
                # 提取 prompt 和 output
                if isinstance(input_data, dict):
                    prompt = input_data.get("prompt")
                else:
                    prompt = input_data
                
                output = inference.get("output")
                
                if prompt and isinstance(prompt, str) and prompt.strip():
                    if output and isinstance(output, str) and output.strip():
                        # 有输出：返回 prompt + output 格式
                        return {
                            "input": {"prompt": prompt.strip()},
                            "output": output.strip(),
                        }
                    else:
                        # 只有 prompt：返回 text 格式
                        return {"text": prompt.strip()}
        
        # 如果自动检测失败，尝试通用格式
        if self.source_type == "auto":
            # 尝试提取 text 字段
            if "text" in obj:
                text = obj.get("text")
                if text and isinstance(text, str) and text.strip():
                    return {"text": text.strip()}
            
            # 尝试提取 prompt 字段
            if "prompt" in obj:
                prompt = obj.get("prompt")
                if prompt and isinstance(prompt, str) and prompt.strip():
                    return {"text": prompt.strip()}
        
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class CombinedSaladSafetyDataset(Dataset):
    """
    组合多个 SALAD 数据集的安全样本
    """
    
    def __init__(
        self,
        file_paths: List[Union[str, Path]],
        source_types: Optional[List[str]] = None,
        max_samples_per_file: Optional[int] = None,
        max_total_samples: Optional[int] = None,
    ):
        """
        Args:
            file_paths: 数据集文件路径列表
            source_types: 对应的数据源类型列表（None 表示自动检测）
            max_samples_per_file: 每个文件的最大样本数
            max_total_samples: 总最大样本数
        """
        self.samples: List[Dict[str, Any]] = []
        
        if source_types is None:
            source_types = ["auto"] * len(file_paths)
        
        for file_path, source_type in zip(file_paths, source_types):
            dataset = SaladSafetyDataset(
                file_path=file_path,
                source_type=source_type,
                max_samples=max_samples_per_file,
            )

            # 统一样本格式，避免 DataLoader 在默认 collate 时因 key 不一致报错
            # 目标：所有样本都转成 {"text": "..."} 这一种形式，便于 SNIP 打分
            normalized_samples: List[Dict[str, Any]] = []
            for s in dataset.samples:
                # 已经是 text 的，直接保留
                if isinstance(s, dict) and "text" in s:
                    text = s.get("text")
                    if isinstance(text, str) and text.strip():
                        normalized_samples.append({"text": text.strip()})
                    continue

                # prompt/response 形式 -> 合并成单段 text
                if isinstance(s, dict) and "prompt" in s and "response" in s:
                    prompt = s.get("prompt") or ""
                    response = s.get("response") or ""
                    parts = []
                    if isinstance(prompt, str) and prompt.strip():
                        parts.append(prompt.strip())
                    if isinstance(response, str) and response.strip():
                        parts.append(response.strip())
                    if parts:
                        normalized_samples.append({"text": "\n".join(parts)})
                    continue

                # Alpaca / evaluation 形式：{"input": {...}, "output": "..."}
                if isinstance(s, dict) and "input" in s and "output" in s:
                    inp = s.get("input")
                    if isinstance(inp, dict):
                        prompt = inp.get("prompt") or ""
                    else:
                        prompt = inp or ""
                    output = s.get("output") or ""
                    parts = []
                    if isinstance(prompt, str) and prompt.strip():
                        parts.append(prompt.strip())
                    if isinstance(output, str) and output.strip():
                        parts.append(output.strip())
                    if parts:
                        normalized_samples.append({"text": "\n".join(parts)})
                    continue

                # 兜底：把整个样本转成字符串
                try:
                    text = str(s).strip()
                except Exception:
                    text = ""
                if text:
                    normalized_samples.append({"text": text})

            dataset.samples = normalized_samples

            remaining = (
                max_total_samples - len(self.samples)
                if max_total_samples is not None
                else None
            )
            
            if remaining is not None and remaining <= 0:
                break
            
            if remaining is not None:
                dataset.samples = dataset.samples[:remaining]
            
            self.samples.extend(dataset.samples)
        
        print(
            f"[CombinedSaladSafetyDataset] 总共加载了 {len(self.samples)} 个安全样本"
            f" (来自 {len(file_paths)} 个文件)"
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
