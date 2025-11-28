"""
模型加载和推理模块
提供 Llama 推理模型和 Llama Guard 安全审核模型的加载与使用
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# 默认推理模型切换为 Instruct 版本
LLM_ID = "meta-llama/Llama-3.2-3B-Instruct"
GUARD_ID = "meta-llama/Llama-Guard-3-1B"

# 模型路径配置：优先使用本地路径（/cache/），如果不存在则使用HuggingFace ID
LLM_LOCAL_PATH = os.getenv("LLM_LOCAL_PATH", "F:/models/meta-llama_Llama-3.2-3B-Instruct")
GUARD_LOCAL_PATH = os.getenv("GUARD_LOCAL_PATH", "F:/models/meta-llama_Llama-Guard-3-1B")
# Docker容器内路径（优先检查 /cache，这是当前容器的挂载点）
LLM_CONTAINER_PATH = os.getenv("LLM_CONTAINER_PATH", "/cache/meta-llama_Llama-3.2-3B-Instruct")
GUARD_CONTAINER_PATH = os.getenv("GUARD_CONTAINER_PATH", "/cache/meta-llama_Llama-Guard-3-1B")
# 备用路径
LLM_WORKSPACE_PATH = "/workspace/models/meta-llama_Llama-3.2-3B-Instruct"
GUARD_WORKSPACE_PATH = "/workspace/models/meta-llama_Llama-Guard-3-1B"


def resolve_dtype() -> torch.dtype:
    """选择合适的数据类型以平衡稳定性与显存占用。"""
    if not torch.cuda.is_available():
        return torch.float32

    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def get_model_path(model_id: str, local_path: str, container_path: str, workspace_path: str = "") -> str:
    """
    获取模型路径：优先使用本地路径，如果不存在则使用HuggingFace ID
    
    Args:
        model_id: HuggingFace模型ID
        local_path: Windows本地路径（F盘）
        container_path: Docker容器内路径（主要）
        workspace_path: Docker容器内备用路径
    
    Returns:
        模型路径（本地路径、容器路径或HuggingFace ID）
    """
    container_path_obj = Path(container_path)
    local_path_obj = Path(local_path)
    workspace_path_obj = Path(workspace_path) if workspace_path else None
    
    # 检查主要容器路径
    if container_path_obj.exists():
        config_file = container_path_obj / "config.json"
        if config_file.exists():
            print(f"[ModelManager] 使用容器内路径: {container_path}")
            return str(container_path_obj)
    
    # 检查备用容器路径
    if workspace_path_obj and workspace_path_obj.exists():
        config_file = workspace_path_obj / "config.json"
        if config_file.exists():
            print(f"[ModelManager] 使用备用容器路径: {workspace_path}")
            return str(workspace_path_obj)
    
    # 检查Windows本地路径（在容器内通常不存在，但保留检查）
    if local_path_obj.exists():
        config_file = local_path_obj / "config.json"
        if config_file.exists():
            print(f"[ModelManager] 使用本地路径: {local_path}")
            return str(local_path_obj)
    
    # 检查父目录是否存在（可能模型在子目录中）
    for parent_path in [container_path_obj.parent, workspace_path_obj.parent if workspace_path_obj else None]:
        if parent_path and parent_path.exists():
            # 尝试查找模型目录
            for item in parent_path.iterdir():
                if item.is_dir() and (item / "config.json").exists():
                    print(f"[ModelManager] 在父目录中找到模型: {item}")
                    return str(item)
    
    # 否则使用HuggingFace ID
    print(f"[ModelManager] 本地路径不存在，将使用HuggingFace ID: {model_id}")
    return model_id


class ModelManager:
    """管理推理模型和 Guard 模型的单例类"""

    _instance: Optional[ModelManager] = None
    _llm_tokenizer: Optional[AutoTokenizer] = None
    _llm_model: Optional[AutoModelForCausalLM] = None
    _guard_tokenizer: Optional[AutoTokenizer] = None
    _guard_model: Optional[AutoModelForCausalLM] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_llm(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """加载推理模型（懒加载）"""
        if self._llm_tokenizer is None or self._llm_model is None:
            torch_dtype = resolve_dtype()
            
            # 确定实际使用的模型路径
            model_path = get_model_path(
                LLM_ID,
                LLM_LOCAL_PATH,
                LLM_CONTAINER_PATH,
                LLM_WORKSPACE_PATH
            )
            
            print(f"[ModelManager] Loading LLM: {model_path} (dtype: {torch_dtype})")
            self._llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self._llm_tokenizer.pad_token is None:
                self._llm_tokenizer.pad_token = self._llm_tokenizer.eos_token

            self._llm_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch_dtype,  # 使用dtype代替torch_dtype（修复deprecation警告）
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self._llm_model.eval()
            print("[ModelManager] LLM loaded successfully")
        return self._llm_tokenizer, self._llm_model

    def load_guard(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """加载 Guard 模型（懒加载）"""
        if self._guard_tokenizer is None or self._guard_model is None:
            torch_dtype = resolve_dtype()
            
            # 确定实际使用的模型路径
            model_path = get_model_path(
                GUARD_ID,
                GUARD_LOCAL_PATH,
                GUARD_CONTAINER_PATH,
                GUARD_WORKSPACE_PATH
            )
            
            print(f"[ModelManager] Loading Guard: {model_path} (dtype: {torch_dtype})")
            self._guard_tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self._guard_tokenizer.pad_token is None:
                self._guard_tokenizer.pad_token = self._guard_tokenizer.eos_token

            self._guard_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch_dtype,  # 使用dtype代替torch_dtype（修复deprecation警告）
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self._guard_model.eval()
            print("[ModelManager] Guard loaded successfully")
        return self._guard_tokenizer, self._guard_model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stop_sequences: list[str] | None = None,
    ) -> Tuple[str, int, int, float]:
        """
        生成文本

        Returns:
            (output_text, input_tokens, output_tokens, latency_ms)
        """
        tokenizer, model = self.load_llm()
        device = next(model.parameters()).device

        # 构建完整提示
        if system_prompt:
            full_prompt = f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        else:
            full_prompt = prompt

        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        input_token_count = inputs.input_ids.shape[1]

        # 准备停止序列
        stop_token_ids = []
        if stop_sequences:
            for seq in stop_sequences:
                token_ids = tokenizer.encode(seq, add_special_tokens=False)
                if token_ids:
                    stop_token_ids.extend(token_ids)

        start_time = time.time()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else None,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        latency_ms = (time.time() - start_time) * 1000

        # 解码输出（只取新生成的部分）
        generated_ids = output_ids[0][input_token_count:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        output_token_count = len(generated_ids)

        return output_text, input_token_count, output_token_count, latency_ms

    def format_guard_prompt(self, user_text: str) -> str:
        """构造 Llama Guard 预期的对话模板。"""
        system_directive = (
            "You are Llama Guard, a safety classifier. "
            "Classify the user's request as SAFE or UNSAFE. "
            "Return JSON in format: {\"verdict\": \"SAFE\"|\"UNSAFE\", \"reason\": \"<short explanation>\"}."
        )
        return (
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{system_directive}\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{user_text}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

    def moderate(
        self,
        text: str,
        threshold: float = 0.5,
        categories: Optional[list[str]] = None,
    ) -> dict:
        """
        使用 Guard 模型审核文本

        Returns:
            GuardResult 格式的字典
        """
        tokenizer, model = self.load_guard()
        device = next(model.parameters()).device

        prompt = self.format_guard_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                top_p=0.8,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        
        # 解析 Guard 响应
        # 首先尝试找到完整的 JSON 对象（支持嵌套）
        verdict_raw = "UNSAFE"
        reason = ""
        json_parsed = False
        
        # 方法1: 尝试直接解析整个响应
        try:
            guard_json = json.loads(response)
            verdict_raw = guard_json.get("verdict", "UNSAFE").upper()
            reason = guard_json.get("reason", "") or "Guard classification completed"
            json_parsed = True
        except json.JSONDecodeError:
            # 方法2: 尝试提取 JSON 对象（处理嵌套）
            # 找到第一个 { 和最后一个匹配的 }
            start_idx = response.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(response)):
                    if response[i] == '{':
                        brace_count += 1
                    elif response[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    try:
                        guard_json = json.loads(json_str)
                        verdict_raw = guard_json.get("verdict", "UNSAFE").upper()
                        reason = guard_json.get("reason", "") or "Guard classification completed"
                        json_parsed = True
                    except json.JSONDecodeError:
                        pass
        
        # 方法3: 如果 JSON 解析失败，尝试从文本中提取 verdict
        if not json_parsed:
            response_upper = response.upper()
            # 检查是否包含 verdict 信息
            if "SAFE" in response_upper:
                # 确保不是 UNSAFE 的一部分
                safe_idx = response_upper.find("SAFE")
                unsafe_idx = response_upper.find("UNSAFE")
                if unsafe_idx == -1 or (safe_idx != -1 and safe_idx < unsafe_idx):
                    verdict_raw = "SAFE"
                    reason = "Guard classified as SAFE"
                else:
                    verdict_raw = "UNSAFE"
                    # 尝试提取原因文本
                    reason_match = re.search(r'reason["\']?\s*:\s*["\']?([^"\']+)', response, re.IGNORECASE)
                    if reason_match:
                        reason = reason_match.group(1).strip()
                    else:
                        reason = "Guard classified as UNSAFE"
            elif "UNSAFE" in response_upper:
                verdict_raw = "UNSAFE"
                # 尝试提取原因文本
                reason_match = re.search(r'reason["\']?\s*:\s*["\']?([^"\']+)', response, re.IGNORECASE)
                if reason_match:
                    reason = reason_match.group(1).strip()
                else:
                    # 尝试提取更多上下文作为原因
                    unsafe_idx = response_upper.find("UNSAFE")
                    context_start = max(0, unsafe_idx - 50)
                    context_end = min(len(response), unsafe_idx + 100)
                    context = response[context_start:context_end].strip()
                    reason = f"Guard classified as UNSAFE: {context}"
            else:
                # 默认情况：无法确定，但提供原始响应的一部分
                verdict_raw = "UNSAFE"
                reason = f"Guard response (unable to parse): {response[:150]}"
        
        # 确保 reason 不为空
        if not reason:
            reason = "Guard classification completed"

        # 转换为前端需要的格式
        is_safe = verdict_raw == "SAFE"
        
        # 计算风险分数（基于 verdict 和 threshold）
        # 如果 UNSAFE，分数应该高于 threshold
        risk_score = 0.3 if is_safe else max(threshold + 0.2, 0.7)
        
        # 确定严重程度
        if is_safe or risk_score < threshold:
            severity = "low"
            verdict = "allow"
        elif risk_score < 0.7:
            severity = "medium"
            verdict = "flag"
        else:
            severity = "high"
            verdict = "block"

        # 构建类别分数（简化版，实际应该从 Guard 输出中提取）
        default_categories = categories or ["violence", "politics", "self-harm", "harassment"]
        category_scores = [
            {
                "id": cat,
                "label": cat,
                "score": risk_score * (0.8 + (i * 0.1)) if not is_safe else 0.1,
            }
            for i, cat in enumerate(default_categories)
        ]

        return {
            "verdict": verdict,
            "severity": severity,
            "rationale": [reason] if reason else ["Guard classification completed"],
            "categories": category_scores,
            "blockedText": text if verdict == "block" else None,
        }

