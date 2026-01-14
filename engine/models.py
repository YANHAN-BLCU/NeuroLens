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


# 默认推理模型：8B 版本（ModelScope）
LLM_ID = "LLM-Research/Meta-Llama-3-8B-Instruct"
GUARD_ID = "LLM-Research/Llama-Guard-3-8B"

# 模型路径配置：优先使用本地路径（/cache/），如果不存在则使用 ModelScope ID
LLM_LOCAL_PATH = os.getenv("LLM_LOCAL_PATH", "F:/models/Meta-Llama-3-8B-Instruct")
GUARD_LOCAL_PATH = os.getenv("GUARD_LOCAL_PATH", "F:/models/Llama-Guard-3-8B")
# Docker容器内路径（优先检查 /cache，这是当前容器的挂载点）
LLM_CONTAINER_PATH = os.getenv("LLM_CONTAINER_PATH", "/cache/Meta-Llama-3-8B-Instruct")
GUARD_CONTAINER_PATH = os.getenv("GUARD_CONTAINER_PATH", "/cache/Llama-Guard-3-8B")
# 备用路径（ms_models 目录）
LLM_WORKSPACE_PATH = "/workspace/ms_models/Meta-Llama-3-8B-Instruct"
GUARD_WORKSPACE_PATH = "/workspace/ms_models/Llama-Guard-3-8B"


def resolve_dtype() -> torch.dtype:
    """选择合适的数据类型以平衡稳定性与显存占用。"""
    if not torch.cuda.is_available():
        return torch.float32

    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def get_model_path(model_id: str, local_path: str, container_path: str, workspace_path: str = "") -> str:
    """
    获取模型路径：优先使用本地路径，如果不存在则检查 ModelScope 缓存或下载
    
    Args:
        model_id: ModelScope模型ID
        local_path: Windows本地路径（F盘）
        container_path: Docker容器内路径（主要）
        workspace_path: Docker容器内备用路径
    
    Returns:
        模型路径（本地路径、容器路径或 ModelScope 缓存路径）
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
    
    # 检查备用容器路径（workspace_path，通常是 /workspace/ms_models/...）
    if workspace_path_obj and workspace_path_obj.exists():
        config_file = workspace_path_obj / "config.json"
        if config_file.exists():
            print(f"[ModelManager] 使用备用容器路径: {workspace_path}")
            return str(workspace_path_obj)
    
    # 检查父目录是否存在（可能模型在子目录中）
    # 优先检查 workspace_path 的父目录（/workspace/ms_models）
    parent_paths = []
    if workspace_path_obj:
        parent_paths.append(workspace_path_obj.parent)
    if container_path_obj.parent not in parent_paths:
        parent_paths.append(container_path_obj.parent)
    
    # 提取模型名称（从 model_id 或路径中）
    model_name_parts = model_id.split("/")
    model_name = model_name_parts[-1] if len(model_name_parts) > 0 else ""
    model_id_flat = model_id.replace("/", "_")
    
    for parent_path in parent_paths:
        if parent_path and parent_path.exists():
            # 尝试查找模型目录
            for item in parent_path.iterdir():
                if item.is_dir():
                    # 检查是否有 config.json
                    if (item / "config.json").exists():
                        # 检查目录名是否匹配模型名称（更宽松的匹配）
                        item_name = item.name
                        if (model_name in item_name or 
                            item_name in model_name or
                            model_id_flat in item_name or
                            model_id_flat.replace("-", "_") in item_name.replace("-", "_")):
                            print(f"[ModelManager] 在父目录中找到模型: {item}")
                            return str(item)
    
    # 额外检查：直接检查 /workspace/ms_models 目录（如果存在）
    ms_models_dir = Path("/workspace/ms_models")
    if ms_models_dir.exists() and ms_models_dir.is_dir():
        print(f"[ModelManager] 检查 /workspace/ms_models 目录...")
        print(f"[ModelManager] 查找模型: {model_name} (ID: {model_id})")
        
        # 列出所有子目录用于调试
        found_dirs = []
        for item in ms_models_dir.iterdir():
            if item.is_dir():
                has_config = (item / "config.json").exists()
                found_dirs.append((item.name, has_config))
                if has_config:
                    print(f"[ModelManager]   发现目录: {item.name} (有 config.json)")
        
        # 尝试匹配模型目录
        for item in ms_models_dir.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                item_name = item.name
                # 提取关键匹配词：Meta-Llama-3-8B-Instruct -> ["Meta", "Llama", "3", "8B", "Instruct"]
                # Llama-Guard-3-8B -> ["Llama", "Guard", "3", "8B"]
                key_parts = [p for p in model_name.split("-") if len(p) > 1]
                # 更宽松的匹配：只要包含关键部分就认为匹配
                matches = (
                    model_name in item_name or 
                    item_name in model_name or
                    model_id_flat in item_name or
                    any(part in item_name for part in key_parts if len(part) > 2) or
                    # 特殊匹配：Meta-Llama-3-8B-Instruct 匹配包含 "Meta-Llama-3-8B" 的目录
                    (model_name.startswith("Meta-Llama") and "Meta-Llama" in item_name) or
                    (model_name.startswith("Llama-Guard") and "Llama-Guard" in item_name)
                )
                if matches:
                    print(f"[ModelManager] 在 /workspace/ms_models 中找到匹配的模型: {item}")
                    return str(item)
        
        # 如果没找到匹配的，但只有一个包含 config.json 的目录，也使用它
        config_dirs = [item for item in ms_models_dir.iterdir() 
                      if item.is_dir() and (item / "config.json").exists()]
        if len(config_dirs) == 1:
            print(f"[ModelManager] 在 /workspace/ms_models 中找到唯一模型目录: {config_dirs[0]}")
            return str(config_dirs[0])
        elif len(config_dirs) > 1:
            print(f"[ModelManager] 警告: /workspace/ms_models 中有多个模型目录: {[d.name for d in config_dirs]}")
            # 尝试根据模型类型选择
            for config_dir in config_dirs:
                dir_name = config_dir.name
                if model_name.startswith("Meta-Llama") and "Llama" in dir_name and "Guard" not in dir_name:
                    print(f"[ModelManager] 选择推理模型目录: {config_dir}")
                    return str(config_dir)
                elif model_name.startswith("Llama-Guard") and "Guard" in dir_name:
                    print(f"[ModelManager] 选择 Guard 模型目录: {config_dir}")
                    return str(config_dir)
    
    # 检查Windows本地路径（在容器内通常不存在，但保留检查）
    if local_path_obj.exists():
        config_file = local_path_obj / "config.json"
        if config_file.exists():
            print(f"[ModelManager] 使用本地路径: {local_path}")
            return str(local_path_obj)
    
    # 检查 ModelScope 缓存目录（递归搜索）
    modelscope_cache = os.getenv("MODELSCOPE_CACHE") or os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
    if not modelscope_cache:
        modelscope_cache = str(Path.home() / ".cache" / "modelscope" / "hub")
    
    modelscope_cache_path = Path(modelscope_cache)
    if modelscope_cache_path.exists():
        # ModelScope 的目录结构可能是：
        # - cache_dir/model_id/revision/
        # - cache_dir/models--org--model_name/snapshots/hash/
        # 递归搜索包含 config.json 的目录
        model_name_parts = model_id.split("/")
        model_org = model_name_parts[0] if len(model_name_parts) > 0 else ""
        model_name = model_name_parts[-1] if len(model_name_parts) > 0 else ""
        
        # 搜索策略1: 直接匹配 model_id 格式 (org_model_name)
        model_id_flat = model_id.replace("/", "_")
        for root, dirs, files in os.walk(modelscope_cache_path):
            root_path = Path(root)
            # 检查当前目录是否有 config.json
            if (root_path / "config.json").exists():
                # 检查目录名是否匹配
                dir_name = root_path.name
                parent_name = root_path.parent.name if root_path.parent != root_path else ""
                # 匹配多种可能的命名格式
                if (model_id_flat in dir_name or 
                    model_name in dir_name or 
                    model_org in dir_name or
                    model_id_flat in parent_name or
                    model_name in parent_name):
                    print(f"[ModelManager] 在 ModelScope 缓存中找到模型: {root_path}")
                    return str(root_path)
        
        # 搜索策略2: 查找 models--org--model 格式的目录
        models_prefix = f"models--{model_org.replace('-', '--')}--{model_name.replace('-', '--')}"
        for item in modelscope_cache_path.iterdir():
            if item.is_dir() and models_prefix in item.name:
                # 在 snapshots 子目录中查找
                snapshots_dir = item / "snapshots"
                if snapshots_dir.exists():
                    for snapshot in snapshots_dir.iterdir():
                        if snapshot.is_dir() and (snapshot / "config.json").exists():
                            print(f"[ModelManager] 在 ModelScope 缓存中找到模型: {snapshot}")
                            return str(snapshot)
    
    # 如果所有路径都不存在，返回 ModelScope ID
    print(f"[ModelManager] 警告: 未找到模型路径，将使用 ModelScope ID: {model_id}")
    print(f"[ModelManager] 提示: 请确保模型已下载，或运行 python scripts/download_models.py --all-8b 下载模型")
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
            
            # 检查路径是否存在
            if "/" in model_path and not Path(model_path).exists():
                raise FileNotFoundError(
                    f"模型路径不存在: {model_path}\n"
                    f"请确保模型已下载，或运行: python scripts/download_models.py --all-8b"
                )
            
            self._llm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self._llm_tokenizer.pad_token is None:
                self._llm_tokenizer.pad_token = self._llm_tokenizer.eos_token

            self._llm_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
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
            
            # 检查路径是否存在
            if "/" in model_path and not Path(model_path).exists():
                raise FileNotFoundError(
                    f"模型路径不存在: {model_path}\n"
                    f"请确保模型已下载，或运行: python scripts/download_models.py --all-8b"
                )
            
            self._guard_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self._guard_tokenizer.pad_token is None:
                self._guard_tokenizer.pad_token = self._guard_tokenizer.eos_token

            self._guard_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
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

        # 确定是否使用采样
        do_sample = temperature > 0
        
        # 构建生成参数字典，避免传递无效参数
        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        
        # 只在 do_sample=True 时添加采样相关参数，避免警告
        if do_sample:
            generate_kwargs["temperature"] = temperature
            if top_p < 1.0:  # top_p=1.0 时不需要传递
                generate_kwargs["top_p"] = top_p
            if top_k > 0:
                generate_kwargs["top_k"] = top_k
        
        # repetition_penalty 通常总是有效
        if repetition_penalty != 1.0:
            generate_kwargs["repetition_penalty"] = repetition_penalty

        start_time = time.time()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                **generate_kwargs,
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
        threshold: float = 0.7,
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

