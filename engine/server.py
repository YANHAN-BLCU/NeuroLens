"""
FastAPI 后端服务器
提供推理和审核 API 端点
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine.models import ModelManager

app = FastAPI(title="NeuroBreak API", version="1.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化模型管理器
model_manager = ModelManager()


# Pydantic 模型定义（匹配前端类型）
class InferenceConfig(BaseModel):
    modelId: str
    temperature: float
    topP: float
    topK: int
    maxTokens: int
    repetitionPenalty: float
    presencePenalty: float
    frequencyPenalty: float
    stopSequences: list[str]
    stream: bool


class GuardConfig(BaseModel):
    modelId: str
    threshold: float
    autoBlock: bool
    categories: list[str]


class PipelineRequest(BaseModel):
    prompt: str
    context: Optional[str] = None
    systemPrompt: Optional[str] = None
    inferenceConfig: InferenceConfig
    guardConfig: GuardConfig


class GuardCategoryScore(BaseModel):
    id: str
    label: str
    score: float
    description: Optional[str] = None


class GuardResult(BaseModel):
    verdict: str  # "allow" | "flag" | "block"
    severity: str  # "low" | "medium" | "high" | "critical"
    rationale: list[str]
    categories: list[GuardCategoryScore]
    blockedText: Optional[str] = None


class PipelineResponse(BaseModel):
    id: str
    createdAt: str
    inference: dict
    guard: GuardResult


class ModerationRequest(BaseModel):
    text: str
    threshold: float
    categories: Optional[list[str]] = None


@app.get("/")
async def root():
    """健康检查端点"""
    return {
        "status": "ok",
        "service": "NeuroBreak API",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """健康检查端点"""
    return {"status": "healthy"}


@app.post("/api/pipeline/run", response_model=PipelineResponse)
async def run_pipeline(request: PipelineRequest):
    """
    执行推理 + Guard 联合流程
    """
    try:
        # 1. 先进行 Guard 审核（如果启用自动拦截）
        if request.guardConfig.autoBlock:
            guard_result = model_manager.moderate(
                request.prompt,
                threshold=request.guardConfig.threshold,
                categories=request.guardConfig.categories,
            )
            # 如果被拦截，直接返回
            if guard_result["verdict"] == "block":
                return PipelineResponse(
                    id=str(uuid.uuid4()),
                    createdAt=datetime.now(timezone.utc).isoformat(),
                    inference={
                        "output": "",
                        "tokens": {"input": 0, "output": 0},
                        "latencyMs": 0,
                        "finishReason": "blocked",
                    },
                    guard=GuardResult(**guard_result),
                )

        # 2. 执行推理
        output_text, input_tokens, output_tokens, latency_ms = model_manager.generate(
            prompt=request.prompt,
            system_prompt=request.systemPrompt,
            max_tokens=request.inferenceConfig.maxTokens,
            temperature=request.inferenceConfig.temperature,
            top_p=request.inferenceConfig.topP,
            top_k=request.inferenceConfig.topK,
            repetition_penalty=request.inferenceConfig.repetitionPenalty,
            stop_sequences=request.inferenceConfig.stopSequences,
        )

        # 3. 对输出进行 Guard 审核
        guard_result = model_manager.moderate(
            output_text,
            threshold=request.guardConfig.threshold,
            categories=request.guardConfig.categories,
        )

        # 4. 构建响应
        return PipelineResponse(
            id=str(uuid.uuid4()),
            createdAt=datetime.now(timezone.utc).isoformat(),
            inference={
                "output": output_text,
                "tokens": {"input": input_tokens, "output": output_tokens},
                "latencyMs": round(latency_ms, 2),
                "finishReason": "stop",
            },
            guard=GuardResult(**guard_result),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@app.post("/api/moderate", response_model=GuardResult)
async def moderate_only(request: ModerationRequest):
    """
    独立安全审核文本
    """
    try:
        guard_result = model_manager.moderate(
            request.text,
            threshold=request.threshold,
            categories=request.categories,
        )
        return GuardResult(**guard_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Moderation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)

