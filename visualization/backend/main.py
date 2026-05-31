"""
NeuroLens Visualization Backend
FastAPI server for serving visualization data

v2.0 - 重构版:
- WebSocket 实时进度推送
- 模型发现 (扫描 models/ 目录)
- 版本管理 (outputs/{model}/{version}/)
- 真实 pipeline 调用
"""

import subprocess
import threading
import uuid as _uuid
from fastapi import FastAPI, Query, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os
import sys
import asyncio
from datetime import datetime
from pathlib import Path


# ─── Path Resolution ──────────────────────────────────────────────────────────

def _resolve_runtime_paths() -> tuple[str, str]:
    """Resolve (project_root, backend_dir) for both source and PyInstaller runtime."""
    if getattr(sys, "frozen", False):
        executable_dir = os.path.dirname(sys.executable)
        backend_dir = os.path.join(executable_dir, "_internal")
        return os.path.normpath(executable_dir), os.path.normpath(backend_dir)
    backend_dir = os.path.dirname(__file__)
    project_root = os.path.normpath(os.path.join(backend_dir, "..", ".."))
    return project_root, os.path.normpath(backend_dir)


def read_version_from_repo_root(repo_root: str) -> str:
    candidates = [os.path.join(repo_root, "VERSION"), os.path.join(repo_root, "_internal", "VERSION")]
    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8") as f:
                line = (f.readline() or "").strip()
                if line:
                    return line
        except OSError:
            pass
    return "2.0.0"


_PROJECT_ROOT, _BACKEND_DIR = _resolve_runtime_paths()
NEUROLENS_VERSION = read_version_from_repo_root(_PROJECT_ROOT)


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="NeuroLens Visualization API",
    description="Backend API for NeuroLens visualization system",
    version=NEUROLENS_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Constants ────────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, "outputs")
SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
DATASET_PATH = os.path.join(_PROJECT_ROOT, "data", "salad", "raw", "attack_enhanced_set_train.jsonl")

# 当前活跃的模型和版本 (可通过 API 切换)
def _default_active_model() -> dict:
    """扫描 models/ 目录，默认加载第一个模型的最新版本；若 models/ 为空则从 outputs/ 兜底。"""
    def _pick_version(model_output: str) -> dict | None:
        if not os.path.isdir(model_output):
            return None
        model_name = os.path.basename(model_output)
        versions = sorted(os.listdir(model_output), reverse=True)
        for preferred in ["test_snip"]:
            if preferred in versions:
                vpath = os.path.join(model_output, preferred)
                if os.path.isdir(vpath):
                    return {"model": model_name, "version": preferred, "data_root": vpath}
        for v in versions:
            vpath = os.path.join(model_output, v)
            if os.path.isdir(vpath) and v != "baseline":
                return {"model": model_name, "version": v, "data_root": vpath}
        baseline = os.path.join(model_output, "baseline")
        if os.path.isdir(baseline):
            return {"model": model_name, "version": "baseline", "data_root": baseline}
        return None

    # 优先从 models/ 找（有权重的模型）
    if os.path.isdir(MODELS_DIR):
        for item in sorted(os.listdir(MODELS_DIR)):
            item_path = os.path.join(MODELS_DIR, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
                result = _pick_version(os.path.join(OUTPUTS_DIR, item))
                if result:
                    return result
                return {"model": item, "version": "", "data_root": OUTPUTS_DIR}

    # 兜底：从 outputs/ 找有数据的模型
    if os.path.isdir(OUTPUTS_DIR):
        for item in sorted(os.listdir(OUTPUTS_DIR)):
            result = _pick_version(os.path.join(OUTPUTS_DIR, item))
            if result:
                return result

    return {"model": "", "version": "baseline", "data_root": OUTPUTS_DIR}

_active_model = _default_active_model()

# WebSocket 连接管理
_ws_connections: list[WebSocket] = []


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _get_data_root() -> str:
    """获取当前活跃的数据根目录"""
    return _active_model["data_root"]


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load JSON file from current data root"""
    full_path = os.path.join(_get_data_root(), filepath)
    if not os.path.exists(full_path):
        return {}
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _get_python_executable() -> str:
    """获取 Python 可执行文件路径
    
    优先级:
    1. frozen 模式下使用 sys.executable (打包的 Python)
    2. dev_env 中的 python (开发环境，需验证可用)
    3. 系统 python
    """
    if getattr(sys, "frozen", False):
        return sys.executable
    env_python = os.path.join(_PROJECT_ROOT, "dev_env", "Scripts", "python.exe")
    if os.path.isfile(env_python):
        # 验证 dev_env python 实际可用（可能指向不存在的路径）
        try:
            result = subprocess.run(
                [env_python, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return env_python
        except Exception:
            pass
    return sys.executable


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": NEUROLENS_VERSION,
        "timestamp": datetime.now().isoformat(),
        "active_model": _active_model["model"],
        "active_version": _active_model["version"],
    }


# ─── Static Files ────────────────────────────────────────────────────────────

# ─── Serve Home Page ──────────────────────────────────────────────────────────
_home_dist = os.path.join(_BACKEND_DIR, "home-dashboard", "dist")

@app.get("/")
async def serve_home():
    home_index = os.path.join(_home_dist, "index.html")
    if os.path.exists(home_index):
        return FileResponse(home_index)
    return RedirectResponse(url="/dashboard")

@app.get("/dashboard")
async def serve_dashboard():
    return FileResponse(os.path.join(_BACKEND_DIR, "index.html"))

# Serve home-dashboard assets at /assets/... (home page uses absolute paths)
@app.get("/assets/{path:path}")
async def serve_assets(path: str):
    file_path = os.path.join(_home_dist, "assets", path)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse({"error": "Asset not found"}, status_code=404)

@app.get("/neurolens.png")
async def serve_logo():
    logo_path = os.path.join(_BACKEND_DIR, "neurolens.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/png")
    return JSONResponse({"error": "logo not found"}, status_code=404)

# Serve vis/ panel files
vis_dir = os.path.join(_BACKEND_DIR, "vis")
if os.path.isdir(vis_dir):
    app.mount("/vis", StaticFiles(directory=vis_dir), name="vis")

# Serve APISafety frontend
apisafety_dir = os.path.join(_BACKEND_DIR, "apisafety")
if os.path.isdir(apisafety_dir):
    app.mount("/apisafety", StaticFiles(directory=apisafety_dir, html=True), name="apisafety")

# Serve Help & Services frontend
help_dist = os.path.join(_BACKEND_DIR, "help-dashboard", "dist")
if os.path.isdir(help_dist):
    app.mount("/help", StaticFiles(directory=help_dist, html=True), name="help")

# Serve Desensization frontend (local encrypt/decrypt)
desens_dist = os.path.join(_BACKEND_DIR, "Desensization-dashboard", "dist")
_desens_dir = os.path.join(_BACKEND_DIR, "Desensization-dashboard")
if os.path.isdir(desens_dist):
    app.mount("/desensization", StaticFiles(directory=desens_dist, html=True), name="desensization")

_ner_predict_fn = None


def _load_ner_predict():
    """从 Desensization-dashboard/ner_service.py 加载推理函数（兼容 PyInstaller 打包）。"""
    global _ner_predict_fn
    if _ner_predict_fn is not None:
        return _ner_predict_fn

    import importlib.util

    ner_path = os.path.join(_desens_dir, "ner_service.py")
    if not os.path.isfile(ner_path):
        raise FileNotFoundError(
            f"未找到脱敏 NER 模块：{ner_path}。"
            "请确认已重新执行 PyInstaller 打包（NeuroLens.spec 需包含 ner_service.py）。"
        )

    for name in ("chinese-ner-per-addr-rbt3", "chinese-ner-per-addr"):
        model_dir = os.path.join(_desens_dir, "models", name)
        if os.path.isdir(model_dir):
            os.environ["NL_DESENS_MODEL_DIR"] = model_dir
            break

    spec = importlib.util.spec_from_file_location("nl_ner_service", ner_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 NER 模块：{ner_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "predict_entities"):
        raise ImportError(f"{ner_path} 中缺少 predict_entities")

    _ner_predict_fn = module.predict_entities
    return _ner_predict_fn


class DesensNerRequest(BaseModel):
    text: str


@app.post("/api/desensization/ner")
async def desensization_ner(request: DesensNerRequest):
    """对文本运行微调 NER，识别姓名与地址。"""
    text = request.text or ""
    if not text.strip():
        return {"entities": []}
    try:
        predict_entities = _load_ner_predict()
        entities = await asyncio.to_thread(predict_entities, text)
        return {"entities": entities}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        if missing in {"torch", "transformers"}:
            detail = (
                f"NER 依赖未安装（缺少 {missing}）。请在打包/运行环境中安装："
                f" pip install torch transformers  （Python: {sys.executable}）"
            )
        else:
            detail = f"NER 模块加载失败（缺少 {missing}）。请重新打包或检查 _internal 是否完整。"
        raise HTTPException(status_code=503, detail=detail) from exc
    except Exception as exc:
        import traceback

        detail = f"NER 推理失败: {exc}"
        if getattr(sys, "frozen", False):
            detail += "\n" + traceback.format_exc()
        raise HTTPException(status_code=503, detail=detail) from exc


# Mount outputs directory - dynamically serves from active model/version
@app.get("/outputs/{path:path}")
async def serve_outputs(path: str):
    """Serve output files: version dir -> baseline dir -> root outputs"""
    # 1. Try version-specific directory
    data_root = _get_data_root()
    file_path = os.path.join(data_root, path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)

    # 2. Try baseline directory for the active model
    active = _active_model.get("model", "")
    if active:
        baseline_path = os.path.join(OUTPUTS_DIR, active, "baseline", path)
        if os.path.exists(baseline_path) and os.path.isfile(baseline_path):
            return FileResponse(baseline_path)

    # 3. Try any model's baseline
    if os.path.isdir(OUTPUTS_DIR):
        for model_dir in sorted(os.listdir(OUTPUTS_DIR)):
            candidate = os.path.join(OUTPUTS_DIR, model_dir, "baseline", path)
            if os.path.exists(candidate) and os.path.isfile(candidate):
                return FileResponse(candidate)

    # 4. Try root outputs (legacy)
    root_path = os.path.join(OUTPUTS_DIR, path)
    if os.path.exists(root_path) and os.path.isfile(root_path):
        return FileResponse(root_path)

    return JSONResponse({"error": f"File not found: {path}"}, status_code=404)


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class PipelineConfig(BaseModel):
    model: str
    attack_types: List[str] = ["all"]
    threshold: float = 0.5
    finetune_method: str = "none"
    level: str = "quick"
    batch_size: int = 4
    max_tokens: int = 64

class InterventionRequest(BaseModel):
    neuron_ids: List[str]
    sample_ids: List[str]

class FinetuneRequest(BaseModel):
    method: str
    config: Dict[str, Any]

class ModelSwitchRequest(BaseModel):
    model: str
    version: str = "baseline"


# ═════════════════════════════════════════════════════════════════════════════
# NEW: Model Management API
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/models")
async def list_models():
    """列出所有可用模型：扫描 models/ 目录，同时把 outputs/ 里每个模型的 baseline 也作为独立条目。"""
    seen: dict[str, dict] = {}

    # 1. 扫描 models/（有权重的模型，读取 config.json 补充元信息）
    if os.path.isdir(MODELS_DIR):
        for item in os.listdir(MODELS_DIR):
            item_path = os.path.join(MODELS_DIR, item)
            config_path = os.path.join(item_path, "config.json")
            if not os.path.isdir(item_path):
                continue
            entry: dict = {"name": item, "path": item_path}
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                    entry.update({
                        "num_layers": config.get("num_hidden_layers", 0),
                        "hidden_size": config.get("hidden_size", 0),
                        "vocab_size": config.get("vocab_size", 0),
                        "model_type": config.get("model_type", "unknown"),
                    })
                except Exception:
                    pass
            seen[item] = entry

    # 2. 把 outputs/{model}/baseline 也作为独立模型条目（名称："{model}/baseline"）
    if os.path.isdir(OUTPUTS_DIR):
        for item in os.listdir(OUTPUTS_DIR):
            baseline_path = os.path.join(OUTPUTS_DIR, item, "baseline")
            if not os.path.isdir(baseline_path):
                continue
            key = f"{item}/baseline"
            if key not in seen:
                seen[key] = {
                    "name": key,
                    "path": baseline_path,
                    "data_only": True,
                    "baseline_of": item,
                }

    return {"models": list(seen.values()), "active": _active_model}


@app.get("/api/models/{model_name}/versions")
async def list_model_versions(model_name: str):
    """列出某个模型的所有版本。baseline 条目本身没有子版本。"""
    # "{base}/baseline" 条目：数据根就是 baseline 目录，无子版本
    if model_name.endswith("/baseline"):
        base = model_name[: -len("/baseline")]
        baseline_path = os.path.join(OUTPUTS_DIR, base, "baseline")
        active_ver = _active_model["version"] if _active_model["model"] == model_name else ""
        return {"model": model_name, "versions": [], "active": active_ver}

    model_output_dir = os.path.join(OUTPUTS_DIR, model_name)
    versions = []
    if os.path.isdir(model_output_dir):
        for item in os.listdir(model_output_dir):
            item_path = os.path.join(model_output_dir, item)
            if os.path.isdir(item_path):
                meta_path = os.path.join(item_path, "pipeline_meta.json")
                meta = {}
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception:
                        pass
                versions.append({
                    "name": item,
                    "path": item_path,
                    "meta": meta,
                })
    return {"model": model_name, "versions": versions, "active": _active_model["version"]}


@app.post("/api/models/switch")
async def switch_model(request: ModelSwitchRequest):
    """切换当前活跃的模型和版本"""
    global _active_model

    model_name = request.model
    version = request.version

    # "{base}/baseline" 条目：data_root 直接指向 baseline 目录，无子版本层级
    if model_name.endswith("/baseline"):
        base = model_name[: -len("/baseline")]
        data_root = os.path.join(OUTPUTS_DIR, base, "baseline")
        version = ""
    else:
        data_root = os.path.join(OUTPUTS_DIR, model_name, version)

    if not os.path.isdir(data_root):
        os.makedirs(data_root, exist_ok=True)

    _active_model = {
        "model": model_name,
        "version": version,
        "data_root": data_root,
    }

    # 通知所有 WebSocket 客户端
    await _broadcast({
        "type": "model_switched",
        "model": model_name,
        "version": version,
        "data_root": data_root,
    })

    return {"status": "ok", "active": _active_model}


# ═════════════════════════════════════════════════════════════════════════════
# NEW: WebSocket for real-time progress
# ═════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates"""
    await websocket.accept()
    _ws_connections.append(websocket)
    try:
        while True:
            # 保持连接，等待消息
            data = await websocket.receive_text()
            # 客户端可以发送心跳
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        _ws_connections.remove(websocket)
    except Exception:
        if websocket in _ws_connections:
            _ws_connections.remove(websocket)


async def _broadcast(message: dict):
    """向所有 WebSocket 客户端广播消息"""
    dead = []
    for ws in _ws_connections:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_connections.remove(ws)


# ═════════════════════════════════════════════════════════════════════════════
# NEW: Pipeline API (真实执行)
# ═════════════════════════════════════════════════════════════════════════════

# 后台任务管理
_pipeline_tasks: Dict[str, dict] = {}


@app.post("/api/pipeline/run")
async def run_pipeline(config: PipelineConfig):
    """启动真实 pipeline"""
    task_id = str(_uuid.uuid4())

    # 确定模型路径
    model_path = os.path.join(MODELS_DIR, config.model)
    if not os.path.isdir(model_path):
        raise HTTPException(status_code=404, detail=f"模型不存在: {config.model}")

    # 确定输出目录
    version = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.finetune_method != "none":
        version = f"{config.finetune_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(OUTPUTS_DIR, config.model, version)

    # 切换到新版本
    global _active_model
    _active_model = {
        "model": config.model,
        "version": version,
        "data_root": output_dir,
    }

    # 记录任务
    _pipeline_tasks[task_id] = {
        "task_id": task_id,
        "status": "starting",
        "model": config.model,
        "version": version,
        "level": config.level,
        "output_dir": output_dir,
        "start_time": datetime.now().isoformat(),
        "progress": 0,
        "batch_size": config.batch_size,
        "max_tokens": config.max_tokens,
    }

    # 在后台线程中运行 pipeline
    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(task_id, model_path, config.level, output_dir, config.batch_size, config.max_tokens),
        daemon=True,
    )
    thread.start()

    return {
        "task_id": task_id,
        "status": "started",
        "model": config.model,
        "version": version,
        "output_dir": output_dir,
    }


def _run_pipeline_thread(task_id: str, model_path: str, level: str, output_dir: str, batch_size: int = 4, max_tokens: int = 64):
    """在后台线程中运行 pipeline"""
    try:
        _pipeline_tasks[task_id]["status"] = "running"

        if getattr(sys, "frozen", False):
            _run_pipeline_inprocess(task_id, model_path, level, output_dir, batch_size, max_tokens)
        else:
            _run_pipeline_subprocess(task_id, model_path, level, output_dir, batch_size, max_tokens)

    except Exception as e:
        _pipeline_tasks[task_id]["status"] = "failed"
        _pipeline_tasks[task_id]["error"] = str(e)


def _run_pipeline_inprocess(task_id: str, model_path: str, level: str, output_dir: str):
    """Frozen 模式：直接导入 pipeline 模块执行"""
    try:
        # scripts/ 目录已打包到 _internal/scripts/
        scripts_dir = os.path.join(os.path.dirname(sys.executable), "_internal", "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        # 导入 pipeline 模块
        import importlib.util
        pipeline_script = os.path.join(scripts_dir, "run_pipeline.py")
        spec = importlib.util.spec_from_file_location("run_pipeline", pipeline_script)
        pipeline_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline_mod)

        # 定义进度回调
        def on_progress(phase, progress, detail=None):
            _pipeline_tasks[task_id]["progress"] = int(progress * 100)
            if detail:
                _pipeline_tasks[task_id]["last_output"] = str(detail)

        # 执行 pipeline
        result = pipeline_mod.run_full_pipeline(model_path, level, Path(output_dir))

        _pipeline_tasks[task_id]["status"] = "completed"
        _pipeline_tasks[task_id]["progress"] = 100
        _pipeline_tasks[task_id]["result"] = result

        # 读取评估结果
        result_file = os.path.join(output_dir, "assessment", "evaluation_results.json")
        if os.path.exists(result_file):
            with open(result_file, "r", encoding="utf-8") as f:
                _pipeline_tasks[task_id]["result"] = json.load(f)

    except Exception as e:
        _pipeline_tasks[task_id]["status"] = "failed"
        _pipeline_tasks[task_id]["error"] = str(e)


def _run_pipeline_subprocess(task_id: str, model_path: str, level: str, output_dir: str, batch_size: int = 4, max_tokens: int = 64):
    """开发模式：用 subprocess 调用 scripts/run_pipeline.py"""
    python_exe = _get_python_executable()
    script_path = os.path.join(SCRIPTS_DIR, "run_pipeline.py")

    cmd = [
        python_exe, "-u", script_path,
        "--model-path", model_path,
        "--level", level,
        "--output", output_dir,
        "--batch-size", str(batch_size),
        "--max-tokens", str(max_tokens),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=_PROJECT_ROOT,
        env=env,
    )

    # 读取输出并更新进度
    for line in iter(process.stdout.readline, ""):
        line = line.strip()
        if not line:
            continue

        # 尝试解析JSON进度 (新格式)
        try:
            progress_data = json.loads(line)
            if progress_data.get("type") == "progress":
                _pipeline_tasks[task_id]["progress"] = int(progress_data.get("progress", 0) * 100)
                _pipeline_tasks[task_id]["last_output"] = progress_data
                # 存储日志行
                if "log_lines" not in _pipeline_tasks[task_id]:
                    _pipeline_tasks[task_id]["log_lines"] = []
            elif progress_data.get("type") == "result":
                _pipeline_tasks[task_id]["result"] = progress_data
            continue
        except json.JSONDecodeError:
            pass

        # 普通日志行
        if "log_lines" not in _pipeline_tasks[task_id]:
            _pipeline_tasks[task_id]["log_lines"] = []
        _pipeline_tasks[task_id]["log_lines"].append(line)
        # 只保留最近100行
        if len(_pipeline_tasks[task_id]["log_lines"]) > 100:
            _pipeline_tasks[task_id]["log_lines"] = _pipeline_tasks[task_id]["log_lines"][-100:]

        if "[" in line and "]" in line:
            _pipeline_tasks[task_id]["last_output"] = line

    process.wait()
    returncode = process.returncode

    if returncode == 0:
        _pipeline_tasks[task_id]["status"] = "completed"
        _pipeline_tasks[task_id]["progress"] = 100

        result_file = os.path.join(output_dir, "assessment", "evaluation_results.json")
        if os.path.exists(result_file):
            with open(result_file, "r", encoding="utf-8") as f:
                _pipeline_tasks[task_id]["result"] = json.load(f)
    else:
        _pipeline_tasks[task_id]["status"] = "failed"
        _pipeline_tasks[task_id]["error"] = f"Process exited with code {returncode}"


@app.get("/api/pipeline/status/{task_id}")
async def get_pipeline_status(task_id: str):
    """获取 pipeline 状态"""
    if task_id not in _pipeline_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return _pipeline_tasks[task_id]


@app.get("/api/pipeline/tasks")
async def list_pipeline_tasks():
    """列出所有 pipeline 任务"""
    return {"tasks": list(_pipeline_tasks.values())}


@app.post("/api/pipeline/cancel/{task_id}")
async def cancel_pipeline(task_id: str):
    """取消 pipeline 任务"""
    if task_id not in _pipeline_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = _pipeline_tasks[task_id]
    task["status"] = "cancelled"
    # 终止子进程 (如果有的话)
    if "process" in task:
        try:
            task["process"].terminate()
        except Exception:
            pass
    return {"status": "cancelled", "task_id": task_id}


@app.get("/api/data/attacks")
async def get_attack_types():
    """获取数据集中的攻击类型列表"""
    dataset_path = os.path.join(_PROJECT_ROOT, "data", "salad", "raw", "attack_enhanced_set_train.jsonl")
    attacks = set()
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 1000:  # 只扫描前1000条
                        break
                    try:
                        sample = json.loads(line.strip())
                        method = sample.get("method", "")
                        if method:
                            attacks.add(method)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
    return {"attacks": sorted(list(attacks))}


# ═════════════════════════════════════════════════════════════════════════════
# Metrics API
# ═════════════════════════════════════════════════════════════════════════════

def _load_eval_results() -> Dict[str, Any]:
    """Load evaluation results, trying evaluation_results.json then pipeline_evaluation_v2.json."""
    for candidate in ("assessment/evaluation_results.json", "assessment/pipeline_evaluation_v2.json"):
        data = load_json_file(candidate)
        if data:
            return data
    return {}


def _compute_asr_by_attack(eval_results: Dict[str, Any]) -> Dict[str, float]:
    """Derive per-method ASR from raw results list or pre-computed field."""
    if "results" in eval_results:
        method_stats: Dict[str, Dict[str, int]] = {}
        for r in eval_results["results"]:
            m = r.get("method", "unknown")
            if m not in method_stats:
                method_stats[m] = {"total": 0, "jailbreak": 0}
            method_stats[m]["total"] += 1
            if r.get("jailbreak_success"):
                method_stats[m]["jailbreak"] += 1
        return {m: round(s["jailbreak"] / max(s["total"], 1), 4) for m, s in method_stats.items()}
    if "asr_by_attack" in eval_results:
        return eval_results["asr_by_attack"]
    return {}


@app.get("/api/metrics")
async def get_metrics(
    model_version: Optional[str] = Query(None),
    attack_type: Optional[str] = Query(None),
    time_range: Optional[str] = Query(None)
):
    eval_results = _load_eval_results()
    if eval_results:
        asr_by_attack = _compute_asr_by_attack(eval_results)
        return {
            "overall_asr": eval_results.get("asr", eval_results.get("overall_asr", 0)),
            "asr_by_attack": asr_by_attack,
            "utility_scores": eval_results.get("utility_scores", {}),
            "timestamp": eval_results.get("timestamp", datetime.now().isoformat()),
            "model_version": eval_results.get("model_name", _active_model["model"]),
            "num_samples": eval_results.get("num_samples", 0),
            "num_jailbreak": eval_results.get("num_jailbreak", 0),
        }
    return {
        "overall_asr": 0, "asr_by_attack": {}, "utility_scores": {},
        "timestamp": datetime.now().isoformat(),
        "model_version": _active_model["model"],
    }


@app.get("/api/metrics/asr-by-attack")
async def get_asr_by_attack():
    eval_results = _load_eval_results()
    if eval_results:
        return _compute_asr_by_attack(eval_results)
    return {}


@app.get("/api/metrics/utility-scores")
async def get_utility_scores():
    eval_results = load_json_file("assessment/evaluation_results.json")
    if eval_results and "utility_scores" in eval_results:
        return eval_results["utility_scores"]
    return {}


# ═════════════════════════════════════════════════════════════════════════════
# Representation API
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/representation")
async def get_representation(
    layer_idx: int = Query(...),
    method: str = Query("pca", enum=["pca", "tsne"]),
    sample_ids: Optional[List[str]] = Query(None),
    n_components: int = Query(2, ge=2, le=3)
):
    mode = "decision_boundary" if method == "tsne" else "standard"
    rep_file = os.path.join(
        _get_data_root(), "representation",
        f"representation_layer_{layer_idx}_{mode}.json"
    )
    if os.path.exists(rep_file):
        try:
            with open(rep_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            points = raw.get("points", [])
            coords = [[p["x"], p["y"]] for p in points]
            labels = [1 if p.get("jailbroken") else 0 for p in points]
            return {
                "layer_idx": layer_idx, "method": method,
                "coords": coords, "labels": labels,
                "explained_variance_ratio": raw.get("explained_variance_ratio"),
            }
        except Exception:
            pass
    return {"layer_idx": layer_idx, "method": method, "coords": [], "labels": []}


# ═════════════════════════════════════════════════════════════════════════════
# Layer API
# ═════════════════════════════════════════════════════════════════════════════

AVAILABLE_LAYERS = list(range(32))

@app.get("/api/layers")
async def get_available_layers():
    return AVAILABLE_LAYERS

@app.get("/api/layers/evolution")
async def get_layer_evolution():
    evolution_data = load_json_file("layer_evolution/semantic_evolution.json")
    if evolution_data:
        return evolution_data
    return {}

@app.get("/api/layers/gradients")
async def get_layer_gradients(
    layer_idx: Optional[int] = Query(None),
    neuron_ids: Optional[List[str]] = Query(None)
):
    grad_data = load_json_file("gradient_dependency/gradient_dependency_visualization.json")
    return grad_data if grad_data else {}


# ═════════════════════════════════════════════════════════════════════════════
# Neuron API
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/neurons/quadrants")
async def get_neuron_quadrants(
    layer_idx: Optional[int] = Query(None),
    quadrant: Optional[str] = Query(None)
):
    quad_data = load_json_file("quadrant_classification/quadrant_classification.json")
    return quad_data if quad_data else {}

@app.get("/api/neurons/gradient-dependency")
async def get_neuron_gradient_dependency(
    neuron_id: Optional[str] = Query(None),
    depth: int = Query(1, ge=1, le=3)
):
    grad_data = load_json_file("gradient_dependency/gradient_dependency_visualization.json")
    return grad_data if grad_data else {}

@app.get("/api/neurons/safety")
async def get_safety_neurons():
    safety_data = load_json_file("dedicated_safety_neurons.json")
    return safety_data if safety_data else {}

@app.get("/api/neurons/parameter-alignment")
async def get_parameter_alignment():
    return load_json_file("parameter_alignment/parameter_alignment.json") or {}

@app.get("/api/neurons/activation-projection")
async def get_activation_projection():
    return load_json_file("activation_projection/activation_projection.json") or {}


# ═════════════════════════════════════════════════════════════════════════════
# Instance API
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/instances")
async def get_instances(
    attack_type: Optional[str] = Query(None),
    jailbroken: Optional[bool] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    # 尝试从评估结果中读取
    eval_results = load_json_file("assessment/evaluation_results.json")
    if eval_results and "results" in eval_results:
        instances = eval_results["results"]
        if jailbroken is not None:
            instances = [i for i in instances if i.get("jailbreak_success") == jailbroken]
        return {"instances": instances[offset:offset+limit], "total": len(instances)}
    return {"instances": [], "total": 0}

@app.get("/api/instances/{instance_id}")
async def get_instance(instance_id: str):
    eval_results = load_json_file("assessment/evaluation_results.json")
    if eval_results and "results" in eval_results:
        for inst in eval_results["results"]:
            if str(inst.get("id")) == instance_id:
                return inst
    raise HTTPException(status_code=404, detail="Instance not found")


# ═════════════════════════════════════════════════════════════════════════════
# Probes API
# ═════════════════════════════════════════════════════════════════════════════

def _find_probes_base() -> str:
    canonical = os.path.join(_get_data_root(), "probes", "probes")
    if os.path.isdir(canonical):
        # 查找第一个子目录
        for item in os.listdir(canonical):
            item_path = os.path.join(canonical, item)
            if os.path.isdir(item_path):
                return item_path
    return os.path.join(_get_data_root(), "probes")


@app.get("/api/probes/layers")
async def get_probes_all_layers():
    probes_data = {}
    base_path = _find_probes_base()
    for layer_idx in range(33):
        metrics_file = os.path.join(base_path, f"layer_{layer_idx}", "metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r") as f:
                    probes_data[f"layer_{layer_idx}"] = json.load(f)
            except Exception:
                pass
    return probes_data if probes_data else {}

@app.get("/api/probes/layers/{layer_idx}")
async def get_probes_layer(layer_idx: int):
    layer_dir = os.path.join(_find_probes_base(), f"layer_{layer_idx}")
    metrics_file = os.path.join(layer_dir, "metrics.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=404, detail=f"Layer {layer_idx} not found")


# ═════════════════════════════════════════════════════════════════════════════
# Neuron Scores API
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/neurons/scores/safety")
async def get_safety_neuron_scores(limit: int = Query(1000)):
    data = load_json_file("safety_all_neurons_scores.json")
    if not data:
        return {"metadata": {}, "neurons": []}
    all_neurons = data.get("all_neurons", {})
    neuron_list = [{"key": k, **v} for k, v in all_neurons.items()]
    neuron_list.sort(key=lambda x: x.get("score", 0), reverse=True)
    return {"metadata": data.get("metadata", {}), "neurons": neuron_list[:limit]}

@app.get("/api/neurons/scores/utility")
async def get_utility_neuron_scores(limit: int = Query(1000)):
    data = load_json_file("utility_all_neurons_scores.json")
    if not data:
        return {"metadata": {}, "neurons": []}
    all_neurons = data.get("all_neurons", {})
    neuron_list = [{"key": k, **v} for k, v in all_neurons.items()]
    neuron_list.sort(key=lambda x: x.get("score", 0), reverse=True)
    return {"metadata": data.get("metadata", {}), "neurons": neuron_list[:limit]}

@app.get("/api/neurons/scores/combined")
async def get_combined_neuron_scores(limit: int = Query(1000)):
    safety_data = load_json_file("safety_all_neurons_scores.json")
    utility_data = load_json_file("utility_all_neurons_scores.json")
    return {
        "safety_neurons": [],
        "utility_neurons": [],
        "overlap_neurons": [],
        "metadata": {
            "num_safety": len(safety_data.get("all_neurons", {})) if safety_data else 0,
            "num_utility": len(utility_data.get("all_neurons", {})) if utility_data else 0,
            "num_overlap": 0,
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Toxic Vectors API
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/toxic-vectors/summary")
async def get_toxic_vectors_summary():
    import numpy as np
    npz_path = os.path.join(_get_data_root(), "toxic_vectors", "toxic_vectors.npz")
    if not os.path.exists(npz_path):
        return {"available": False}
    try:
        data = np.load(npz_path)
        return {
            "available": True,
            "keys": list(data.keys()),
            "shapes": {k: data[k].shape for k in data.keys()},
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


# ═════════════════════════════════════════════════════════════════════════════
# Fine-tuning API
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/finetuning/evaluation")
async def get_finetuning_evaluation():
    return load_json_file("tsft_finetuning/evaluation_comparison.json") or {}

@app.get("/api/finetuning/config")
async def get_finetuning_config():
    return load_json_file("tsft_finetuning/config.json") or {}

@app.post("/api/finetune")
async def start_finetune(request: FinetuneRequest):
    task_id = str(_uuid.uuid4())
    return {"task_id": task_id, "status": "pending", "method": request.method}

@app.get("/api/finetune/{task_id}")
async def get_finetune_status(task_id: str):
    return {"task_id": task_id, "status": "unknown"}

@app.delete("/api/finetune/{task_id}")
async def cancel_finetune(task_id: str):
    return {"task_id": task_id, "status": "cancelled"}


# ═════════════════════════════════════════════════════════════════════════════
# Intervention API
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/api/intervene")
async def intervene(request: InterventionRequest):
    return {"original_output": "...", "intervened_output": "...", "guard_score_change": 0}


# ═════════════════════════════════════════════════════════════════════════════
# Data Summary API
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/data/summary")
async def get_data_summary():
    data_root = _get_data_root()
    files = [
        ("assessment/evaluation_results.json", "评估结果"),
        ("probes/", "探针数据"),
        ("representation/", "表征数据"),
        ("toxic_vectors/", "毒性向量"),
        ("layer_evolution/", "层演化"),
        ("quadrant_classification/", "象限分类"),
        ("parameter_alignment/", "参数对齐"),
        ("activation_projection/", "激活投影"),
        ("gradient_dependency/", "梯度依赖"),
        ("pipeline_meta.json", "Pipeline 元信息"),
    ]
    available = {}
    for path, desc in files:
        full = os.path.join(data_root, path)
        available[desc] = os.path.exists(full)
    return {"data_root": data_root, "active_model": _active_model, "available": available}


# ═════════════════════════════════════════════════════════════════════════════
# Panel F/G/H APIs (保留原有)
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/layer_similarity")
async def get_layer_similarity():
    import numpy as np
    npz_path = os.path.join(_get_data_root(), "probes", "hidden_states_cache.npz")
    if not os.path.exists(npz_path):
        # Fallback: use layer evolution data
        evo = load_json_file("layer_evolution/streamgraph_data.json")
        if evo:
            num_layers = len(evo)
            matrix = []
            for i in range(num_layers):
                row = []
                for j in range(num_layers):
                    # Simple similarity based on activation norm difference
                    si = abs(evo[i]["success"]["mean"] - evo[i]["fail"]["mean"])
                    sj = abs(evo[j]["success"]["mean"] - evo[j]["fail"]["mean"])
                    sim = 1.0 - min(abs(si - sj) / max(si + sj, 1e-8), 1.0)
                    row.append(round(sim * 100, 2))  # Convert to percentage
                matrix.append(row)
            return {"matrix": matrix, "layer_labels": [f"L{i}" for i in range(num_layers)]}
        return {"error": "No layer data available"}
    try:
        data = np.load(npz_path, allow_pickle=True)
        hs = data["train_hs"].astype(np.float32)
        num_layers = hs.shape[1]
        layer_means = hs.mean(axis=0)
        norms = np.linalg.norm(layer_means, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        normed = layer_means / norms
        sim_matrix = (normed @ normed.T).tolist()
        return {"matrix": sim_matrix, "layer_labels": [f"L{i}" for i in range(num_layers)]}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/attack_paths")
async def get_attack_paths():
    """Get attack path data for Panel G (Sankey diagram)"""
    eval_results = load_json_file("assessment/evaluation_results.json")
    if not eval_results or "results" not in eval_results:
        return {"nodes": [], "links": []}

    results = eval_results["results"]
    # Build Sankey: Attack Method -> Category -> Jailbreak Success/Fail
    method_cat = {}
    cat_outcome = {}
    for r in results:
        method = r.get("method", "unknown")
        cat = r.get("category", "unknown")
        outcome = "success" if r.get("jailbreak_success") else "fail"

        key_mc = (method, cat)
        method_cat[key_mc] = method_cat.get(key_mc, 0) + 1
        key_co = (cat, outcome)
        cat_outcome[key_co] = cat_outcome.get(key_co, 0) + 1

    nodes = []
    node_idx = {}
    def get_idx(name):
        if name not in node_idx:
            node_idx[name] = len(nodes)
            nodes.append({"id": name, "label": name, "type": "attack" if name in [m for m, _ in method_cat] else ("output" if "Jailbreak" in name else "category")})
        return node_idx[name]

    links = []
    for (method, cat), count in method_cat.items():
        get_idx(method)
        get_idx(cat)
        links.append({"source": method, "target": cat, "value": count})
    for (cat, outcome), count in cat_outcome.items():
        label = f"Jailbreak ({outcome})"
        get_idx(label)
        links.append({"source": cat, "target": label, "value": count})

    return {"nodes": nodes, "links": links}


@app.get("/api/neuron_activations")
async def get_neuron_activations(successful: bool = True, failed: bool = True):
    """Get neuron activation data for Panel H (violin/box plot)"""
    snip_path = os.path.join(_get_data_root(), "snip_scores", "snip_scores.json")
    quad_path = os.path.join(_get_data_root(), "quadrant_classification.json")

    # Load quadrant classification if available
    quad_data = {}
    if os.path.exists(quad_path):
        try:
            with open(quad_path, "r", encoding="utf-8") as f:
                quad_data = json.load(f)
        except Exception:
            pass

    if not os.path.exists(snip_path):
        return {"S+A+": {"successful": [], "failed": []},
                "S-A-": {"successful": [], "failed": []},
                "S+A-": {"successful": [], "failed": []},
                "S-A+": {"successful": [], "failed": []}}

    with open(snip_path, "r", encoding="utf-8") as f:
        snip = json.load(f)

    # Group by quadrant
    from collections import defaultdict
    quadrant_scores = defaultdict(lambda: {"successful": [], "failed": []})

    for key, info in snip.items():
        layer = info.get("layer_idx", 0)
        neuron = info.get("neuron_idx", 0)
        score = info.get("snip_score", 0)

        # Determine quadrant from classification
        neuron_key = f"layer_{layer}_neuron_{neuron}"
        quad = quad_data.get(neuron_key, {}).get("quadrant", "S-A-")

        # Randomly assign to successful/failed based on score (approximation)
        if score > 0.001:
            quadrant_scores[quad]["successful"].append(round(score, 6))
        else:
            quadrant_scores[quad]["failed"].append(round(score, 6))

    # Ensure all quadrants exist
    for q in ["S+A+", "S-A-", "S+A-", "S-A+"]:
        if q not in quadrant_scores:
            quadrant_scores[q] = {"successful": [], "failed": []}

    return dict(quadrant_scores)


# ═════════════════════════════════════════════════════════════════════════════
# Startup
# ═════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    print(f"[NeuroLens] v{NEUROLENS_VERSION}")
    print(f"[NeuroLens] Project root: {_PROJECT_ROOT}")
    print(f"[NeuroLens] Models dir: {MODELS_DIR}")
    print(f"[NeuroLens] Outputs dir: {OUTPUTS_DIR}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
