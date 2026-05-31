"""
NeuroLens desktop launcher.
Starts FastAPI locally and opens it in an embedded native window.
"""

import os
import sys

# PyInstaller 冻结环境：在 torch/transformers 导入前 stub torch._dynamo
if getattr(sys, "frozen", False):
    import types as _types

    def _allow_in_graph(fn=None):
        if fn is None:
            return lambda func: func
        return fn

    def _disable(fn=None, recursive=True, reason=None, wrapping=True):
        if fn is None:
            return lambda func: func
        return fn

    _dynamo = _types.ModuleType("torch._dynamo")
    _dynamo._nl_stub = True
    _dynamo.allow_in_graph = _allow_in_graph
    _dynamo.disable = _disable
    _dynamo.is_compiling = lambda: False
    _dynamo.mark_static_address = lambda tensor: None
    _dynamo.reset = lambda: None
    _dynamo.config = _types.SimpleNamespace(
        cache_size_limit=64,
        capture_scalar_outputs=False,
    )
    _trace_mod = _types.ModuleType("torch._dynamo._trace_wrapped_higher_order_op")

    class _TransformGetItemToIndex:
        def __init__(self, *args, **kwargs):
            pass

    _trace_mod.TransformGetItemToIndex = _TransformGetItemToIndex
    sys.modules["torch._dynamo"] = _dynamo
    sys.modules["torch._dynamo._trace_wrapped_higher_order_op"] = _trace_mod

# 关键：添加 _internal 目录到 sys.path 以便导入 main.py
if getattr(sys, 'frozen', False):
    # Data files are in _internal subdirectory of the executable
    executable_dir = os.path.dirname(sys.executable)
    internal_dir = os.path.join(executable_dir, "_internal")
    if internal_dir not in sys.path:
        sys.path.insert(0, internal_dir)

import socket
import threading
import time
import urllib.request
import traceback
from typing import Optional
_SERVER_ERROR: Optional[str] = None
_WEBVIEW_IMPORT_ERROR: Optional[str] = None


# 打印启动信息用于调试
print(f"[NeuroLens] Python: {sys.version}")
print(f"[NeuroLens] Frozen: {getattr(sys, 'frozen', False)}")
if getattr(sys, 'frozen', False):
    print(f"[NeuroLens] _MEIPASS: {getattr(sys, '_MEIPASS', 'N/A')}")
    print(f"[NeuroLens] executable: {sys.executable}")
    print(f"[NeuroLens] cwd: {os.getcwd()}")

try:
    import uvicorn
    print("[NeuroLens] uvicorn imported OK")
except Exception as e:
    print(f"[NeuroLens] uvicorn import FAILED: {e}")
    traceback.print_exc()

webview = None
try:
    import webview
    print("[NeuroLens] webview imported OK")
except Exception as e:
    _WEBVIEW_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    print(f"[NeuroLens] webview import FAILED: {e}")
    traceback.print_exc()

try:
    from main import NEUROLENS_VERSION, app
    print(f"[NeuroLens] main module imported OK, version: {NEUROLENS_VERSION}")
except Exception as e:
    print(f"[NeuroLens] main module import FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


def _find_free_port(host: str, preferred_port: int) -> int:
    """Return preferred port if available, otherwise a random free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        if sock.connect_ex((host, preferred_port)) != 0:
            return preferred_port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _wait_server_ready(url: str, timeout_seconds: float = 15.0) -> bool:
    """Poll health endpoint until ready or timeout."""
    start = time.time()
    while time.time() - start < timeout_seconds:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def _resolve_frontend_url(root_url: str, dashboard_url: str) -> str:
    """Always open / (home page); /health already confirmed the server is up."""
    return root_url


def _run_server(host: str, port: int) -> None:
    global _SERVER_ERROR
    print(f"[NeuroLens] Starting server at {host}:{port}")
    try:
        # PyInstaller windowed/frozen runtime may break uvicorn's default logging formatter.
        # Disable uvicorn log dictConfig here to avoid formatter initialization failure.
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            log_config=None,
            access_log=False,
        )
        server = uvicorn.Server(config)
        print("[NeuroLens] Server running, calling server.run()")
        server.run()
        print("[NeuroLens] Server stopped")
    except Exception as e:
        _SERVER_ERROR = f"{type(e).__name__}: {e}"
        print(f"[NeuroLens] Server error: {e}")
        traceback.print_exc()


def main() -> None:
    global _SERVER_ERROR
    _SERVER_ERROR = None
    print("[NeuroLens] main() started")
    host = "127.0.0.1"
    preferred_port = int(os.getenv("NEUROLENS_PORT", "6008"))
    startup_timeout = float(os.getenv("NEUROLENS_STARTUP_TIMEOUT", "45"))
    port = _find_free_port(host, preferred_port)
    print(f"[NeuroLens] Using port: {port}")

    server_thread = threading.Thread(target=_run_server, args=(host, port), daemon=True)
    server_thread.start()

    health_url = f"http://{host}:{port}/health"
    root_url = f"http://{host}:{port}/"
    dashboard_url = f"http://{host}:{port}/dashboard"
    print(f"[NeuroLens] Waiting for server at {health_url}")

    if not _wait_server_ready(health_url, timeout_seconds=startup_timeout):
        print("[NeuroLens] ERROR: Server failed to start")
        if _SERVER_ERROR:
            raise RuntimeError(f"NeuroLens backend failed to start: {_SERVER_ERROR}")
        if not server_thread.is_alive():
            raise RuntimeError("NeuroLens backend failed to start: server thread exited early.")
        raise RuntimeError(
            f"NeuroLens backend failed to start within {startup_timeout:.0f}s. "
            "Set NEUROLENS_STARTUP_TIMEOUT to a larger value if needed."
        )

    if webview is None:
        raise RuntimeError(
            "pywebview is not available in packaged runtime. "
            f"Import error: {_WEBVIEW_IMPORT_ERROR or 'unknown'}"
        )

    app_url = _resolve_frontend_url(root_url, dashboard_url)
    print(f"[NeuroLens] Server ready, opening window with {app_url}")
    window = webview.create_window(
        title=f"NeuroLens Visualization v{NEUROLENS_VERSION}",
        url=app_url,
        width=1400,
        height=900,
        min_size=(1100, 720),
    )
    webview.start()
    if window is None:
        raise RuntimeError("Window initialization failed.")


if __name__ == "__main__":
    main()
