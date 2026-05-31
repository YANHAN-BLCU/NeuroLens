"""PyInstaller runtime hooks for torch/transformers/scipy in frozen builds."""

import importlib.util
import sys
import types


def _install_torch_dynamo_stub() -> None:
    if not getattr(sys, "frozen", False):
        return

    existing = sys.modules.get("torch._dynamo")
    if existing is not None and getattr(existing, "_nl_stub", False):
        return

    def allow_in_graph(fn=None):
        if fn is None:
            return lambda func: func
        return fn

    def disable(fn=None, recursive=True, reason=None, wrapping=True):
        if fn is None:
            return lambda func: func
        return fn

    dynamo = types.ModuleType("torch._dynamo")
    dynamo._nl_stub = True
    dynamo.allow_in_graph = allow_in_graph
    dynamo.disable = disable
    dynamo.is_compiling = lambda: False
    dynamo.mark_static_address = lambda tensor: None
    dynamo.reset = lambda: None
    dynamo.config = types.SimpleNamespace(
        cache_size_limit=64,
        capture_scalar_outputs=False,
    )

    trace_mod = types.ModuleType("torch._dynamo._trace_wrapped_higher_order_op")

    class TransformGetItemToIndex:
        def __init__(self, *args, **kwargs):
            pass

    trace_mod.TransformGetItemToIndex = TransformGetItemToIndex

    sys.modules["torch._dynamo"] = dynamo
    sys.modules["torch._dynamo._trace_wrapped_higher_order_op"] = trace_mod

    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        torch_mod._dynamo = dynamo


class _ScipyDistnInfrastructureLoader:
    """Fix scipy `del obj` NameError under PyInstaller frozen importer."""

    def find_spec(self, fullname, path, target=None):
        if fullname != "scipy.stats._distn_infrastructure":
            return None
        if fullname in sys.modules:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        import os
        from pathlib import Path

        patch_candidates = [
            Path(getattr(sys, "_MEIPASS", "")) / "frozen_patches" / "scipy_stats_distn_infrastructure.py",
            Path(os.path.dirname(sys.executable)) / "_internal" / "frozen_patches" / "scipy_stats_distn_infrastructure.py",
        ]
        source = None
        for candidate in patch_candidates:
            if candidate.is_file():
                source = candidate.read_text(encoding="utf-8")
                break
        if source is None:
            raise ImportError("无法定位 patched scipy.stats._distn_infrastructure 源文件")

        exec(compile(source, module.__name__, "exec"), module.__dict__)


def _install_scipy_distn_patch() -> None:
    if not getattr(sys, "frozen", False):
        return
    if any(isinstance(hook, _ScipyDistnInfrastructureLoader) for hook in sys.meta_path):
        return
    sys.meta_path.insert(0, _ScipyDistnInfrastructureLoader())


_install_torch_dynamo_stub()
_install_scipy_distn_patch()
