"""脱敏 NER 推理服务：识别姓名(PER) 与详细地址(ADDR)。"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

_ner_lock = threading.Lock()
_model_bundle = None

LABEL_ZH = {"PER": "姓名", "ADDR": "地址"}
KEEP_TAGS = frozenset(LABEL_ZH)


def _install_torch_dynamo_stub() -> None:
    """PyInstaller 冻结环境下真实 torch._dynamo 会触发 NameError。"""
    if not getattr(sys, "frozen", False):
        return

    import types

    existing = sys.modules.get("torch._dynamo")
    if existing is not None and getattr(existing, "_nl_stub", False):
        return

    def _noop_decorator(fn=None, *args, **kwargs):
        if fn is None:
            return lambda func: func
        return fn

    class _DynamoConfig:
        cache_size_limit = 64
        capture_scalar_outputs = False

        def __getattr__(self, name):
            return False

    class _DynamoStub(types.ModuleType):
        _nl_stub = True

        def __init__(self):
            super().__init__("torch._dynamo")
            self.allow_in_graph = _noop_decorator
            self.disable = _noop_decorator
            self.is_compiling = lambda: False
            self.mark_static_address = lambda tensor: None
            self.reset = lambda: None
            self.config = _DynamoConfig()

        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return _noop_decorator

    dynamo = _DynamoStub()

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
    def find_spec(self, fullname, path, target=None):
        if fullname != "scipy.stats._distn_infrastructure":
            return None
        if fullname in sys.modules:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        patch_candidates = [
            Path(__file__).resolve().parents[1] / "frozen_patches" / "scipy_stats_distn_infrastructure.py",
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


def default_model_dir() -> Path:
    env_dir = os.environ.get("NL_DESENS_MODEL_DIR")
    if env_dir:
        return Path(env_dir)

    root = Path(__file__).resolve().parent

    def _resolve_model_dir(model_dir: Path) -> Path | None:
        if not model_dir.is_dir():
            return None
        for weight_name in ("model.safetensors", "pytorch_model.bin"):
            if (model_dir / weight_name).exists():
                return model_dir
        checkpoints = sorted(
            (p for p in model_dir.glob("checkpoint-*") if p.is_dir()),
            key=lambda p: int(p.name.split("-")[-1]),
            reverse=True,
        )
        for ckpt in checkpoints:
            for weight_name in ("model.safetensors", "pytorch_model.bin"):
                if (ckpt / weight_name).exists():
                    return ckpt
        return None

    for name in ("chinese-ner-per-addr-rbt3", "chinese-ner-per-addr"):
        resolved = _resolve_model_dir(root / "models" / name)
        if resolved is not None:
            return resolved
    return root / "models" / "chinese-ner-per-addr-rbt3"


def _resolve_label(id2label: dict, label_id: int) -> str:
    return id2label.get(label_id, id2label.get(str(label_id), "O"))


def _aggregate_bio(text: str, offsets: list[tuple[int, int]], pred_ids: list[int], id2label: dict) -> list[dict]:
    entities: list[dict] = []
    current: dict | None = None

    for label_id, (start, end) in zip(pred_ids, offsets):
        if start == end == 0:
            if current:
                entities.append(current)
                current = None
            continue

        label = _resolve_label(id2label, int(label_id))
        if label == "O":
            if current:
                entities.append(current)
                current = None
            continue

        tag = label.split("-", 1)[-1]
        if tag not in KEEP_TAGS:
            if current:
                entities.append(current)
                current = None
            continue

        if label.startswith("B-") or current is None or current["label"] != tag:
            if current:
                entities.append(current)
            current = {"label": tag, "start": start, "end": end}
        else:
            current["end"] = end

    if current:
        entities.append(current)

    return [
        {
            "start": ent["start"],
            "end": ent["end"],
            "label": ent["label"],
            "type": LABEL_ZH[ent["label"]],
            "text": text[ent["start"] : ent["end"]],
        }
        for ent in entities
    ]


def _load_model_bundle():
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle

    _install_torch_dynamo_stub()
    _install_scipy_distn_patch()

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"导入 torch 失败: {exc}") from exc

    model_dir = default_model_dir()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"NER 模型目录不存在: {model_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
    except Exception as exc:
        raise RuntimeError(f"导入 BertTokenizerFast 失败: {exc}") from exc

    try:
        from transformers.models.bert.modeling_bert import BertForTokenClassification
    except Exception as exc:
        raise RuntimeError(f"导入 BertForTokenClassification 失败: {exc}") from exc

    try:
        tokenizer = BertTokenizerFast.from_pretrained(str(model_dir), local_files_only=True)
        model = BertForTokenClassification.from_pretrained(str(model_dir), local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f"加载 NER 模型失败: {exc}") from exc

    model.to(device)
    model.eval()

    _model_bundle = (tokenizer, model, device)
    return _model_bundle


def predict_entities(text: str) -> list[dict]:
    """对文本运行 NER，返回实体列表。"""
    if not text or not text.strip():
        return []

    import torch

    with _ner_lock:
        tokenizer, model, device = _load_model_bundle()

        encoding = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )
        offset_mapping = encoding.pop("offset_mapping")[0].tolist()
        inputs = {key: value.to(device) for key, value in encoding.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        pred_ids = logits.argmax(-1)[0].cpu().tolist()
        return _aggregate_bio(text, offset_mapping, pred_ids, model.config.id2label)


if getattr(sys, "frozen", False):
    _install_torch_dynamo_stub()
    _install_scipy_distn_patch()
