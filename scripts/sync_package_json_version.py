"""将 visualization/frontend/package.json 的 version 与仓库根 VERSION 首行对齐。"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    vf = root / "VERSION"
    if not vf.is_file():
        print(f"缺少 {vf}", file=sys.stderr)
        return 1
    ver = (vf.read_text(encoding="utf-8").strip().splitlines() or [""])[0].strip()
    if not ver:
        print("VERSION 首行为空", file=sys.stderr)
        return 1
    pkg_path = root / "visualization" / "frontend" / "package.json"
    data = json.loads(pkg_path.read_text(encoding="utf-8"))
    if data.get("version") == ver:
        return 0
    data["version"] = ver
    pkg_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"已更新 {pkg_path.name} -> {ver}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
