"""从仓库根目录 VERSION 读取语义化版本号（首行）。"""

from __future__ import annotations

import os


def read_version_from_repo_root(repo_root: str | os.PathLike[str]) -> str:
    root = os.fspath(repo_root)
    p = os.path.join(root, "VERSION")
    try:
        with open(p, "r", encoding="utf-8") as f:
            line = (f.readline() or "").strip()
            if line:
                return line
    except OSError:
        pass
    raise RuntimeError(
        f"无法读取版本号：请确保 VERSION 文件存在且首行非空（{p}）。"
    )
