"""
NeuroBreak Engine Package
"""

from pathlib import Path

from engine._repo_version import read_version_from_repo_root

__version__ = read_version_from_repo_root(Path(__file__).resolve().parent.parent)
