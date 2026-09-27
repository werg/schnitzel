"""SCHNITZELJAGD latent memory system. No downloads on import."""
from __future__ import annotations

__version__ = "0.4.0"
__all__ = ["SchnitzelAgent", "SDKBAgent", "__version__"]


def __getattr__(name: str):
    if name in {"SchnitzelAgent", "SDKBAgent"}:
        from .agent import SchnitzelAgent
        return SchnitzelAgent
    raise AttributeError(name)
