"""Compatibility package for code that still imports the former ``sdkb`` name.

``sdkb.X`` resolves to the very module object ``schnitz.X`` (not a second copy of
the same file), so classes, module state and monkeypatches are shared."""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

import schnitz as _schnitz

__version__ = _schnitz.__version__
__all__ = ["SchnitzelAgent", "SDKBAgent", "__version__"]
__path__: list[str] = []  # submodules come only from the alias finder below


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self):
        self._specs = {}

    def find_spec(self, name, path=None, target=None):
        if name.startswith("sdkb."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        module = importlib.import_module("schnitz" + spec.name[len("sdkb"):])
        self._specs[spec.name] = module.__spec__
        return module

    def exec_module(self, module):
        # importlib sets ``__spec__`` to the alias spec on the returned module;
        # restore the real one so ``schnitz.X`` keeps its identity
        module.__spec__ = self._specs.pop(module.__name__.replace("schnitz", "sdkb", 1),
                                          module.__spec__)


if not any(isinstance(finder, _Alias) for finder in sys.meta_path):
    sys.meta_path.insert(0, _Alias())


def __getattr__(name: str):
    if name in {"SchnitzelAgent", "SDKBAgent"}:
        from schnitz.agent import SchnitzelAgent
        return SchnitzelAgent
    raise AttributeError(name)
