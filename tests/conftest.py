"""Expose the integration's Home-Assistant-free modules as the package `ef`.

Importing `custom_components.entityfuture` would execute the integration's
real `__init__.py`, which needs a full Home Assistant install. The modules
tested here (const, model, metrics, features, backtest) never import Home
Assistant, so we register the directory under a stub package name instead.
"""
from __future__ import annotations

from pathlib import Path
import sys
import types

_DIR = Path(__file__).resolve().parents[1] / "custom_components" / "entityfuture"

if "ef" not in sys.modules:
    package = types.ModuleType("ef")
    package.__path__ = [str(_DIR)]
    sys.modules["ef"] = package
