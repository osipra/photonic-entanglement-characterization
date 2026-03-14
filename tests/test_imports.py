"""Smoke tests for scaffolded package imports."""

from __future__ import annotations

from importlib import import_module

import pytest

MODULE_NAMES = [
    "states",
    "metrics",
    "mle",
    "tomography",
    "bell",
    "chsh",
    "waveplates",
    "io",
    "plotting",
]


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_module_imports(module_name: str) -> None:
    """Each public module should import successfully."""
    module = import_module(f"pec.{module_name}")

    assert module.__doc__

