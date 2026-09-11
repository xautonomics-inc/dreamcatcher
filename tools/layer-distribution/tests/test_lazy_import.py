"""Test lazy importing of slicer and zero-dependency core modules."""

from __future__ import annotations

import sys

import pytest


def test_lazy_slicer_export_without_numpy_or_gguf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify layer_distribution imports without numpy or gguf installed."""
    # Unload layer_distribution and dependents if loaded
    for mod in list(sys.modules):
        if mod.startswith("layer_distribution"):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    # Block numpy and gguf imports
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setitem(sys.modules, "gguf", None)

    import layer_distribution

    assert callable(layer_distribution.verify)
    assert callable(layer_distribution.precheck)
    assert callable(layer_distribution.files_for_window)
    assert callable(layer_distribution.rebalance)
    assert "slice_model" in dir(layer_distribution)

    if "plan_stages" in dir(layer_distribution):
        plan_func = getattr(layer_distribution, "plan_stages")  # noqa: B009
        assert callable(plan_func)

    # Accessing slice_model when numpy/gguf is blocked fails with ImportError
    with pytest.raises((ImportError, ModuleNotFoundError)):
        _ = layer_distribution.slice_model


def test_lazy_getattr_unknown_attribute() -> None:
    """Verify AttributeError on unknown attribute."""
    import layer_distribution

    with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
        _ = layer_distribution.nonexistent
