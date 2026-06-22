"""Unit tests for the cross-language reference resolution documented in the READMEs:
``--no-reference`` → off ([]), one or more ``--reference-lang`` → exactly those,
neither → fall back to config ``[translator] reference_langs`` (default None = auto).
CLI always overrides config.

``_resolve_reference_langs`` lives in ``__main__.py``, which transitively imports the
heavy ML stack via ``.utils`` (torch/cv2 — not installed in CI). So, exactly like the
other smoke tests, we load the module from source with the three relative imports it
needs stubbed out; the function under test only reads duck-typed attributes."""

import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_main_with_stubs():
    pkg_name = "mtl_main_stub"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = []  # mark as a package so relative imports resolve
    sys.modules[pkg_name] = pkg
    # __main__.py does: from .args import build_parser / from .config import Config /
    # from .utils import get_logger, init_logging, set_log_level — stub them all.
    stubs = {
        "args": ["build_parser"],
        "config": ["Config"],
        "utils": ["get_logger", "init_logging", "set_log_level"],
    }
    for sub_name, attrs in stubs.items():
        sub = types.ModuleType(f"{pkg_name}.{sub_name}")
        for attr in attrs:
            setattr(sub, attr, lambda *a, **k: None)
        sys.modules[f"{pkg_name}.{sub_name}"] = sub
        setattr(pkg, sub_name, sub)

    src = REPO / "manga_translator_lite" / "__main__.py"
    spec = importlib.util.spec_from_file_location(f"{pkg_name}.__main__", src)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    spec.loader.exec_module(mod)
    return mod


resolve = _load_main_with_stubs()._resolve_reference_langs


def _args(no_reference=False, reference_lang=None):
    return SimpleNamespace(no_reference=no_reference, reference_lang=reference_lang)


def _cfg(reference_langs):
    return SimpleNamespace(translator=SimpleNamespace(reference_langs=reference_langs))


def test_no_reference_flag_forces_off():
    # --no-reference wins even when config has codes → explicit off ([]).
    assert resolve(_args(no_reference=True), _cfg(["CHS"])) == []


def test_explicit_reference_langs_override_config():
    # One or more --reference-lang → exactly those, ignoring config.
    assert resolve(_args(reference_lang=["CHS", "KOR"]), _cfg(None)) == ["CHS", "KOR"]


def test_no_reference_beats_explicit_reference_langs():
    # Both flags present: --no-reference takes precedence (off).
    assert resolve(_args(no_reference=True, reference_lang=["CHS"]), _cfg(None)) == []


@pytest.mark.parametrize("config_value", [None, [], ["JPN"]])
def test_falls_back_to_config_when_no_cli_flags(config_value):
    # No CLI flags → config value passes through verbatim:
    # None = auto, [] = off, [codes] = manual.
    assert resolve(_args(), _cfg(config_value)) == config_value
