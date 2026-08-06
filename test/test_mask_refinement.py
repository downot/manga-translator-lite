"""Smoke tests for dependency-light mask refinement rules.

Loads the rule module standalone so core smoke tests do not import the package
initializer or the OpenCV/CRF-heavy mask refinement pipeline.
"""

import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_component_rules():
    p = REPO / "manga_translator_lite" / "mask_refinement" / "component_rules.py"
    spec = importlib.util.spec_from_file_location("mtl_component_rules_standalone", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rules = _load_component_rules()
_is_tiny_component_inside_line = rules._is_tiny_component_inside_line


def test_tiny_component_inside_text_line_is_kept():
    assert _is_tiny_component_inside_line(4, 0.2, 0.01)


def test_tiny_component_near_but_outside_text_line_is_rejected():
    assert not _is_tiny_component_inside_line(4, 0.0, 0.01)


def test_larger_component_does_not_use_tiny_component_rule():
    assert not _is_tiny_component_inside_line(10, 0.2, 0.01)
