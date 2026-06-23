"""Path-confinement tests for the workspace schema — the authoritative guard that
keeps a tampered/CLI-written pages.json from reading or writing outside the task.

Loads schema.py standalone (stdlib-only imports) so this runs without torch/cv2."""

import importlib.util
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_schema():
    p = REPO / "manga_translator_lite" / "pipeline" / "schema.py"
    spec = importlib.util.spec_from_file_location("mtl_schema_standalone", p)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: the @dataclass fields use string annotations
    # (from __future__ import annotations), which dataclasses resolves via
    # sys.modules[cls.__module__] during class creation.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


schema = _load_schema()


# ---- safe_workspace_path: confines page.clean reads -------------------------
def test_safe_workspace_path_allows_inside(tmp_path):
    got = schema.safe_workspace_path(str(tmp_path), "clean/0001.png")
    assert got is not None and got.endswith("clean/0001.png")


def test_safe_workspace_path_rejects_traversal(tmp_path):
    assert schema.safe_workspace_path(str(tmp_path), "../../etc/passwd") is None
    assert schema.safe_workspace_path(str(tmp_path), "clean/../../x.png") is None


def test_safe_workspace_path_rejects_absolute(tmp_path):
    assert schema.safe_workspace_path(str(tmp_path), "/etc/passwd") is None


def test_safe_workspace_path_rejects_empty(tmp_path):
    assert schema.safe_workspace_path(str(tmp_path), "") is None


# ---- Page.from_dict: name/original reduced to a basename --------------------
def test_page_from_dict_basenames_name_and_original():
    p = schema.Page.from_dict({
        "index": 0,
        "name": "../../escape.png",
        "original": "/abs/path/orig.png",
        "size": [10, 10],
        "clean": "clean/0001.png",
        "blocks": [],
    })
    assert p.name == "escape.png"
    assert p.original == "orig.png"
    # clean is a relative read path — kept intact (confined later at render time).
    assert p.clean == "clean/0001.png"
