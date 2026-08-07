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


def test_page_preserves_chapter_start_marker():
    p = schema.Page.from_dict({
        "index": 1,
        "name": "001.png",
        "original": "001.png",
        "size": [10, 10],
        "clean": "clean/0001.png",
        "blocks": [],
        "chapter_start": True,
        "chapter_name": "CH1",
    })

    assert p.chapter_start is True
    assert p.chapter_name == "CH1"
    assert p.to_dict()["chapter_start"] is True
    assert p.to_dict()["chapter_name"] == "CH1"


def test_legacy_page_schema_round_trips_without_new_fields():
    """A pre-v4-style page remains valid after extract behaviour changes.

    New extraction still writes the established blocks/erase_regions shapes; this
    guards the reader used by render/editor against requiring any new field.
    """
    legacy = {
        "index": 2,
        "name": "0003.jpg",
        "original": "0003.jpg",
        "size": [1200, 1800],
        "clean": "clean/0002_0003.jpg",
        "blocks": [{
            "id": "p0002_b000",
            "text": "old OCR",
            "bbox": [10, 20, 30, 40],
            "polygon": [[10, 20], [40, 20], [40, 60], [10, 60]],
            "lines": [[[10, 20], [40, 20], [40, 60], [10, 60]]],
        }],
        "erase_regions": [[[50, 20], [80, 20], [80, 60], [50, 60]]],
    }
    page = schema.Page.from_dict(legacy)

    assert page.clean == "clean/0002_0003.jpg"  # old JPEG intermediates remain readable
    assert page.blocks[0].ocr_text == "old OCR"  # old files lacked ocr_text
    assert page.to_dict()["erase_regions"] == legacy["erase_regions"]


def test_optional_translation_semantics_and_provenance_remain_backward_compatible():
    block = schema.Block.from_dict({
        "id": "p0000_b000", "text": "x", "bbox": [0, 0, 1, 1],
        "polygon": [], "lines": [],
    })
    old_translation = schema.Translation.from_dict({"text": "y", "edited": False})

    assert block.kind == "auto" and block.article_id == ""
    assert old_translation.source_hash == "" and old_translation.profile == ""
    assert schema.Translation(text="y", source_hash="hash", profile="magazine").to_dict()["profile"] == "magazine"


def test_layout_overrides_apply_per_language(tmp_path):
    ws = schema.Workspace(root=str(tmp_path), target_lang="ENG", pages=[
        schema.Page(
            index=0,
            name="0001.png",
            original="0001.png",
            size=(100, 100),
            clean="clean/0001.png",
            blocks=[schema.Block(
                id="p0000_b000",
                text="x",
                bbox=[0, 0, 10, 10],
                polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
                lines=[],
            )],
        )
    ])
    schema.save_layout_overrides(str(tmp_path), "ENG", {
        "p0000_b000": {
            "bbox": [1, 2, 30, 40],
            "fixed_region": True,
            "direction": "h",
        }
    })

    assert schema.apply_layout_overrides(ws, "ENG") == 1
    assert ws.pages[0].blocks[0].bbox == [1, 2, 30, 40]
    assert ws.pages[0].blocks[0].fixed_region is True
    assert ws.pages[0].blocks[0].direction == "h"


def test_render_report_and_glossary_helpers_round_trip(tmp_path):
    report = {"version": 1, "blocks": {"p0000_b000": {"overflow": True}}}
    path = schema.save_render_report(str(tmp_path), report)

    assert path.endswith("render_report.json")
    assert schema.load_render_report(str(tmp_path)) == report

    glossary_path = tmp_path / "glossary.json"
    glossary_path.write_text('{"先輩":{"rule":"fixed","CHS":"学长"}}', encoding="utf-8")
    assert schema.load_glossary(str(tmp_path))["先輩"]["CHS"] == "学长"
