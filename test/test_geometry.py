"""Unit tests for the pure box-geometry helpers in pipeline/geometry.py — the math
behind detector fusion (fusion_iou / fusion_overlap_limit) and the extract --overwrite
translation salvage. geometry.py is dependency-free, so we load it standalone (like the
config/args smoke tests) without importing extract.py and its ML stack."""

import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_geometry():
    p = REPO / "manga_translator_lite" / "pipeline" / "geometry.py"
    spec = importlib.util.spec_from_file_location("mtl_geometry_standalone", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


geo = _load_geometry()


# --- _iou_xyxy: IoU of (x1, y1, x2, y2) boxes -------------------------------------

def test_iou_xyxy_identical_boxes():
    assert geo._iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_xyxy_disjoint_boxes():
    assert geo._iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_xyxy_edge_touching_is_zero():
    # Boxes sharing only an edge have zero intersection area.
    assert geo._iou_xyxy((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_iou_xyxy_half_overlap():
    # Two 10x10 boxes overlapping in a 5x10 strip: inter=50, union=150.
    assert geo._iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_iou_xyxy_degenerate_zero_area_box():
    assert geo._iou_xyxy((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0


# --- _overlap_min: intersection over the smaller box (containment) ----------------

def test_overlap_min_full_containment_is_one():
    # A small box fully inside a large one → intersection == smaller box's area.
    # This is the duplicate case IoU misses (here IoU would be only 0.04).
    big, small = (0, 0, 100, 100), (10, 10, 20, 20)
    assert geo._overlap_min(big, small) == pytest.approx(1.0)
    assert geo._iou_xyxy(big, small) == pytest.approx((10 * 10) / (100 * 100))


def test_overlap_min_disjoint_is_zero():
    assert geo._overlap_min((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_overlap_min_partial_over_smaller():
    # inter = 5x10 = 50; smaller box area = 10x10 = 100 → 0.5.
    assert geo._overlap_min((0, 0, 10, 10), (5, 0, 25, 10)) == pytest.approx(0.5)


# --- _compute_iou: IoU of (x, y, w, h) boxes (overwrite salvage matching) ----------

def test_compute_iou_identical_boxes():
    assert geo._compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_compute_iou_disjoint_boxes():
    assert geo._compute_iou([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0


def test_compute_iou_quarter_overlap():
    # (x,y,w,h): overlap is a 5x5 = 25 square; union = 100 + 100 - 25 = 175.
    assert geo._compute_iou([0, 0, 10, 10], [5, 5, 10, 10]) == pytest.approx(25 / 175)


def test_compute_iou_zero_area_boxes_no_zerodivision():
    # union_area == 0 must be handled, not raise ZeroDivisionError.
    assert geo._compute_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0


def test_match_boxes_by_iou_is_one_to_one_after_a_block_splits():
    matches = geo.match_boxes_by_iou(
        [[0, 0, 10, 10], [0, 0, 9, 9]],
        [[0, 0, 10, 10]],
    )

    assert matches == [(0, 0)]


def test_match_boxes_by_iou_keeps_distinct_matches():
    matches = geo.match_boxes_by_iou(
        [[0, 0, 10, 10], [20, 0, 10, 10]],
        [[0, 0, 10, 10], [20, 0, 10, 10]],
    )

    assert matches == [(0, 0), (1, 1)]
