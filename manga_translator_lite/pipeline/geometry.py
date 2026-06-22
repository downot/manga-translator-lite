"""Pure box-geometry helpers shared by the extract pipeline.

Kept dependency-free (typing only — no cv2/numpy/torch) so they can be unit-tested
standalone without pulling in the ML stack. These back three documented behaviours:

  - ``_iou_xyxy``   → detector fusion's ``fusion_iou`` ("is this secondary box new?")
  - ``_overlap_min``→ detector fusion's ``fusion_overlap_limit`` (containment dedup)
  - ``_compute_iou``→ ``extract --overwrite`` translation salvage (spatial matching)
"""
from __future__ import annotations

from typing import List, Sequence


def _iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two axis-aligned (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _overlap_min(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over the *smaller* box's area — a containment score.

    Unlike IoU, this stays high when one box sits inside a much larger one
    (e.g. a small primary text line fully covered by a large box-detector
    region), which is exactly the duplicate case IoU misses.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    m = min(area_a, area_b)
    return inter / m if m > 0 else 0.0


def _compute_iou(box1: List[int], box2: List[int]) -> float:
    """IoU of two (x, y, w, h) boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area
