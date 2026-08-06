"""Step 1: detection + OCR + inpainting → workspace files.

Walks the input path, runs the local CV pipeline on each image, and writes:

  - workspace/<task>/clean/<idx>_<name>.png    text-removed image
  - workspace/<task>/pages.json                metadata + OCR text per block

When the input directory contains subdirectories, each subdirectory is treated
as a separate *task*.  When the input is a flat directory of images (no sub-
directories), a single task named after the directory basename is created.

Translation is left blank for the translate step to fill in.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from ..config import Config, Detector, Inpainter
from ..detection import dispatch as dispatch_detection, prepare as prepare_detection
from ..inpainting import dispatch as dispatch_inpainting, prepare as prepare_inpainting
from ..mask_refinement import dispatch as dispatch_mask_refinement
from ..ocr import dispatch as dispatch_ocr, prepare as prepare_ocr
from ..textline_merge import dispatch as dispatch_textline_merge
from .geometry import _compute_iou, _iou_xyxy, _overlap_min
from ..utils import (
    TextBlock,
    get_logger,
    is_valuable_text,
    load_image,
    natural_sort,
    sort_regions,
)
from .schema import (
    Block, Page, Workspace, block_id, save_workspace, load_workspace,
    load_translations, save_translations, get_translations_dir
)

logger = get_logger('extract')

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def _select_device(use_gpu: bool) -> str:
    if not use_gpu:
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _free_vram(device: str) -> None:
    """Return the caching allocator's unused blocks to the driver.

    Detection (ctd at detection_size, e.g. 2560) and inpainting (lama at
    inpainting_size, e.g. 2048) run sequentially but their big activation tensors
    are sized very differently. PyTorch's caching allocator keeps the large
    detection-sized blocks "reserved" after detection frees them, so when lama then
    allocates differently-sized blocks the reserved pool grows / fragments and the
    observed peak VRAM (and OOM risk) balloons. Emptying the cache between the two
    heavy stages lets each allocate fresh, which keeps high-recall settings
    (detection_size 2560 + a secondary detector) within budget instead of OOM-ing.
    """
    if device == 'cuda':
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _effective_detection_size(cfg: Config, img_rgb: np.ndarray) -> int:
    """Resolve the detector input size for THIS page.

    A fixed positive ``detection_size`` is used verbatim. ``-1`` switches to AUTO:
    size the page from its own longest side × ``detection_size_scale``, snapped to a
    multiple of 64 (the detector's stride) and clamped to [min, max]. This adapts per
    page — small pages stay fast, large pages keep detail — instead of one global value.
    """
    d = cfg.detector
    ds = d.detection_size
    if ds and ds > 0:
        return ds
    longest = int(max(img_rgb.shape[0], img_rgb.shape[1]))
    scale = d.detection_size_scale if d.detection_size_scale and d.detection_size_scale > 0 else 1.0
    lo = max(64, int(getattr(d, "detection_size_min", 1024) or 1024))
    hi = max(lo, int(getattr(d, "detection_size_max", 2560) or 2560))
    val = int(round(longest * scale / 64.0)) * 64
    val = max(64, val)
    return max(lo, min(hi, val))


async def _dispatch_det(cfg: Config, detector: Detector, img_rgb: np.ndarray, device: str,
                        verbose: bool, box_threshold: float, detection_size: int = None):
    if detection_size is None:
        detection_size = _effective_detection_size(cfg, img_rgb)
    return await dispatch_detection(
        detector,
        img_rgb,
        detection_size,
        cfg.detector.text_threshold,
        box_threshold,
        cfg.detector.unclip_ratio,
        cfg.detector.det_invert,
        cfg.detector.det_gamma_correct,
        cfg.detector.det_rotate,
        cfg.detector.det_auto_rotate,
        device,
        verbose,
    )


async def _detect_fused(cfg: Config, img_rgb: np.ndarray, device: str, verbose: bool):
    """Run the primary detector and, when configured, fuse in a secondary detector's
    regions that the primary missed (IoU below fusion_iou with every primary region).

    Returns (textlines, mask_raw, mask, secondary_only) where ``secondary_only`` are the
    extra Quadrilaterals contributed by the secondary detector. They have no reliable
    stroke-level mask, so the caller derives a conservative local mask before erasing.
    """
    # Resolve the detection size once for this page (auto-sizes when detection_size = -1)
    # and reuse it for the secondary detector so both run at the same resolution.
    eff_ds = _effective_detection_size(cfg, img_rgb)
    if cfg.detector.detection_size is None or cfg.detector.detection_size <= 0:
        logger.info(f"detection_size=auto → {eff_ds} (page {img_rgb.shape[1]}x{img_rgb.shape[0]}, "
                    f"scale={cfg.detector.detection_size_scale})")

    textlines, mask_raw, mask = await _dispatch_det(
        cfg, cfg.detector.detector, img_rgb, device, verbose, cfg.detector.box_threshold, eff_ds)
    # OCR backends are allowed to mutate (or, when they merge lines, replace)
    # Quadrilateral objects. Preserve the detector-stage facts on each original
    # object before OCR so later steps never need object identity or OCR .prob.
    for det_index, line in enumerate(textlines):
        line.source_det_indices = (det_index,)
        line.det_prob = float(line.prob)
        line.is_secondary_only = False

    secondary = cfg.detector.secondary_detector
    secondary_only: List = []
    if secondary != Detector.none and secondary != cfg.detector.detector:
        sec_box_thr = (cfg.detector.secondary_box_threshold
                       if cfg.detector.secondary_box_threshold is not None
                       else cfg.detector.box_threshold)
        sec_textlines, _, _ = await _dispatch_det(
            cfg, secondary, img_rgb, device, verbose, sec_box_thr, eff_ds)
        primary_boxes = [tl.xyxy for tl in textlines]
        thr = cfg.detector.fusion_iou
        overlap_limit = cfg.detector.fusion_overlap_limit
        page_area = float(img_rgb.shape[0] * img_rgb.shape[1])
        max_area = cfg.detector.fusion_max_area_ratio * page_area
        oversize = 0
        dup = 0
        for s in sec_textlines:
            x1, y1, x2, y2 = s.xyxy
            # A box detector can return one huge box for a stylized title / SFX spanning the
            # art; even local-mask fallback is not worth the risk, so drop it.
            if max_area > 0 and (x2 - x1) * (y2 - y1) > max_area:
                oversize += 1
                continue
            # A region is genuinely "new" only if it neither overlaps a primary box (IoU) nor
            # sits on top of / inside one (containment). The containment test catches a large
            # secondary box covering small primary text lines — low IoU but a clear duplicate,
            # which would otherwise be OCR'd again as a partial copy.
            if any(_iou_xyxy(s.xyxy, pb) >= thr or _overlap_min(s.xyxy, pb) >= overlap_limit
                   for pb in primary_boxes):
                dup += 1
                continue
            det_index = len(textlines) + len(secondary_only)
            s.source_det_indices = (det_index,)
            s.det_prob = float(s.prob)
            s.is_secondary_only = True
            secondary_only.append(s)
        textlines = textlines + secondary_only
        logger.info(f"detector fusion: {cfg.detector.detector.value}={len(primary_boxes)} "
                    f"+{len(secondary_only)} new from {secondary.value} "
                    f"(of {len(sec_textlines)}; {dup} overlap, {oversize} oversize dropped)")

    return textlines, mask_raw, mask, secondary_only


def _list_images(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    files: List[str] = []
    for n in natural_sort(os.listdir(input_path)):
        ext = os.path.splitext(n)[1].lower()
        if ext in IMG_EXTS and not n.startswith('.'):
            files.append(os.path.join(input_path, n))
    return files


def _discover_input_tasks(input_path: str) -> List[Tuple[str, str]]:
    """Discover tasks from the input directory.

    Returns a list of (task_name, task_input_dir) tuples.
    - If input_path contains image-bearing subdirectories, each subdirectory
      becomes a separate task.
    - If input_path is a flat directory of images (or a single file), it becomes
      one task named after the directory's basename.
    """
    input_path = os.path.abspath(input_path)

    if os.path.isfile(input_path):
        parent = os.path.dirname(input_path)
        return [(os.path.basename(parent), parent)]

    # Check if input_path has subdirectories with images
    subdirs: List[Tuple[str, str]] = []
    for entry in natural_sort(os.listdir(input_path)):
        full = os.path.join(input_path, entry)
        if os.path.isdir(full) and not entry.startswith('.'):
            # Check if this subdir has images
            if _list_images(full):
                subdirs.append((entry, full))

    if subdirs:
        return subdirs

    # No subdirectories with images — treat as single flat task
    return [(os.path.basename(input_path), input_path)]


def _polygon_from_lines(region: TextBlock) -> List[List[int]]:
    rect = region.min_rect.reshape(-1, 2)
    return [[int(p[0]), int(p[1])] for p in rect[:4]]


def _bbox_xywh(region: TextBlock) -> List[int]:
    x, y, w, h = region.xywh
    return [int(x), int(y), int(w), int(h)]


def _quad_from_textline(textline) -> List[List[int]]:
    """4-point polygon of a detected text line (for erase-only region records)."""
    pts = np.array(textline.pts).reshape(-1, 2)
    return [[int(p[0]), int(p[1])] for p in pts[:4]]


def _mask_blocks_from_lines(textlines: List) -> List[TextBlock]:
    """Make the lightweight geometry objects required by mask refinement.

    The refiner immediately flattens ``TextBlock.lines`` back into individual
    quadrilaterals.  Building full merged text regions here used to repeat the
    O(n²) text-line merge performed for translation, without changing the erase
    mask.  One block per line preserves exactly the geometry the refiner needs.
    """
    return [TextBlock([tl.pts], [tl.text or ''], font_size=max(1, tl.font_size))
            for tl in textlines]


def _source_det_indices(region) -> set[int]:
    """Stable detector-origin indexes carried through OCR and line merging."""
    return set(getattr(region, 'source_det_indices', ()))


def _secondary_stroke_mask(
    img_rgb: np.ndarray,
    raw_mask: Optional[np.ndarray],
    secondary_only: List,
    legacy_box_fill: bool,
) -> Tuple[np.ndarray, set]:
    """Return a conservative erase mask for secondary-detector-only boxes.

    RT-DETR finds useful missed regions but only supplies a rectangle.  Filling
    that rectangle destroys balloon interiors and artwork.  Prefer the primary
    detector's pixel mask inside the box; when it has no signal, fall back to a
    dark-ink Otsu mask only in a light, low-texture balloon-like crop.  Complex
    artwork is intentionally left untouched rather than guessed at.
    """
    h, w = img_rgb.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    erased_source_indices: set[int] = set()
    source_mask = raw_mask
    if source_mask is not None and source_mask.shape[:2] != (h, w):
        source_mask = cv2.resize(source_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    for region in secondary_only:
        x1, y1, x2, y2 = (int(round(v)) for v in region.xyxy)
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        if legacy_box_fill:
            cv2.rectangle(out, (x1, y1), (x2, y2), 255, thickness=-1)
            erased_source_indices.update(_source_det_indices(region))
            continue

        # Keep the boundary out of the candidate area: it is often the balloon
        # outline, not text.  A two-pixel inset is enough even for small labels.
        inset = max(1, min(3, min(x2 - x1, y2 - y1) // 12))
        ix1, iy1, ix2, iy2 = x1 + inset, y1 + inset, x2 - inset, y2 - inset
        if ix2 - ix1 < 2 or iy2 - iy1 < 2:
            continue
        roi_area = (ix2 - ix1) * (iy2 - iy1)
        candidate = np.zeros((iy2 - iy1, ix2 - ix1), dtype=np.uint8)

        if source_mask is not None:
            raw_roi = source_mask[iy1:iy2, ix1:ix2]
            candidate[raw_roi > 0] = 255

        coverage = float(cv2.countNonZero(candidate)) / roi_area
        if coverage < 0.002:
            # The fallback is deliberately limited to bright, low-detail regions.
            # In an illustrated area, thresholding dark ink would also select line art.
            rgb_roi = img_rgb[iy1:iy2, ix1:ix2]
            gray = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2GRAY)
            if float(gray.mean()) >= 170 and float(gray.std()) <= 65:
                _, candidate = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                coverage = float(cv2.countNonZero(candidate)) / roi_area

        # A text mask should be sparse.  Dense masks are usually artwork, a dark
        # balloon, or a detector failure; refusing them is safer than whole-box erase.
        if not (0.002 <= coverage <= 0.35):
            continue
        candidate = cv2.dilate(
            candidate,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        out[iy1:iy2, ix1:ix2] = cv2.bitwise_or(out[iy1:iy2, ix1:ix2], candidate)
        erased_source_indices.update(_source_det_indices(region))

    return out, erased_source_indices


def _save_clean_png(img_rgb: np.ndarray, path: str) -> None:
    """Persist a lossless workspace intermediate without recompressing artwork."""
    Image.fromarray(img_rgb).save(path, format='PNG', compress_level=3)


def _serialise_block(region: TextBlock, page_idx: int, block_idx: int) -> Block:
    fg, bg = region.get_font_colors()
    lines = [
        [[int(p[0]), int(p[1])] for p in line]
        for line in region.lines.tolist()
    ]
    return Block(
        id=block_id(page_idx, block_idx),
        text=region.text,
        ocr_text=region.text,
        bbox=_bbox_xywh(region),
        polygon=_polygon_from_lines(region),
        lines=lines,
        font_size=int(region.font_size),
        angle=float(region.angle),
        fg_color=[int(fg[0]), int(fg[1]), int(fg[2])],
        bg_color=[int(bg[0]), int(bg[1]), int(bg[2])],
        direction=str(region._direction or "auto"),
        alignment=str(region._alignment or "auto"),
        prob=float(region.prob),
    )


async def _process_image(
    img_path: str,
    page_idx: int,
    cfg: Config,
    device: str,
    workspace: Workspace,
    verbose: bool,
) -> Page:
    logger.info(f"[page {page_idx}] {os.path.basename(img_path)}")

    pil = Image.open(img_path).convert('RGB')
    img_rgb, _ = load_image(pil)
    h, w = img_rgb.shape[:2]

    # 1. detection (optionally fused with a secondary detector to boost recall)
    # Free the previous page's inpaint-sized reserved blocks before ctd allocates its
    # (larger) detection_size activations, so peak VRAM stays flat across pages.
    _free_vram(device)
    textlines, mask_raw, mask, secondary_only = await _detect_fused(cfg, img_rgb, device, verbose)

    clean_name = f"{page_idx:04d}_{os.path.splitext(os.path.basename(img_path))[0]}.png"
    clean_rel = f"clean/{clean_name}"
    clean_abs = os.path.join(workspace.clean_dir, clean_name)
    os.makedirs(workspace.clean_dir, exist_ok=True)

    if not textlines:
        logger.info(f"[page {page_idx}] no text detected — marked as no_text, writing lossless copy")
        _save_clean_png(img_rgb, clean_abs)
        return Page(
            index=page_idx,
            name=os.path.basename(img_path),
            size=(w, h),
            original=os.path.basename(img_path),
            clean=clean_rel,
            blocks=[],
            no_text=True,
        )

    # 2. Snapshot the detector-selected primary erase set BEFORE OCR.  OCR backends
    # overwrite ``tl.prob`` with recognition confidence, so reading it afterwards
    # silently mixes two incompatible confidence scales.
    primary_erase_lines = [
        tl for tl in textlines
        if not getattr(tl, 'is_secondary_only', False)
        and tl.det_prob >= cfg.detector.erase_detection_threshold
    ]
    # OCR returns only lines above its own recognition-confidence gate. OCR
    # confidence decides translation; detector confidence only decides erase-only
    # handling for marks that do not become final translation blocks.
    ocr_textlines = await dispatch_ocr(cfg.ocr.ocr, img_rgb, textlines, cfg.ocr, device, verbose)

    min_len = cfg.ocr.min_text_length

    # 3a. Translation set (rules unchanged): valuable OCR → merge → valuable filter → sort.
    valuable_textlines = [tl for tl in ocr_textlines if tl.text and tl.text.strip()]
    if valuable_textlines:
        text_regions = await dispatch_textline_merge(valuable_textlines, w, h, verbose=verbose)
        text_regions = [r for r in text_regions if r.text and is_valuable_text(r.text)
                        and len(r.text) >= min_len]
        text_regions = sort_regions(
            text_regions,
            right_to_left=cfg.render.rtl,
            img=img_rgb,
            force_simple_sort=cfg.force_simple_sort,
        )
    else:
        text_regions = []

    # Provenance is now a safety property rather than presentation metadata: a
    # region with no detector source has no geometry that we can prove was
    # erased. This should not occur with bundled OCR/merge backends, but failing
    # closed prevents an extension backend from rendering over untouched source.
    untraceable_regions = [r for r in text_regions if not _source_det_indices(r)]
    if untraceable_regions:
        logger.warning(
            f"[page {page_idx}] skipped {len(untraceable_regions)} translated region(s) "
            "without detector provenance"
        )
        text_regions = [r for r in text_regions if _source_det_indices(r)]

    # 3b. Secondary-only boxes do not have reliable stroke geometry. Derive local
    # strokes conservatively; whole-box fill is opt-in. A secondary OCR result is
    # allowed to become a Block only when this step has actually erased its source.
    secondary_erased_source_indices: set[int] = set()
    if secondary_only:
        secondary_mask, secondary_erased_source_indices = _secondary_stroke_mask(
            img_rgb, mask_raw, secondary_only, cfg.detector.secondary_box_fill)
        if secondary_mask.any():
            if mask is None:
                mask = secondary_mask
            else:
                mask = cv2.bitwise_or(mask, secondary_mask)

        secondary_source_indices = set().union(
            *(_source_det_indices(tl) for tl in secondary_only)
        ) if secondary_only else set()
        rejected_secondary_sources = secondary_source_indices - secondary_erased_source_indices
        if rejected_secondary_sources:
            dropped = [r for r in text_regions
                       if _source_det_indices(r) & rejected_secondary_sources]
            if dropped:
                logger.warning(
                    f"[page {page_idx}] skipped {len(dropped)} translated secondary region(s) "
                    "because a safe erase mask could not be derived"
                )
                text_regions = [r for r in text_regions
                                if not (_source_det_indices(r) & rejected_secondary_sources)]

    # 3c. The invariant: every final Block must contribute its own geometry to the
    # erase refinement input. Detector-confidence thresholds only govern extra,
    # untranslated primary candidates; they may never exclude a rendered Block.
    translated_source_indices = set().union(
        *(_source_det_indices(region) for region in text_regions)
    ) if text_regions else set()
    primary_erase_only_lines = [
        tl for tl in primary_erase_lines
        if not (_source_det_indices(tl) & translated_source_indices)
    ]

    # 4. Refine final primary Block geometry plus threshold-approved untranslated
    # primary candidates. OR it with any detector/secondary mask rather than
    # replacing those masks, so translated text cannot be omitted by a threshold.
    # ``TextBlock`` does not retain per-line secondary flags. A fused region is
    # normally disjoint; include it here only if it has a primary source. Pure
    # secondary regions are already covered by ``secondary_mask`` above.
    primary_source_indices = {
        index for tl in textlines if not getattr(tl, 'is_secondary_only', False)
        for index in _source_det_indices(tl)
    }
    primary_block_regions = [
        region for region in text_regions
        if _source_det_indices(region) & primary_source_indices
    ]
    erase_blocks = primary_block_regions + _mask_blocks_from_lines(primary_erase_only_lines)
    if erase_blocks:
        refined_mask = await dispatch_mask_refinement(
            erase_blocks,
            img_rgb,
            mask_raw if mask_raw is not None else np.zeros((h, w), dtype=np.uint8),
            'fit_text',
            cfg.mask_dilation_offset,
            cfg.ocr.ignore_bubble,
            verbose,
            cfg.kernel_size,
        )
        mask = refined_mask if mask is None else cv2.bitwise_or(mask, refined_mask)

    erase_only_quads = [
        _quad_from_textline(tl)
        for tl in primary_erase_only_lines
    ]
    erase_only_quads.extend(
        _quad_from_textline(tl)
        for tl in secondary_only
        if _source_det_indices(tl).issubset(secondary_erased_source_indices)
        and not (_source_det_indices(tl) & translated_source_indices)
    )

    # 5. inpainting
    # Release the detection/OCR stage's reserved VRAM before lama allocates its own
    # large activations — this is what keeps detection_size=2560 + secondary detector
    # from inflating peak VRAM / OOM-ing on big pages.
    _free_vram(device)
    if mask is not None and mask.any():
        inpainted = await dispatch_inpainting(
            cfg.inpainter.inpainter,
            img_rgb,
            mask,
            cfg.inpainter,
            cfg.inpainter.inpainting_size,
            device,
            verbose,
        )
    else:
        inpainted = img_rgb

    _save_clean_png(inpainted, clean_abs)

    logger.info(f"[page {page_idx}] saved clean → {clean_rel} "
                f"({len(text_regions)} blocks, {len(erase_only_quads)} erase-only)")

    blocks = [_serialise_block(r, page_idx, i) for i, r in enumerate(text_regions)]
    # no_text only when there is nothing to translate AND nothing was erased.
    return Page(
        index=page_idx,
        name=os.path.basename(img_path),
        size=(w, h),
        original=os.path.basename(img_path),
        clean=clean_rel,
        blocks=blocks,
        erase_regions=erase_only_quads,
        no_text=(not blocks and not erase_only_quads),
    )


def _merge_task_translations(workspace: Workspace, existing_pages: Dict[str, Page]) -> None:
    """Merge existing translations into the newly extracted workspace based on spatial IoU."""
    trans_dir = get_translations_dir(workspace.root)
    if not os.path.isdir(trans_dir):
        return
        
    langs = [f[:-5] for f in os.listdir(trans_dir) if f.endswith('.json')]
    if not langs:
        return
        
    logger.info(f"[task: {workspace.task_name}] Merging translations for: {', '.join(langs)}")
    
    for lang in langs:
        old_trans = load_translations(workspace.root, lang)
        new_trans = {}
        
        for new_p in workspace.pages:
            fname = os.path.basename(new_p.original)
            if fname in existing_pages:
                old_p = existing_pages[fname]
                for new_b in new_p.blocks:
                    # Manually added blocks keep their stable id → carry translation by id.
                    if getattr(new_b, 'user_added', False):
                        if new_b.id in old_trans:
                            new_trans[new_b.id] = old_trans[new_b.id]
                        continue
                    best_iou = 0.0
                    best_old_b_id = None
                    for old_b in old_p.blocks:
                        iou = _compute_iou(new_b.bbox, old_b.bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_old_b_id = old_b.id

                    if best_iou > 0.3 and best_old_b_id in old_trans:
                        new_trans[new_b.id] = old_trans[best_old_b_id]

        save_translations(workspace.root, lang, new_trans)


def _reinsert_user_blocks(workspace: Workspace, existing_pages: Dict[str, Page]) -> int:
    """Carry manually-added (user_added) blocks from the old workspace into the
    freshly re-extracted pages, matched by original filename. Re-detection never
    produces these, so without this they would be lost on ``--overwrite``."""
    restored = 0
    for new_p in workspace.pages:
        fname = os.path.basename(new_p.original)
        old_p = existing_pages.get(fname)
        if not old_p:
            continue
        for ob in old_p.blocks:
            if getattr(ob, 'user_added', False):
                new_p.blocks.append(ob)
                restored += 1
    return restored

def _normalize_block_ids(workspace: Workspace) -> Dict[str, str]:
    """Rewrite block ids so they match each page's final index/position.

    On resume, reused pages keep their old ids (``p{old_idx}_b...``) while their
    ``page.index`` is reassigned, leaving the id's embedded page number stale.
    Regenerate ids deterministically and return the old→new map so callers can
    keep translation files (which are keyed by block id) in sync.
    """
    id_map: Dict[str, str] = {}
    for page in workspace.pages:
        det_idx = 0  # count detected blocks only; user_added blocks keep their stable id
        for blk in page.blocks:
            if getattr(blk, 'user_added', False):
                continue
            new_id = block_id(page.index, det_idx)
            det_idx += 1
            if blk.id != new_id:
                id_map[blk.id] = new_id
                blk.id = new_id
    return id_map


def _remap_translation_keys(workspace: Workspace, id_map: Dict[str, str]) -> None:
    """Apply an old→new block-id map to every translation file in the task."""
    if not id_map:
        return
    trans_dir = get_translations_dir(workspace.root)
    if not os.path.isdir(trans_dir):
        return
    for f in os.listdir(trans_dir):
        if not f.endswith('.json'):
            continue
        lang = f[:-5]
        trans = load_translations(workspace.root, lang)
        if not trans:
            continue
        remapped = {id_map.get(k, k): v for k, v in trans.items()}
        save_translations(workspace.root, lang, remapped)


async def _extract_task(
    task_name: str,
    task_input_dir: str,
    task_work_dir: str,
    cfg: Config,
    device: str,
    verbose: bool,
    target_lang: Optional[str],
    overwrite: bool,
) -> Workspace:
    """Extract a single task (one subdirectory worth of images)."""
    files = _list_images(task_input_dir)
    if not files:
        raise FileNotFoundError(f"No images found under {task_input_dir}")

    logger.info(f"[task: {task_name}] Found {len(files)} image(s)")

    # Resume logic: Load existing workspace if pages.json exists.
    # Try to load existing workspace to salvage translations even if we overwrite
    existing_pages: Dict[str, Page] = {}
    if os.path.exists(os.path.join(task_work_dir, "pages.json")):
        try:
            ws_old = load_workspace(task_work_dir)
            existing_pages = {os.path.basename(p.original): p for p in ws_old.pages}
            if not overwrite:
                logger.info(f"[task: {task_name}] Found existing workspace with {len(existing_pages)} processed pages. Resuming...")
            else:
                logger.info(f"[task: {task_name}] Found existing workspace. Will attempt to merge translations into new extraction...")
        except Exception as e:
            logger.warning(f"[task: {task_name}] Could not load existing workspace: {e}. Starting fresh.")

    workspace = Workspace(
        root=task_work_dir,
        target_lang=target_lang or cfg.translator.target_lang,
        source_lang=cfg.translator.source_lang,
        task_name=task_name,
    )
    for i, path in enumerate(files):
        fname = os.path.basename(path)
        if not overwrite and fname in existing_pages:
            page = existing_pages[fname]
            page.index = i
            workspace.pages.append(page)
            logger.info(f"[task: {task_name}] [page {i}] skipped (already processed): {os.path.basename(path)}")
            continue

        try:
            page = await _process_image(path, i, cfg, device, workspace, verbose)
            workspace.pages.append(page)
            # Periodic checkpoint for crash recovery (avoid an O(n^2) full
            # rewrite of pages.json on every single page).
            if (i + 1) % 10 == 0:
                save_workspace(workspace)
        except Exception as e:
            logger.error(f"[task: {task_name}] Failed on {path}: {e}")
            if verbose:
                raise

    if overwrite and existing_pages:
        # Preserve manually-added blocks across the re-extraction, then migrate translations.
        restored = _reinsert_user_blocks(workspace, existing_pages)
        if restored:
            logger.info(f"[task: {task_name}] Preserved {restored} manually-added block(s) across re-extract")
        _merge_task_translations(workspace, existing_pages)

    # Keep block ids consistent with final page positions and carry translations along.
    id_map = _normalize_block_ids(workspace)
    if id_map:
        logger.info(f"[task: {task_name}] Normalized {len(id_map)} block id(s) to match page order")
        _remap_translation_keys(workspace, id_map)

    save_workspace(workspace)
    logger.info(f"[task: {workspace.task_name}] Workspace written: {workspace.pages_json_path}")
    return workspace


async def run_extract(
    input_path: str,
    work_dir: str,
    cfg: Config,
    verbose: bool = False,
    target_lang: Optional[str] = None,
    overwrite: bool = False,
) -> List[Workspace]:
    """Run detection + OCR + inpainting over the input and write the workspace.

    Returns a list of Workspace objects, one per task (subdirectory).
    """
    input_path = os.path.abspath(os.path.expanduser(input_path))
    work_dir = os.path.abspath(os.path.expanduser(work_dir))
    os.makedirs(work_dir, exist_ok=True)

    tasks = _discover_input_tasks(input_path)
    if not tasks:
        raise FileNotFoundError(f"No images found under {input_path}")

    logger.info(f"Discovered {len(tasks)} task(s): {[t[0] for t in tasks]}")

    device = _select_device(cfg.use_gpu)
    logger.info(f"Using device: {device}")

    # Pre-load models once.
    await prepare_detection(cfg.detector.detector)
    if (cfg.detector.secondary_detector != Detector.none
            and cfg.detector.secondary_detector != cfg.detector.detector):
        await prepare_detection(cfg.detector.secondary_detector)
    await prepare_ocr(cfg.ocr.ocr, device)
    if cfg.inpainter.inpainter != Inpainter.none:
        await prepare_inpainting(cfg.inpainter.inpainter, device)

    workspaces: List[Workspace] = []
    for task_name, task_input_dir in tasks:
        task_work_dir = os.path.join(work_dir, task_name)
        ws = await _extract_task(
            task_name, task_input_dir, task_work_dir, cfg, device,
            verbose, target_lang, overwrite,
        )
        workspaces.append(ws)

    logger.info(f"All tasks complete: {len(workspaces)} workspace(s) written under {work_dir}")
    return workspaces
