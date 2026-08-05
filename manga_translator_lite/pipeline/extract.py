"""Step 1: detection + OCR + inpainting → workspace files.

Walks the input path, runs the local CV pipeline on each image, and writes:

  - workspace/<task>/clean/<idx>_<name>.png    text-removed image
  - workspace/<task>/pages.json                metadata + OCR text per block

Each image-bearing subdirectory is treated as a separate *task*. Root-level
images remain their own task even when subdirectories are present, and a single
input image remains a one-image task.

Translation is left blank for the translate step to fill in.
"""
from __future__ import annotations

import os
import shutil
from typing import Dict, List, Optional, Set

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
from .geometry import _iou_xyxy, _overlap_min, match_boxes_by_iou
from .input_discovery import discover_input_tasks, source_fingerprint, source_matches
from ..utils import (
    TextBlock,
    cv2_imwrite,
    get_logger,
    is_valuable_text,
    load_image,
    sort_regions,
)
from .schema import (
    Block, Page, Workspace, block_id, save_workspace, load_workspace,
    load_translations, save_translations, get_translations_dir
)

logger = get_logger('extract')

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
    extra Quadrilaterals contributed by the secondary detector — they have no stroke-level
    mask data, so the caller box-fills them into the erase mask.
    """
    # Resolve the detection size once for this page (auto-sizes when detection_size = -1)
    # and reuse it for the secondary detector so both run at the same resolution.
    eff_ds = _effective_detection_size(cfg, img_rgb)
    if cfg.detector.detection_size is None or cfg.detector.detection_size <= 0:
        logger.info(f"detection_size=auto → {eff_ds} (page {img_rgb.shape[1]}x{img_rgb.shape[0]}, "
                    f"scale={cfg.detector.detection_size_scale})")

    textlines, mask_raw, mask = await _dispatch_det(
        cfg, cfg.detector.detector, img_rgb, device, verbose, cfg.detector.box_threshold, eff_ds)

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
            # art; box-filling it would wipe a large region, so drop oversized candidates.
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
            secondary_only.append(s)
        textlines = textlines + secondary_only
        logger.info(f"detector fusion: {cfg.detector.detector.value}={len(primary_boxes)} "
                    f"+{len(secondary_only)} new from {secondary.value} "
                    f"(of {len(sec_textlines)}; {dup} overlap, {oversize} oversize dropped)")

    return textlines, mask_raw, mask, secondary_only


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
    source_size, source_mtime_ns = source_fingerprint(img_path)

    # 1. detection (optionally fused with a secondary detector to boost recall)
    # Free the previous page's inpaint-sized reserved blocks before ctd allocates its
    # (larger) detection_size activations, so peak VRAM stays flat across pages.
    _free_vram(device)
    textlines, mask_raw, mask, secondary_only = await _detect_fused(cfg, img_rgb, device, verbose)

    _, ext = os.path.splitext(os.path.basename(img_path))
    if not ext:
        ext = '.png'
    clean_name = f"{page_idx:04d}_{os.path.splitext(os.path.basename(img_path))[0]}{ext}"
    clean_rel = f"clean/{clean_name}"
    clean_abs = os.path.join(workspace.clean_dir, clean_name)
    os.makedirs(workspace.clean_dir, exist_ok=True)

    if not textlines:
        logger.info(f"[page {page_idx}] no text detected — marked as no_text, copying original")
        shutil.copy2(img_path, clean_abs)
        return Page(
            index=page_idx,
            name=os.path.basename(img_path),
            size=(w, h),
            original=os.path.basename(img_path),
            clean=clean_rel,
            blocks=[],
            no_text=True,
            source_size=source_size,
            source_mtime_ns=source_mtime_ns,
        )

    # 2. OCR — keep every detected line. OCR text may be empty for handwriting/symbols;
    # those lines are still erased below (just not translated).
    ocr_textlines = await dispatch_ocr(cfg.ocr.ocr, img_rgb, textlines, cfg.ocr, device, verbose)

    min_len = cfg.ocr.min_text_length

    def _is_translatable(tl) -> bool:
        return bool(tl.text and tl.text.strip() and is_valuable_text(tl.text)
                    and len(tl.text) >= min_len)

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

    # 3b. Erase-only regions = detected lines rejected by the translation rules
    # (empty OCR, symbols, handwritten kana). Recorded for reclean; still erased below.
    erase_only_quads = [_quad_from_textline(tl) for tl in ocr_textlines if not _is_translatable(tl)]

    # 4. mask refinement — build the erase mask from ALL detected lines (translate set ∪
    # erase-only set), so residual non-translated text gets inpainted too. mask_refinement
    # needs merged TextBlocks (it reads .lines), so merge every detected line first.
    if mask is None and ocr_textlines:
        erase_blocks = await dispatch_textline_merge(ocr_textlines, w, h, verbose=verbose)
        if erase_blocks:
            mask = await dispatch_mask_refinement(
                erase_blocks,
                img_rgb,
                mask_raw if mask_raw is not None else np.zeros((h, w), dtype=np.uint8),
                'fit_text',
                cfg.mask_dilation_offset,
                cfg.ocr.ignore_bubble,
                verbose,
                cfg.kernel_size,
            )

    # 4b. Secondary-detector-only regions have no stroke-level mask (the primary missed
    # them), so the stroke refinement above can't erase them. Box-fill their boxes into the
    # mask — coarser than stroke masks, but it's only the extra recall the primary lacked.
    if secondary_only:
        if mask is None:
            mask = np.zeros((h, w), dtype=np.uint8)
        for q in secondary_only:
            x1, y1, x2, y2 = (int(round(v)) for v in q.xyxy)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    # 5. inpainting
    # Release the detection/OCR stage's reserved VRAM before lama allocates its own
    # large activations — this is what keeps detection_size=2560 + secondary detector
    # from inflating peak VRAM / OOM-ing on big pages.
    _free_vram(device)
    inpaint_done = False
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
        inpaint_done = True
    else:
        inpainted = img_rgb

    if not inpaint_done:
        shutil.copy2(img_path, clean_abs)
    else:
        pil_img = Image.fromarray(inpainted)
        _, ext = os.path.splitext(clean_abs)
        ext = ext.lower()
        
        try:
            if pil.format == 'JPEG' or ext in ['.jpg', '.jpeg']:
                # quality='keep' only works on images decoded from JPEG; this is a
                # freshly created array, so save at high quality with 4:4:4
                # subsampling to keep text edges crisp.
                pil_img.save(clean_abs, format='JPEG', quality=95, subsampling=0, optimize=True)
            elif pil.format == 'WEBP' or ext == '.webp':
                pil_img.save(clean_abs, format='WEBP', quality=85, method=6)
            elif pil.format == 'PNG' or ext == '.png':
                pil_img.save(clean_abs, format='PNG', optimize=True)
            else:
                pil_img.save(clean_abs)
        except Exception as e:
            logger.warning(f"[page {page_idx}] PIL save failed for {clean_abs}, fallback to cv2: {e}")
            cv2_imwrite(clean_abs, cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR))

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
        source_size=source_size,
        source_mtime_ns=source_mtime_ns,
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
                new_detected = [b for b in new_p.blocks if not getattr(b, 'user_added', False)]
                old_detected = [b for b in old_p.blocks if not getattr(b, 'user_added', False)]
                for new_idx, old_idx in match_boxes_by_iou(
                    [b.bbox for b in new_detected],
                    [b.bbox for b in old_detected],
                ):
                    old_block = old_detected[old_idx]
                    if old_block.id in old_trans:
                        new_trans[new_detected[new_idx].id] = old_trans[old_block.id]

        save_translations(workspace.root, lang, new_trans)


def _reinsert_user_blocks(
    workspace: Workspace,
    existing_pages: Dict[str, Page],
    page_names: Optional[Set[str]] = None,
) -> int:
    """Carry manually-added (user_added) blocks from the old workspace into the
    freshly re-extracted pages, matched by original filename. Re-detection never
    produces these, so without this they would be lost on ``--overwrite``."""
    restored = 0
    for new_p in workspace.pages:
        fname = os.path.basename(new_p.original)
        if page_names is not None and fname not in page_names:
            continue
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
    files: List[str],
    task_work_dir: str,
    cfg: Config,
    device: str,
    verbose: bool,
    target_lang: Optional[str],
    overwrite: bool,
) -> Workspace:
    """Extract a single task from its explicit list of input images."""
    if not files:
        raise FileNotFoundError(f"No images found for task {task_name}")

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
    failures = []
    reextracted_existing_pages: Set[str] = set()
    for i, path in enumerate(files):
        fname = os.path.basename(path)
        try:
            if not overwrite and fname in existing_pages:
                page = existing_pages[fname]
                if page.source_size is None or page.source_mtime_ns is None:
                    page.source_size, page.source_mtime_ns = source_fingerprint(path)
                    page.index = i
                    workspace.pages.append(page)
                    logger.info(f"[task: {task_name}] [page {i}] skipped (source fingerprint initialized): "
                                f"{os.path.basename(path)}")
                    continue
                if source_matches(path, page.source_size, page.source_mtime_ns):
                    page.index = i
                    workspace.pages.append(page)
                    logger.info(f"[task: {task_name}] [page {i}] skipped (source unchanged): "
                                f"{os.path.basename(path)}")
                    continue
                logger.info(f"[task: {task_name}] [page {i}] source changed; re-extracting: "
                            f"{os.path.basename(path)}")
                reextracted_existing_pages.add(fname)

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
            failures.append((os.path.basename(path), e))
            # Preserve the prior page on a failed refresh. Its stale fingerprint makes
            # the next run retry it, while keeping translations and manual edits intact.
            if fname in existing_pages:
                old_page = existing_pages[fname]
                old_page.index = i
                workspace.pages.append(old_page)

    if failures:
        save_workspace(workspace)
        failed_names = ', '.join(name for name, _ in failures)
        raise RuntimeError(
            f"[task: {task_name}] {len(failures)} page(s) failed: {failed_names}. "
            "Completed pages were checkpointed; rerun extract to retry the failed pages."
        )

    if (overwrite or reextracted_existing_pages) and existing_pages:
        # Preserve manually-added blocks across re-extraction, then migrate translations.
        page_names = None if overwrite else reextracted_existing_pages
        restored = _reinsert_user_blocks(workspace, existing_pages, page_names)
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

    tasks = discover_input_tasks(input_path)
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
    for task_name, task_files in tasks:
        task_work_dir = os.path.join(work_dir, task_name)
        ws = await _extract_task(
            task_name, task_files, task_work_dir, cfg, device,
            verbose, target_lang, overwrite,
        )
        workspaces.append(ws)

    logger.info(f"All tasks complete: {len(workspaces)} workspace(s) written under {work_dir}")
    return workspaces
