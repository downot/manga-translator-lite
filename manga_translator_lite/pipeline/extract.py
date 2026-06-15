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
import shutil
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from ..config import Config, Detector, Inpainter, Ocr
from ..detection import dispatch as dispatch_detection, prepare as prepare_detection
from ..inpainting import dispatch as dispatch_inpainting, prepare as prepare_inpainting
from ..mask_refinement import dispatch as dispatch_mask_refinement
from ..ocr import dispatch as dispatch_ocr, prepare as prepare_ocr
from ..textline_merge import dispatch as dispatch_textline_merge
from ..utils import (
    TextBlock,
    cv2_imwrite,
    get_logger,
    is_valuable_text,
    load_image,
    natural_sort,
    sort_regions,
)
from .schema import (
    Block, Page, Workspace, block_id, save_workspace, load_workspace,
    load_translations, save_translations, get_translations_dir, Translation
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

    # 1. detection
    textlines, mask_raw, mask = await dispatch_detection(
        cfg.detector.detector,
        img_rgb,
        cfg.detector.detection_size,
        cfg.detector.text_threshold,
        cfg.detector.box_threshold,
        cfg.detector.unclip_ratio,
        cfg.detector.det_invert,
        cfg.detector.det_gamma_correct,
        cfg.detector.det_rotate,
        cfg.detector.det_auto_rotate,
        device,
        verbose,
    )

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

    # 5. inpainting
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
    )


def _compute_iou(box1: List[int], box2: List[int]) -> float:
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
