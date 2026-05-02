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
    clean_rel = os.path.join("clean", clean_name)
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

    # 2. OCR
    textlines = await dispatch_ocr(cfg.ocr.ocr, img_rgb, textlines, cfg.ocr, device, verbose)
    textlines = [tl for tl in textlines if tl.text and tl.text.strip()]

    if not textlines:
        logger.info(f"[page {page_idx}] OCR empty — marked as no_text, copying original")
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

    # 3. textline merge → regions
    text_regions = await dispatch_textline_merge(textlines, w, h, verbose=verbose)
    text_regions = [r for r in text_regions if r.text and is_valuable_text(r.text)
                    and len(r.text) >= cfg.ocr.min_text_length]
    text_regions = sort_regions(
        text_regions,
        right_to_left=cfg.render.rtl,
        img=img_rgb,
        force_simple_sort=cfg.force_simple_sort,
    )

    # 4. mask refinement
    if mask is None and text_regions:
        mask = await dispatch_mask_refinement(
            text_regions,
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
        if pil.format == 'JPEG':
            pil_img = Image.fromarray(inpainted)
            pil_img.format = pil.format
            pil_img.info = pil.info
            try:
                pil_img.save(clean_abs, format=pil.format, quality='keep', subsampling='keep')
            except Exception:
                cv2_imwrite(clean_abs, cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR))
        else:
            cv2_imwrite(clean_abs, cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR))

    logger.info(f"[page {page_idx}] saved clean → {clean_rel} ({len(text_regions)} blocks)")

    # If textline merge + filtering produced no valuable blocks, mark as no_text
    if not text_regions:
        return Page(
            index=page_idx,
            name=os.path.basename(img_path),
            size=(w, h),
            original=os.path.basename(img_path),
            clean=clean_rel,
            blocks=[],
            no_text=True,
        )

    blocks = [_serialise_block(r, page_idx, i) for i, r in enumerate(text_regions)]
    return Page(
        index=page_idx,
        name=os.path.basename(img_path),
        size=(w, h),
        original=os.path.basename(img_path),
        clean=clean_rel,
        blocks=blocks,
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
            # Incremental save after each page for crash recovery
            save_workspace(workspace)
        except Exception as e:
            logger.error(f"[task: {task_name}] Failed on {path}: {e}")
            if verbose:
                raise

    if overwrite and existing_pages:
        _merge_task_translations(workspace, existing_pages)

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
