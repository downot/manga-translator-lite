"""Step 1: detection + OCR + inpainting → workspace files.

Walks the input path, runs the local CV pipeline on each image, and writes:

  - workspace/clean/<idx>_<name>.png    text-removed image
  - workspace/pages.json                  metadata + OCR text per block

Translation is left blank for the translate step to fill in.
"""
from __future__ import annotations

import os
from typing import List, Optional

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
    get_logger,
    is_valuable_text,
    load_image,
    natural_sort,
    sort_regions,
)
from .schema import Block, Page, Workspace, block_id, save_workspace, load_workspace

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
    for root, _, names in os.walk(input_path):
        for n in natural_sort(names):
            ext = os.path.splitext(n)[1].lower()
            if ext in IMG_EXTS and not n.startswith('.'):
                files.append(os.path.join(root, n))
    return files


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

    clean_name = f"{page_idx:04d}_{os.path.splitext(os.path.basename(img_path))[0]}.png"
    clean_rel = os.path.join("clean", clean_name)
    clean_abs = os.path.join(workspace.clean_dir, clean_name)
    os.makedirs(workspace.clean_dir, exist_ok=True)

    if not textlines:
        logger.info(f"[page {page_idx}] no text detected, copying as clean image")
        cv2.imwrite(clean_abs, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return Page(
            index=page_idx,
            name=os.path.basename(img_path),
            size=(w, h),
            original=os.path.abspath(img_path),
            clean=clean_rel,
            blocks=[],
        )

    # 2. OCR
    textlines = await dispatch_ocr(cfg.ocr.ocr, img_rgb, textlines, cfg.ocr, device, verbose)
    textlines = [tl for tl in textlines if tl.text and tl.text.strip()]

    if not textlines:
        logger.info(f"[page {page_idx}] OCR empty, copying as clean image")
        cv2.imwrite(clean_abs, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return Page(
            index=page_idx,
            name=os.path.basename(img_path),
            size=(w, h),
            original=os.path.abspath(img_path),
            clean=clean_rel,
            blocks=[],
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

    cv2.imwrite(clean_abs, cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR))
    logger.info(f"[page {page_idx}] saved clean → {clean_rel} ({len(text_regions)} blocks)")

    blocks = [_serialise_block(r, page_idx, i) for i, r in enumerate(text_regions)]
    return Page(
        index=page_idx,
        name=os.path.basename(img_path),
        size=(w, h),
        original=os.path.abspath(img_path),
        clean=clean_rel,
        blocks=blocks,
    )


async def run_extract(
    input_path: str,
    work_dir: str,
    cfg: Config,
    verbose: bool = False,
    target_lang: Optional[str] = None,
    overwrite: bool = False,
) -> Workspace:
    """Run detection + OCR + inpainting over the input and write the workspace."""
    input_path = os.path.abspath(os.path.expanduser(input_path))
    work_dir = os.path.abspath(os.path.expanduser(work_dir))
    os.makedirs(work_dir, exist_ok=True)

    files = _list_images(input_path)
    if not files:
        raise FileNotFoundError(f"No images found under {input_path}")
    logger.info(f"Found {len(files)} image(s)")

    device = _select_device(cfg.use_gpu)
    logger.info(f"Using device: {device}")

    # Pre-load models once.
    await prepare_detection(cfg.detector.detector)
    await prepare_ocr(cfg.ocr.ocr, device)
    if cfg.inpainter.inpainter != Inpainter.none:
        await prepare_inpainting(cfg.inpainter.inpainter, device)

    # Resume logic: Load existing workspace if pages.json exists.
    existing_pages = {}
    if not overwrite and os.path.exists(os.path.join(work_dir, "pages.json")):
        try:
            ws_old = load_workspace(work_dir)
            existing_pages = {os.path.abspath(p.original): p for p in ws_old.pages}
            logger.info(f"Found existing workspace with {len(existing_pages)} processed pages. Resuming...")
        except Exception as e:
            logger.warning(f"Could not load existing workspace: {e}. Starting fresh.")

    workspace = Workspace(
        root=work_dir,
        target_lang=target_lang or cfg.translator.target_lang,
        source_lang=cfg.translator.source_lang,
    )
    for i, path in enumerate(files):
        abs_path = os.path.abspath(path)
        if abs_path in existing_pages:
            # Re-use existing page, but update its index to match current file list if needed.
            # Usually, if the file list is the same, indices won't change.
            page = existing_pages[abs_path]
            page.index = i
            workspace.pages.append(page)
            logger.info(f"[page {i}] skipped (already processed): {os.path.basename(path)}")
            continue

        try:
            page = await _process_image(path, i, cfg, device, workspace, verbose)
            workspace.pages.append(page)
            # Incremental save after each page for crash recovery
            save_workspace(workspace)
        except Exception as e:
            logger.error(f"Failed on {path}: {e}")
            if verbose:
                raise

    save_workspace(workspace)
    logger.info(f"Workspace written: {workspace.pages_json_path}")
    return workspace
