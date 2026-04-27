"""Step 3: combine clean images + translations into final output.

Reads pages.json and the matching ``clean/*.png``, draws translated text onto
each clean image using the rendering module, and writes the result into the
output directory.

This step iterates over all task subdirectories and mirrors the subdirectory
structure in the output directory. **Every** input image produces an output
image — pages marked ``no_text`` are copied as-is so the output count always
matches the input count.
"""
from __future__ import annotations

import os
import shutil
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from ..config import Config
from ..rendering import dispatch as dispatch_rendering
from ..utils import BASE_PATH, TextBlock, get_logger
from .schema import Block, Page, Workspace, discover_tasks, load_workspace

logger = get_logger('render')

DEFAULT_FONT = os.path.join(BASE_PATH, 'fonts', 'Arial-Unicode-Regular.ttf')


def _block_to_textblock(block: Block, target_lang: str, render_cfg) -> TextBlock:
    """Reconstruct a TextBlock for the rendering layer."""
    lines = np.array(block.lines, dtype=np.int32) if block.lines else np.array([block.polygon], dtype=np.int32)
    fg = render_cfg.font_color_fg or block.fg_color
    bg = render_cfg.font_color_bg or block.bg_color

    direction = block.direction
    if render_cfg.direction.value != "auto":
        direction = "h" if render_cfg.direction.value in ("h", "horizontal") else "v"

    alignment = block.alignment
    if render_cfg.alignment.value != "auto":
        alignment = render_cfg.alignment.value

    tb = TextBlock(
        lines=lines.tolist(),
        texts=[block.text],
        font_size=block.font_size or 24,
        angle=block.angle,
        translation=block.translation,
        fg_color=tuple(fg),
        bg_color=tuple(bg),
        direction=direction,
        alignment=alignment,
        target_lang=target_lang,
        prob=block.prob,
    )
    tb.text_raw = block.text
    if render_cfg.font_color_bg is not None:
        tb.adjust_bg_color = False
    return tb


async def _render_page(page: Page, ws: Workspace, cfg: Config, out_dir: str) -> Optional[str]:
    """Render a single page.  Always produces an output file.

    - ``no_text`` pages → copy the original image directly.
    - Pages with blocks but no translations → copy the clean image.
    - Normal pages → render translated text onto the clean image.
    """
    out_name = os.path.splitext(page.name)[0] + ".png"
    out_path = os.path.join(out_dir, out_name)

    # no_text pages — copy original directly (pass-through)
    if page.no_text:
        if page.original and os.path.exists(page.original):
            img = Image.open(page.original).convert('RGB')
            img.save(out_path)
            logger.info(f"[page {page.index}] no_text → copied original as-is → {out_path}")
        elif os.path.exists(os.path.join(ws.root, page.clean)):
            shutil.copy2(os.path.join(ws.root, page.clean), out_path)
            logger.info(f"[page {page.index}] no_text → copied clean image → {out_path}")
        else:
            logger.warning(f"[page {page.index}] no_text but no source found, skipping")
            return None
        return out_path

    clean_path = os.path.join(ws.root, page.clean)
    if not os.path.exists(clean_path):
        logger.warning(f"[page {page.index}] clean image missing: {clean_path}, skipping")
        return None

    img_bgr = cv2.imread(clean_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        logger.warning(f"[page {page.index}] could not read {clean_path}")
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    blocks = [b for b in page.blocks if b.translation and b.translation.strip()]
    if not blocks:
        logger.info(f"[page {page.index}] no translated blocks, copying clean image as-is")
        rendered_rgb = img_rgb
    else:
        text_regions: List[TextBlock] = [
            _block_to_textblock(b, ws.target_lang, cfg.render) for b in blocks
        ]
        if cfg.render.uppercase:
            for tb in text_regions:
                tb.translation = tb.translation.upper()
        elif cfg.render.lowercase:
            for tb in text_regions:
                tb.translation = tb.translation.lower()

        font_path = cfg.render.font_path or DEFAULT_FONT
        rendered_rgb = await dispatch_rendering(
            img_rgb,
            text_regions,
            font_path=font_path,
            font_size_fixed=cfg.render.font_size,
            font_size_offset=cfg.render.font_size_offset,
            font_size_minimum=cfg.render.font_size_minimum,
            hyphenate=not cfg.render.no_hyphenation,
            line_spacing=cfg.render.line_spacing,
            disable_font_border=cfg.render.disable_font_border,
        )

    Image.fromarray(rendered_rgb).save(out_path)
    logger.info(f"[page {page.index}] → {out_path}")
    return out_path


async def _render_task(task_name: str, task_work_dir: str, task_out_dir: str, cfg: Config) -> List[str]:
    """Render all pages for a single task."""
    workspace = load_workspace(task_work_dir)
    os.makedirs(task_out_dir, exist_ok=True)
    logger.info(f"[task: {task_name}] Rendering {len(workspace.pages)} page(s) into {task_out_dir}")

    written: List[str] = []
    for page in workspace.pages:
        path = await _render_page(page, workspace, cfg, task_out_dir)
        if path:
            written.append(path)

    no_text_count = sum(1 for p in workspace.pages if p.no_text)
    logger.info(f"[task: {task_name}] Wrote {len(written)} image(s) "
                f"({no_text_count} no-text pass-through)")
    return written


async def run_render(work_dir: str, out_dir: str, cfg: Config) -> List[str]:
    """Render all tasks under work_dir.

    Mirrors the subdirectory structure: work_dir/<task>/ → out_dir/<task>/.
    """
    work_dir = os.path.abspath(os.path.expanduser(work_dir))
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    tasks = discover_tasks(work_dir)
    if not tasks:
        raise FileNotFoundError(f"No task subdirectories found under {work_dir}")

    logger.info(f"Found {len(tasks)} task(s) to render: {tasks}")

    all_written: List[str] = []
    for task_name in tasks:
        task_work_dir = os.path.join(work_dir, task_name)
        task_out_dir = os.path.join(out_dir, task_name)
        try:
            written = await _render_task(task_name, task_work_dir, task_out_dir, cfg)
            all_written.extend(written)
        except FileNotFoundError:
            logger.warning(f"[task: {task_name}] No pages.json found, skipping.")

    logger.info(f"Rendered {len(all_written)} image(s) total across {len(tasks)} task(s)")
    return all_written
