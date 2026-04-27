"""Step 3: combine clean images + translations into final output.

Reads pages.json and the matching ``clean/*.png``, draws translated text onto
each clean image using the rendering module, and writes the result into the
output directory.
"""
from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from ..config import Config
from ..rendering import dispatch as dispatch_rendering
from ..utils import BASE_PATH, TextBlock, get_logger
from .schema import Block, Page, Workspace, load_workspace

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

    out_name = os.path.splitext(page.name)[0] + ".png"
    out_path = os.path.join(out_dir, out_name)
    Image.fromarray(rendered_rgb).save(out_path)
    logger.info(f"[page {page.index}] → {out_path}")
    return out_path


async def run_render(work_dir: str, out_dir: str, cfg: Config) -> List[str]:
    work_dir = os.path.abspath(os.path.expanduser(work_dir))
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    workspace = load_workspace(work_dir)
    logger.info(f"Rendering {len(workspace.pages)} page(s) into {out_dir}")

    written: List[str] = []
    for page in workspace.pages:
        path = await _render_page(page, workspace, cfg, out_dir)
        if path:
            written.append(path)

    logger.info(f"Wrote {len(written)} image(s)")
    return written
