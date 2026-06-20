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
import sys
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from ..config import Config
from ..rendering import dispatch as dispatch_rendering
from ..utils import BASE_PATH, TextBlock, get_logger, cv2_imread
from .schema import Block, Page, Workspace, discover_tasks, load_workspace, load_translations

logger = get_logger('render')

DEFAULT_FONT = os.path.join(BASE_PATH, 'fonts', 'Arial-Unicode-Regular.ttf')


def _scale_poly(pts, scale: float) -> np.ndarray:
    """Scale a set of points about their bounding-box center; returns int32 array."""
    arr = np.array(pts, dtype=np.float64).reshape(-1, 2)
    if scale == 1.0 or arr.size == 0:
        return arr.astype(np.int32)
    cx = (arr[:, 0].min() + arr[:, 0].max()) / 2
    cy = (arr[:, 1].min() + arr[:, 1].max()) / 2
    arr[:, 0] = cx + (arr[:, 0] - cx) * scale
    arr[:, 1] = cy + (arr[:, 1] - cy) * scale
    return arr.astype(np.int32)


def _block_to_textblock(block: Block, translation_text: str, target_lang: str, render_cfg, box_scale: float = 1.0) -> TextBlock:
    """Reconstruct a TextBlock for the rendering layer."""
    lines = np.array(block.lines, dtype=np.int32) if block.lines else np.array([block.polygon], dtype=np.int32)
    # Apply the task-level uniform box enlargement (unless this block opts out).
    scale = 1.0 if block.scale_exempt else (box_scale or 1.0)
    if scale != 1.0 and lines.size:
        lines = _scale_poly(lines.reshape(-1, 2), scale).reshape(lines.shape)
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
        translation=translation_text,
        fg_color=tuple(fg),
        bg_color=tuple(bg),
        direction=direction,
        alignment=alignment,
        target_lang=target_lang,
        prob=block.prob,
    )
    tb.block_id = block.id
    tb.text_raw = block.text
    tb.fixed_region = block.fixed_region  # user-sized box → renderer must not auto-expand it
    tb.box_scale_applied = scale          # lifts the font-size ceiling so box_scale truly magnifies text
    if render_cfg.font_color_bg is not None:
        tb.adjust_bg_color = False
    return tb


async def _render_page(page: Page, ws: Workspace, cfg: Config, out_dir: str, translations: dict, out_name: str) -> Optional[str]:
    """Render a single page.  Always produces an output file.

    - ``no_text`` pages → copy the original image directly.
    - Pages with blocks but no translations → copy the clean image.
    - Normal pages → render translated text onto the clean image.
    """
    out_path = os.path.join(out_dir, out_name)

    # no_text pages — copy clean directly (which is a copy of original)
    if page.no_text:
        clean_abs = os.path.join(ws.root, page.clean)
        if os.path.exists(clean_abs):
            shutil.copy2(clean_abs, out_path)
            logger.info(f"[page {page.index}] no_text → copied original (from clean) → {out_path}")
        else:
            logger.warning(f"[page {page.index}] no_text but clean image missing, skipping")
            return None
        return out_path

    clean_path = os.path.join(ws.root, page.clean)
    if not os.path.exists(clean_path):
        logger.warning(f"[page {page.index}] clean image missing: {clean_path}, skipping")
        return None

    img_bgr = cv2_imread(clean_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        logger.warning(f"[page {page.index}] could not read {clean_path}")
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Background-fill pass: paint a solid color behind blocks flagged bg_fill (covers
    # residual/original content). Done before drawing text so the text sits on top.
    #   "white" → pure white;  "match" → the block's estimated region color (bg_color),
    # so the patch blends into a tinted / screentone background instead of a white scar.
    fill_blocks = [b for b in page.blocks
                   if getattr(b, 'bg_fill', 'none') in ('white', 'match') and b.polygon]
    for b in fill_blocks:
        sc = 1.0 if b.scale_exempt else (ws.box_scale or 1.0)
        pts = _scale_poly(b.polygon, sc)
        if len(pts) >= 3:
            if getattr(b, 'bg_fill', 'none') == 'match':
                bc = b.bg_color or [255, 255, 255]
                col = (int(bc[0]), int(bc[1]), int(bc[2]))   # bg_color is RGB, img_rgb is RGB
            else:
                col = (255, 255, 255)
            cv2.fillPoly(img_rgb, [pts], col)

    blocks_to_render: List[tuple[Block, str]] = []
    for b in page.blocks:
        t = translations.get(b.id)
        if t and t.text and t.text.strip():
            blocks_to_render.append((b, t.text))

    if not blocks_to_render:
        if not fill_blocks:
            logger.info(f"[page {page.index}] no translated blocks, copying clean image as-is")
            shutil.copy2(clean_path, out_path)
            logger.info(f"[page {page.index}] → {out_path}")
            return out_path
        # Background fills changed the image but there is no text to draw → save the filled image.
        logger.info(f"[page {page.index}] bg-fill only ({len(fill_blocks)} region(s)), no text")
        rendered_rgb = img_rgb
    else:
        text_regions: List[TextBlock] = [
            _block_to_textblock(b, text, ws.target_lang, cfg.render, ws.box_scale) for b, text in blocks_to_render
        ]
        if cfg.render.uppercase:
            for tb in text_regions:
                tb.translation = tb.translation.upper()
        elif cfg.render.lowercase:
            for tb in text_regions:
                tb.translation = tb.translation.lower()

        font_path = cfg.render.font_path or DEFAULT_FONT
        # Per-task overrides (pages.json) take precedence; config.toml is the fallback.
        eff_min = ws.font_size_minimum if ws.font_size_minimum is not None else cfg.render.font_size_minimum
        eff_expand = (ws.font_size_minimum_expand_limit if ws.font_size_minimum_expand_limit is not None
                      else cfg.render.font_size_minimum_expand_limit)
        rendered_rgb = await dispatch_rendering(
            img_rgb,
            text_regions,
            font_path=font_path,
            font_size_fixed=cfg.render.font_size,
            font_size_offset=cfg.render.font_size_offset,
            font_size_minimum=eff_min,
            font_size_minimum_expand_limit=eff_expand,
            hyphenate=not cfg.render.no_hyphenation,
            line_spacing=cfg.render.line_spacing,
            disable_font_border=cfg.render.disable_font_border,
        )

    pil_img = Image.fromarray(rendered_rgb)
    _, ext = os.path.splitext(out_path)
    ext_lower = ext.lower()

    clean_format = None
    try:
        pil_clean = Image.open(clean_path)
        clean_format = pil_clean.format
    except Exception:
        pass

    try:
        if clean_format == 'JPEG' or ext_lower in ['.jpg', '.jpeg']:
            # Freshly rendered array: quality='keep' is not applicable. Save at
            # high quality with 4:4:4 subsampling to keep rendered text crisp.
            pil_img.save(out_path, format='JPEG', quality=95, subsampling=0, optimize=True)
        elif clean_format == 'WEBP' or ext_lower == '.webp':
            pil_img.save(out_path, format='WEBP', quality=85, method=6)
        elif clean_format == 'PNG' or ext_lower == '.png':
            pil_img.save(out_path, format='PNG', optimize=True)
        else:
            pil_img.save(out_path)
    except Exception as e:
        logger.warning(f"[page {page.index}] PIL save failed for {out_path}, falling back: {e}")
        pil_img.save(out_path)

    logger.info(f"[page {page.index}] → {out_path}")
    return out_path


async def _render_task(task_name: str, task_work_dir: str, task_out_dir: str, cfg: Config) -> List[str]:
    """Render all pages for a single task."""
    workspace = load_workspace(task_work_dir)
    os.makedirs(task_out_dir, exist_ok=True)
    logger.info(f"[task: {task_name}] Rendering {len(workspace.pages)} page(s) into {task_out_dir}")

    # Load translations for the target language
    translations = load_translations(workspace.root, workspace.target_lang)

    # If basenames are not unique (e.g. merged chapters that each contain 01.jpg),
    # naming outputs by basename would either overwrite pages or, with a numeric
    # suffix, interleave them when the folder is sorted by name. In that case
    # prefix every output with its reading-order index so the directory always
    # sorts in page order. Unique names are left untouched.
    names = [p.name for p in workspace.pages]
    has_dupes = len(set(names)) != len(names)

    written: List[str] = []
    for i, page in enumerate(workspace.pages):
        out_name = f"{i + 1:04d}_{page.name}" if has_dupes else page.name
        path = await _render_page(page, workspace, cfg, task_out_dir, translations, out_name)
        if path:
            written.append(path)

    no_text_count = sum(1 for p in workspace.pages if p.no_text)
    logger.info(f"[task: {task_name}] Wrote {len(written)} image(s) "
                f"({no_text_count} no-text pass-through)")
    return written


async def run_render(
    work_dir: str,
    out_dir: str,
    cfg: Config,
    check: bool = False,
    no_check: bool = False,
    yes: bool = False,
) -> List[str]:
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

    from rich.console import Console
    console = Console()

    all_written: List[str] = []
    for task_name in tasks:
        task_work_dir = os.path.join(work_dir, task_name)
        task_out_dir = os.path.join(out_dir, task_name)
        try:
            workspace = load_workspace(task_work_dir)
        except Exception as e:
            logger.error(f"[task: {task_name}] Error loading workspace: {e}", exc_info=True)
            continue

        # Proofreading check before rendering
        should_run_check = False
        if check:
            should_run_check = True
        elif no_check:
            should_run_check = False
        elif sys.stdin.isatty():
            try:
                console.print(f"\n[bold cyan]Task: {task_name}[/bold cyan]")
                ans = input("Would you like to run spelling/fluency proofreading checks on translations before rendering? [y/N]: ").strip().lower()
                if ans in ("y", "yes"):
                    should_run_check = True
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Skipping proofreading check...[/yellow]")
                should_run_check = False

        if should_run_check:
            from .check import run_proofread_check
            proceed = await run_proofread_check(workspace, cfg, yes=yes)
            if not proceed:
                logger.info(f"[task: {task_name}] Rendering cancelled by user during proofreading check.")
                raise KeyboardInterrupt("Cancelled by user during proofreading check")

        try:
            written = await _render_task(task_name, task_work_dir, task_out_dir, cfg)
            all_written.extend(written)
        except Exception as e:
            logger.error(f"[task: {task_name}] Error during rendering: {e}", exc_info=True)

    logger.info(f"Rendered {len(all_written)} image(s) total across {len(tasks)} task(s)")
    return all_written
