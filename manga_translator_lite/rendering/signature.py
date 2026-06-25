"""Translator-credit signature baked onto the output pages.

Renders a small, low-key credit box — translator name plus a discreet mention of
this open-source system — into a page corner. Two orientations are supported:

  * **horizontal** — lines stacked top-to-bottom (default for Latin / LTR).
  * **vertical**   — characters stacked top-to-bottom, columns right-to-left
                     (natural for CJK pages).

The pass is intentionally self-contained (PIL only) and runs *after* the normal
render, so it also applies to ``no_text`` pass-through pages. It is a no-op unless
``[signature] enabled = true`` and a ``translator`` (or custom ``text``) is set.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# The bundled fonts live in a top-level ``fonts/`` directory. Mirror the project's
# BASE_PATH convention (repo root = two levels up from this module's package), and
# keep the package dir as a fallback in case of a different layout.
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # …/manga_translator_lite
_FONT_BASES = [os.path.dirname(_PKG_DIR), _PKG_DIR]                       # repo root, then package
_FONT_NAMES = ["msyh.ttc", "SourceHanSans-Bold.ttc", "LXGWWenKai-Medium.ttf",
               "Arial-Unicode-Regular.ttf"]
_BUNDLED_FONTS = [os.path.join(base, "fonts", name)
                  for base in _FONT_BASES for name in _FONT_NAMES]

# Target languages that read best with a vertical credit when direction = auto.
_CJK_LANGS = {"JPN", "JA", "JP", "CHS", "CHT", "ZH", "ZHO", "CN"}

# Fixed open-source credit. This is the project's own attribution and is NOT
# user-configurable — it is always appended to the signature so every output
# carries a discreet pointer back to the system that produced it.
PROJECT_CREDIT = "MTL.downot.moe"


def needs_signature(sig, index: int, total: int) -> bool:
    """Whether the page at ``index`` (0-based) of ``total`` gets a signature."""
    if sig is None or not getattr(sig, "enabled", False):
        return False
    if not (getattr(sig, "translator", "") or getattr(sig, "text", "")):
        return False
    mode = getattr(sig, "pages", None)
    mode = getattr(mode, "value", mode)  # accept enum or str
    if mode in ("none", None):
        return False
    if mode == "every":
        return True
    is_first = index == 0
    is_last = index == max(total - 1, 0)
    if mode == "first":
        return is_first
    if mode == "last":
        return is_last
    if mode == "first_last":
        return is_first or is_last
    return False


def build_name(sig) -> str:
    """The translator-controlled line(s). No fixed prefix — the translator decides
    exactly what to show. Empty ``text`` → just the raw translator name."""
    translator = getattr(sig, "translator", "") or ""
    template = getattr(sig, "text", "") or ""
    if template:
        try:
            return template.replace("\\n", "\n").format(translator=translator, project=PROJECT_CREDIT)
        except Exception:
            return template.replace("\\n", "\n")
    return translator


# Back-compat alias (older callers / tests).
def build_text(sig) -> str:
    name = build_name(sig)
    return name if PROJECT_CREDIT in name else f"{name}\n{PROJECT_CREDIT}"


def resolve_direction(sig, target_lang: str) -> str:
    """horizontal | vertical, resolving 'auto' from the target language."""
    d = getattr(sig, "direction", None)
    d = getattr(d, "value", d)
    if d in ("horizontal", "h"):
        return "horizontal"
    if d in ("vertical", "v"):
        return "vertical"
    # auto
    lang = (target_lang or "").upper()
    return "vertical" if lang in _CJK_LANGS else "horizontal"


def _load_font(sig, render_font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    candidates: List[str] = []
    if getattr(sig, "font_path", ""):
        candidates.append(sig.font_path)
    if render_font_path:
        candidates.append(render_font_path)
    candidates.extend(_BUNDLED_FONTS)
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _corner_xy(position: str, block_w: int, block_h: int, W: int, H: int, margin: int) -> Tuple[int, int]:
    pos = getattr(position, "value", position) or "bottom-right"
    right = "right" in pos
    bottom = "bottom" in pos or pos == "bottom-right" or pos == "bottom-left"
    # default to bottom for unknown
    if "top" in pos:
        bottom = False
    x = (W - block_w - margin) if right else margin
    y = (H - block_h - margin) if bottom else margin
    return max(0, x), max(0, y)


def _stroke_color(text_rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """A faint contrasting outline so the credit stays legible on any art."""
    return (255, 255, 255) if (sum(text_rgb) / 3) < 128 else (0, 0, 0)


def _lighten(rgb: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    """Blend a colour toward white by ``t`` (0 = unchanged, 1 = white)."""
    return tuple(int(round(c + (255 - c) * t)) for c in rgb)


def _measure_h(draw, text, font, stroke_w):
    b = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_w)
    return (b[2] - b[0], b[3] - b[1], b[0], b[1])


def render_signature_on_image(img: Image.Image, sig, target_lang: str = "ENG",
                              render_font_path: Optional[str] = None,
                              scale: float = 1.0, offset: Tuple[int, int] = (0, 0)) -> Image.Image:
    """Return ``img`` (RGB) with the signature composited into the chosen corner.

    Layered style: the translator's name is large and rendered **on top of** a much
    lighter, smaller open-source credit (the two overlap), so the credit reads as a
    discreet watermark the name is stamped over.

    ``scale`` and ``offset`` are the per-task adjustments made by dragging the
    signature in the editor (size multiplier and pixel offset from the corner).
    """
    base = img.convert("RGBA")
    W, H = base.size
    name_text = build_name(sig)
    if not name_text.strip():
        return img
    show_domain = PROJECT_CREDIT not in name_text

    scale = 1.0 if not scale or scale <= 0 else float(scale)
    off_x, off_y = (int(offset[0]), int(offset[1])) if offset else (0, 0)

    # Name is the prominent element; the domain credit is a smaller, fainter layer.
    base_fs = sig.font_size if getattr(sig, "font_size", -1) and sig.font_size > 0 else max(18, round((W + H) / 85))
    name_fs = max(8, int(round(base_fs * scale)))
    domain_fs = max(7, round(name_fs * 0.5))
    margin = sig.margin if getattr(sig, "margin", -1) and sig.margin > 0 else max(8, round((W + H) / 110))

    direction = resolve_direction(sig, target_lang)
    name_font = _load_font(sig, render_font_path, name_fs)
    domain_font = _load_font(sig, render_font_path, domain_fs)

    rgb = sig.color_rgb if hasattr(sig, "color_rgb") else (128, 128, 128)
    op = getattr(sig, "opacity", 0.7)
    op = 0.0 if op is None else max(0.0, min(1.0, float(op)))
    name_fill = (rgb[0], rgb[1], rgb[2], int(255 * op))
    # Domain: much lighter colour + lower opacity → clearly recedes behind the name.
    d_rgb = _lighten(rgb, 0.68)
    domain_fill = (d_rgb[0], d_rgb[1], d_rgb[2], int(255 * op * 0.42))
    s_rgb = _stroke_color(rgb)
    name_stroke = (s_rgb[0], s_rgb[1], s_rgb[2], int(255 * op * 0.55))
    ds_rgb = _stroke_color(d_rgb)
    domain_stroke = (ds_rgb[0], ds_rgb[1], ds_rgb[2], int(255 * op * 0.22))
    name_sw = max(1, round(name_fs / 16))
    domain_sw = max(1, round(domain_fs / 16))

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pos = getattr(sig.position, "value", sig.position) if hasattr(sig, "position") else "bottom-right"
    right = "right" in (pos or "")

    if direction == "horizontal":
        name_lines = name_text.split("\n")
        gap = max(1, round(name_fs * 0.10))
        n_sizes = [_measure_h(draw, ln, name_font, name_sw) for ln in name_lines]
        name_w = max((w for w, _, _, _ in n_sizes), default=0)
        name_h = sum(h for _, h, _, _ in n_sizes) + gap * max(len(name_lines) - 1, 0)
        if show_domain:
            dw, dh, dox, doy = _measure_h(draw, PROJECT_CREDIT, domain_font, domain_sw)
        else:
            dw = dh = 0
        overlap = round(dh * 0.55) if show_domain else 0
        block_w = max(name_w, dw)
        block_h = name_h + (dh - overlap if show_domain else 0)
        x0, y0 = _corner_xy(pos, block_w, block_h, W, H, margin)
        x0 += off_x; y0 += off_y
        # Draw the domain first (lower & behind), then the big name on top.
        if show_domain:
            dy = y0 + name_h - overlap
            dx = (x0 + block_w - dw) if right else x0
            draw.text((dx - dox, dy - doy), PROJECT_CREDIT, font=domain_font,
                      fill=domain_fill, stroke_width=domain_sw, stroke_fill=domain_stroke)
        y = y0
        for ln, (w, h, ox, oy) in zip(name_lines, n_sizes):
            lx = (x0 + block_w - w) if right else x0
            draw.text((lx - ox, y - oy), ln, font=name_font, fill=name_fill,
                      stroke_width=name_sw, stroke_fill=name_stroke)
            y += h + gap
    else:  # vertical: name column(s) on the right; domain a fainter column to the left
        name_segs = name_text.split("\n")
        n_cell = max(1, round(name_fs * 1.12))
        n_colw = max(1, round(name_fs * 1.18))
        name_block_w = max(len(name_segs), 1) * n_colw
        name_block_h = max((len(s) for s in name_segs), default=1) * n_cell
        if show_domain:
            d_cell = max(1, round(domain_fs * 1.12))
            d_colw = max(1, round(domain_fs * 1.18))
            domain_h = len(PROJECT_CREDIT) * d_cell
            overlap = round(d_colw * 0.5)
        else:
            d_colw = domain_h = overlap = 0
        block_w = name_block_w + (d_colw - overlap if show_domain else 0)
        block_h = max(name_block_h, domain_h)
        x0, y0 = _corner_xy(pos, block_w, block_h, W, H, margin)
        x0 += off_x; y0 += off_y
        # name occupies the right part of the block
        name_x0 = x0 + (block_w - name_block_w)
        # domain column tucked to the left, overlapping the name block
        if show_domain:
            dcx = name_x0 - d_colw + overlap
            cy = y0
            for ch in PROJECT_CREDIT:
                b = draw.textbbox((0, 0), ch, font=domain_font, stroke_width=domain_sw)
                cw = b[2] - b[0]
                draw.text((dcx + (d_colw - cw) / 2 - b[0], cy - b[1]), ch, font=domain_font,
                          fill=domain_fill, stroke_width=domain_sw, stroke_fill=domain_stroke)
                cy += d_cell
        for ci, seg in enumerate(name_segs):
            cx = name_x0 + name_block_w - (ci + 1) * n_colw
            cy = y0
            for ch in seg:
                b = draw.textbbox((0, 0), ch, font=name_font, stroke_width=name_sw)
                cw = b[2] - b[0]
                draw.text((cx + (n_colw - cw) / 2 - b[0], cy - b[1]), ch, font=name_font,
                          fill=name_fill, stroke_width=name_sw, stroke_fill=name_stroke)
                cy += n_cell

    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out


def apply_signature_to_file(path: str, sig, target_lang: str = "ENG",
                            render_font_path: Optional[str] = None,
                            scale: float = 1.0, offset: Tuple[int, int] = (0, 0)) -> bool:
    """Open ``path``, bake the signature, and save back in the same format.

    Returns True on success, False if the file could not be processed.
    """
    try:
        src = Image.open(path)
        fmt = src.format
        out = render_signature_on_image(src, sig, target_lang, render_font_path, scale, offset)
        ext = os.path.splitext(path)[1].lower()
        if fmt == "JPEG" or ext in (".jpg", ".jpeg"):
            out.save(path, format="JPEG", quality=95, subsampling=0, optimize=True)
        elif fmt == "WEBP" or ext == ".webp":
            out.save(path, format="WEBP", quality=90, method=6)
        elif fmt == "PNG" or ext == ".png":
            out.save(path, format="PNG", optimize=True)
        else:
            out.save(path)
        return True
    except Exception:
        return False
