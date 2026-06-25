from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel


def hex2rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


class Detector(str, Enum):
    default = "default"
    dbconvnext = "dbconvnext"
    ctd = "ctd"
    craft = "craft"
    paddle = "paddle"
    rtdetr = "rtdetr"      # RT-DETR-v2 comic detector (experimental; box masks — see detection/rtdetr.py)
    none = "none"


class Ocr(str, Enum):
    ocr32px = "32px"
    ocr48px = "48px"
    ocr48px_ctc = "48px_ctc"
    mocr = "mocr"


class Inpainter(str, Enum):
    default = "default"
    lama_large = "lama_large"
    lama_mpe = "lama_mpe"
    none = "none"


class InpaintPrecision(str, Enum):
    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"

    def __str__(self) -> str:
        return self.name


class LLMProvider(str, Enum):
    openai = "openai"          # OpenAI-compatible (chatgpt / deepseek / openrouter / groq / custom)
    gemini = "gemini"          # Google Gemini
    none = "none"              # No translation (passthrough)


class Alignment(str, Enum):
    auto = "auto"
    left = "left"
    center = "center"
    right = "right"


class Direction(str, Enum):
    auto = "auto"
    h = "horizontal"
    v = "vertical"


class DetectorConfig(BaseModel):
    detector: Detector = Detector.default
    detection_size: int = 2048
    text_threshold: float = 0.5
    box_threshold: float = 0.7
    unclip_ratio: float = 2.3
    det_rotate: bool = False
    det_auto_rotate: bool = False
    det_invert: bool = False
    det_gamma_correct: bool = False

    secondary_detector: Detector = Detector.none
    """Optional second detector fused with the primary to boost recall (e.g. 'rtdetr').
    'none' disables fusion (default — behavior is unchanged). Regions the secondary finds
    that the primary misses are added to detection: OCR'd, translated, and box-erased.
    Keep a stroke detector (ctd/default) as the primary so erase masks stay clean."""
    secondary_box_threshold: Optional[float] = None
    """box_threshold for the secondary detector (rtdetr likes ~0.3). None = reuse box_threshold."""
    fusion_iou: float = 0.4
    """A secondary region is 'new' (kept) only if its IoU with every primary region is below this."""
    fusion_overlap_limit: float = 0.5
    """Also drop a secondary region as a duplicate if it covers (or is covered by) at least this
    fraction of any primary region — intersection over the smaller box. Catches a large box-detector
    region sitting on top of small primary text lines (low IoU, but a clear duplicate that would
    otherwise be OCR'd again as a partial copy). Lower it (e.g. 0.3) if duplicates persist."""
    fusion_max_area_ratio: float = 0.1
    """Drop secondary regions whose box covers more than this fraction of the page. A box
    detector (rtdetr) can return one huge box for a stylized title / SFX spanning the art;
    box-filling that into the erase mask would wipe a large area. 0 disables the cap."""


class OcrConfig(BaseModel):
    ocr: Ocr = Ocr.ocr48px
    use_mocr_merge: bool = False
    min_text_length: int = 0
    ignore_bubble: int = 0
    prob: Optional[float] = None


class InpainterConfig(BaseModel):
    inpainter: Inpainter = Inpainter.lama_large
    inpainting_size: int = 2048
    inpainting_precision: InpaintPrecision = InpaintPrecision.bf16


class TranslatorConfig(BaseModel):
    provider: LLMProvider = LLMProvider.openai
    """LLM provider: openai (OpenAI-compatible HTTP), gemini, or none"""
    model: str = "gpt-4o-mini"
    """Model name to use"""
    api_key: Optional[str] = None
    """API key. If unset, falls back to env var (OPENAI_API_KEY / GEMINI_API_KEY)."""
    api_base: Optional[str] = None
    """API base URL. If unset, uses provider default."""
    target_lang: str = "ENG"
    """Destination language code (CHS, CHT, ENG, JPN, KOR, etc.)"""
    source_lang: str = "auto"
    """Source language hint, or 'auto' to let the model detect."""
    batch_chars: int = 1500
    """Approximate character budget per LLM request (1000-2000 recommended)."""
    context_pages: int = 1
    """How many previously translated pages to send as context."""
    timeout: int = 120
    """Request timeout in seconds."""
    max_retries: int = 3
    """Maximum retry attempts per batch."""
    extra_instructions: Optional[str] = None
    """Extra instructions appended to the system prompt (e.g. tone, glossary)."""
    reference_langs: Optional[List[str]] = None
    """Cross-language reference for translation. None = auto (reference every other
    language that has been human-reviewed, i.e. has a .reviewed marker); [] = off
    (no reference); ["CHS", ...] = reference exactly these language codes. The
    referenced languages are only read, never modified."""


class RenderConfig(BaseModel):
    font_path: str = ""
    """Path to font file. Empty string uses bundled defaults."""
    font_size: Optional[int] = None
    """Override font size for all blocks."""
    font_size_offset: int = 0
    font_size_minimum: int = -1
    font_size_minimum_expand_limit: float = 1.5
    font_size_readable_min: int = -1
    """Readability floor (px) for user-sized / fixed boxes. A fixed box never
    auto-expands, so when its translation is long the font shrinks to fit — this
    is the smallest size it may shrink to. Below it the text would be unreadable,
    so rendering stops at this floor and the (now slightly overflowing) block is
    flagged for review in the editor instead of being silently shrunk to a few px.
    ``-1`` = auto (page-relative ≈ (width+height)/300); a positive value is an
    absolute pixel floor. Set a very small value (e.g. ``4``) for the old behavior."""
    line_spacing: Optional[int] = None
    direction: Direction = Direction.auto
    alignment: Alignment = Alignment.auto
    uppercase: bool = False
    lowercase: bool = False
    disable_font_border: bool = False
    no_hyphenation: bool = False
    rtl: bool = True
    fit_to_region: bool = True
    """Whether to shrink font size to fit into the original detected box."""
    font_color: Optional[str] = None
    """Override font color, e.g. 'FFFFFF' or 'FFFFFF:000000' (fg:bg)."""

    @property
    def font_color_fg(self) -> Optional[Tuple[int, int, int]]:
        if not self.font_color:
            return None
        return hex2rgb(self.font_color.split(':')[0])

    @property
    def font_color_bg(self) -> Optional[Tuple[int, int, int]]:
        if not self.font_color or ':' not in self.font_color:
            return None
        return hex2rgb(self.font_color.split(':')[1])


class SignaturePages(str, Enum):
    none = "none"          # never draw
    first = "first"        # only the first page of each work
    last = "last"          # only the last page
    first_last = "first_last"  # first AND last page (default)
    every = "every"        # every page (watermark style)


class SignaturePosition(str, Enum):
    bottom_right = "bottom-right"
    bottom_left = "bottom-left"
    top_right = "top-right"
    top_left = "top-left"


class SignatureConfig(BaseModel):
    """Translator credit baked into the output, plus a low-key mention of this
    open-source system. Rendered as a small horizontal or vertical text box in a
    page corner. Disabled by default; set ``enabled = true`` and a ``translator``."""
    enabled: bool = False
    translator: str = ""
    """Translator name / handle shown in the credit."""
    text: str = ""
    """Custom template for the translator line. Placeholder: {translator} (and
    {project} if you want the credit inline). Use \\n for line breaks. Empty →
    default '译者：{translator}'. The fixed open-source project credit
    (MTL.downot.moe) is always appended and cannot be changed or removed."""
    pages: SignaturePages = SignaturePages.first_last
    direction: Direction = Direction.auto
    """horizontal | vertical | auto. auto = vertical for CJK target languages, else horizontal."""
    position: SignaturePosition = SignaturePosition.bottom_right
    font_size: int = -1
    """Translator-name font size in px. -1 = auto (large, page-relative ≈ (W+H)/85).
    The fixed project credit is drawn at ~half this, lighter, and the name overlaps it."""
    color: str = "595959"
    """Hex RGB of the translator name (clear but low-key grey). The project credit is auto-lightened well beyond this so the name stands out."""
    opacity: float = 0.85
    """0–1 overall opacity. The project credit uses about half of this so it recedes."""
    margin: int = -1
    """Inset from the page edge in px. -1 = auto (≈ (width+height)/120)."""
    font_path: str = ""
    """Optional separate font for the signature. Empty → render font / bundled default."""

    @property
    def color_rgb(self) -> Tuple[int, int, int]:
        try:
            return hex2rgb(self.color)
        except Exception:
            return (128, 128, 128)


class Config(BaseModel):
    detector: DetectorConfig = DetectorConfig()
    ocr: OcrConfig = OcrConfig()
    inpainter: InpainterConfig = InpainterConfig()
    translator: TranslatorConfig = TranslatorConfig()
    render: RenderConfig = RenderConfig()
    signature: SignatureConfig = SignatureConfig()

    kernel_size: int = 3
    """Convolution kernel size used to clean up text mask edges."""
    mask_dilation_offset: int = 20
    """How much to extend the text mask before inpainting."""
    force_simple_sort: bool = False
    """Skip panel-aware sort and use a simpler top-to-bottom / RTL sort."""
    use_gpu: bool = False
    """Use CUDA / MPS for OCR/detection/inpainting if available."""

    @classmethod
    def load(cls, path: Optional[str]) -> "Config":
        if not path:
            from pathlib import Path
            pkg_root = Path(__file__).parent.parent.absolute()
            toml_path = pkg_root / "config.toml"
            json_path = pkg_root / "config.json"
            if toml_path.is_file():
                path = str(toml_path)
            elif json_path.is_file():
                path = str(json_path)
            else:
                return cls()
        import os
        ext = os.path.splitext(path)[1].lower()
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        if ext == '.toml':
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            data = tomllib.loads(content)
        elif ext == '.json':
            import json
            data = json.loads(content)
        else:
            raise ValueError(f"Unsupported config format: {ext}")
        return cls(**data)
