from enum import Enum
from typing import Optional, Tuple

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


class RenderConfig(BaseModel):
    font_path: str = ""
    """Path to font file. Empty string uses bundled defaults."""
    font_size: Optional[int] = None
    """Override font size for all blocks."""
    font_size_offset: int = 0
    font_size_minimum: int = -1
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


class Config(BaseModel):
    detector: DetectorConfig = DetectorConfig()
    ocr: OcrConfig = OcrConfig()
    inpainter: InpainterConfig = InpainterConfig()
    translator: TranslatorConfig = TranslatorConfig()
    render: RenderConfig = RenderConfig()

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
