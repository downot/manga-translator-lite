"""Shared, transport-agnostic building blocks for the editor servers.

This module is part of the open-source manga-translator-lite project and lives at
the repo root (next to ``server.py``) ON PURPOSE:

  * ``server.py`` must be able to start the editor WITHOUT importing the
    ``manga_translator_lite`` package (whose ``__init__`` pulls heavy ML deps).
    So this module imports ONLY the standard library at module load; anything
    heavier (the pipeline package, Pillow) is imported lazily inside functions.
  * Because it is a plain top-level module — not a submodule of the package —
    ``import server_common`` never triggers ``manga_translator_lite/__init__``.

It is the shared base for the MTL editor backends — path-safety helpers, config
reading, thumbnailing, save-payload validation, and the pipeline command dispatch.
It is MTL-aware (it knows the task/workspace/pipeline shape) but must stay free of
DEPLOYMENT-specific concerns (auth, sessions, access control, branding) so that
downstream/custom servers (e.g. an authenticated commercial variant) can reuse it
without inheriting that project's customizations.
"""
from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Callable, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------
# Server-internal files that must never be served/overwritten via task paths even
# when they happen to fall inside an allowed base directory.
PROTECTED_NAMES = {"sessions.json", "access.json", ".env", ".secret"}


def confine(path_str, base) -> Optional[Path]:
    """Return ``path_str`` resolved inside ``base``, or None if it escapes ``base``.

    ``resolve()`` collapses ``..`` and follows symlinks before the containment
    check, so neither traversal nor a symlink pointing outside ``base`` can escape.
    """
    if not path_str:
        return None
    try:
        base = Path(base).resolve()
        p = Path(path_str)
        target = (p if p.is_absolute() else base / p).resolve()
        target.relative_to(base)
        return target
    except Exception:
        return None


def resolve_within(path_str, allowed_bases) -> Optional[Path]:
    """Resolve ``path_str`` and return the Path only if it lives inside one of
    ``allowed_bases`` and is not a protected server file; otherwise None."""
    if not path_str:
        return None
    try:
        target = Path(path_str).resolve()
    except Exception:
        return None
    if target.name in PROTECTED_NAMES:
        return None
    for base in allowed_bases:
        try:
            target.relative_to(Path(base).resolve())
            return target
        except Exception:
            continue
    return None


def is_safe_lang(lang: str) -> bool:
    """Language code is alphanumerics/hyphen/underscore only.

    Prevents directory traversal through the ``lang`` query parameter (it is used
    to build ``translations/<lang>.json`` paths).
    """
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", lang or ""))


# ---------------------------------------------------------------------------
# Save-payload validation
# ---------------------------------------------------------------------------
# /api/save writes client JSON straight to pages.json / translations/<lang>.json.
# Validate at the WRITE boundary so malformed or hostile data never lands on disk
# (where it would otherwise surface as a 500 deep in render/extract, or — for path
# fields — as a read/write outside the workspace). Checks only the security/robustness
# invariants the editor always satisfies; unknown/extra fields are left alone so the
# editor's schema can evolve without server changes.
MAX_BLOCK_TEXT = 50_000   # generous per-block text cap (a bubble is ~hundreds of chars)


class PayloadError(ValueError):
    """A save payload failed validation; carries a short, client-safe message."""


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _safe_name_field(s) -> bool:
    """A bare-filename field (page.name / page.original): a string with no path
    separator or ``..`` traversal — it must not be able to steer a write elsewhere.
    Empty/"." are allowed (harmless, non-traversal; render basenames them anyway)."""
    return isinstance(s, str) and s != ".." \
        and "/" not in s and "\\" not in s and "\x00" not in s


def _safe_relpath_field(s) -> bool:
    """A workspace-relative path field (page.clean): a string that is not absolute,
    has no ``..`` segment, and no NUL byte. May contain a sub-directory (e.g. clean/)."""
    if not isinstance(s, str) or not s or "\x00" in s:
        return False
    norm = s.replace("\\", "/")
    if norm.startswith("/"):
        return False
    return ".." not in norm.split("/")


def _validate_block(b, where):
    if not isinstance(b, dict):
        raise PayloadError(f"{where} must be an object")
    text = b.get("text", "")
    if not isinstance(text, str):
        raise PayloadError(f"{where}.text must be a string")
    if len(text) > MAX_BLOCK_TEXT:
        raise PayloadError(f"{where}.text exceeds {MAX_BLOCK_TEXT} chars")
    bbox = b.get("bbox")
    if bbox is not None and not (isinstance(bbox, list) and len(bbox) == 4 and all(_is_num(v) for v in bbox)):
        raise PayloadError(f"{where}.bbox must be 4 numbers")
    poly = b.get("polygon")
    if poly is not None:
        if not isinstance(poly, list) or not all(
                isinstance(pt, list) and len(pt) == 2 and all(_is_num(v) for v in pt) for pt in poly):
            raise PayloadError(f"{where}.polygon must be a list of [x, y] points")


def validate_pages_payload(data) -> None:
    """Validate a pages.json save body. Raises :class:`PayloadError` on the first
    problem (path-field escape, malformed geometry, oversized text)."""
    if not isinstance(data, dict):
        raise PayloadError("pages.json must be a JSON object")
    pages = data.get("pages", [])
    if not isinstance(pages, list):
        raise PayloadError("pages must be a list")
    for i, page in enumerate(pages):
        if not isinstance(page, dict):
            raise PayloadError(f"page[{i}] must be an object")
        for f in ("name", "original"):
            if f in page and not _safe_name_field(page[f]):
                raise PayloadError(f"page[{i}].{f} is not a safe filename")
        if page.get("clean") and not _safe_relpath_field(page["clean"]):
            raise PayloadError(f"page[{i}].clean escapes the workspace")
        blocks = page.get("blocks", [])
        if not isinstance(blocks, list):
            raise PayloadError(f"page[{i}].blocks must be a list")
        for j, b in enumerate(blocks):
            _validate_block(b, f"page[{i}].block[{j}]")


def validate_translations_payload(data) -> None:
    """Validate a translations/<lang>.json save body — either the legacy whole-object
    form ``{bid: {text, edited}}`` or the collab2 incremental form
    ``{changes: {bid: {text,...}}, deletes: [bid,...]}``. Raises :class:`PayloadError`."""
    if not isinstance(data, dict):
        raise PayloadError("translations must be a JSON object")
    if "changes" in data or "deletes" in data:          # collab2 incremental save
        changes = data.get("changes", {})
        if not isinstance(changes, dict):
            raise PayloadError("changes must be an object")
        for bid, ch in changes.items():
            if not isinstance(ch, dict):
                raise PayloadError(f"changes[{bid}] must be an object")
            t = ch.get("text", "")
            if not isinstance(t, str) or len(t) > MAX_BLOCK_TEXT:
                raise PayloadError(f"changes[{bid}].text is invalid or too large")
        if not isinstance(data.get("deletes", []), list):
            raise PayloadError("deletes must be a list")
        return
    for bid, t in data.items():                          # legacy whole-object save
        txt = t.get("text", "") if isinstance(t, dict) else t
        if not isinstance(txt, str) or len(txt) > MAX_BLOCK_TEXT:
            raise PayloadError(f"translation[{bid}] is invalid or too large")


# ---------------------------------------------------------------------------
# Config / pipeline availability
# ---------------------------------------------------------------------------
def pipeline_available() -> bool:
    """Whether the pipeline package is importable *by location*.

    Cheap presence check via ``find_spec`` so we DON'T import its heavy ML deps
    just to probe whether the editor can offer the pipeline tab.
    """
    try:
        import importlib.util
        return importlib.util.find_spec("manga_translator_lite") is not None
    except Exception:
        return False


def read_config_as_json(path: str) -> str:
    """Return the config file as a JSON string for the editor's render preview.

    Prefers ``manga_translator_lite.Config`` (applies defaults/validation); falls
    back to a plain stdlib TOML/JSON parse so the editor works without the package.
    Never raises — returns ``"{}"`` if nothing can be read (editor uses its defaults).
    """
    try:
        from manga_translator_lite.config import Config
        cfg = Config.load(path)
        return cfg.model_dump_json() if hasattr(cfg, "model_dump_json") else cfg.json()
    except Exception:
        pass
    try:
        ext = os.path.splitext(path)[1].lower()
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if ext == ".toml":
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            return json.dumps(tomllib.loads(content))
        return json.dumps(json.loads(content))
    except Exception:
        return "{}"


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------
def build_thumbnail(src: Path, cache_dir: Path, box: int, quality: int = 72,
                    logger=None) -> Optional[Path]:
    """Build (or reuse a cached) JPEG thumbnail of ``src`` in ``cache_dir``.

    Returns the path of the thumbnail to serve, or None if Pillow is absent or the
    image can't be decoded — the caller should then serve the original image so the
    editor keeps working (just without the bandwidth savings) on a bare install.

    The cache is regenerated when missing or older than the source, and written via
    a temp file + atomic rename so a concurrent request never reads a half-written file.
    """
    try:
        from PIL import Image
        cache_path = cache_dir / (src.name + ".jpg")
        if (not cache_path.exists()) or cache_path.stat().st_mtime < src.stat().st_mtime:
            cache_dir.mkdir(exist_ok=True)
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((box, box))
                import tempfile
                fd, tmp = tempfile.mkstemp(dir=str(cache_dir), suffix=".tmp")
                os.close(fd)
                im.save(tmp, "JPEG", quality=quality)
                os.replace(tmp, cache_path)
        return cache_path
    except Exception as e:
        if logger:
            logger.info(f"Thumbnail fallback for {src.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Pipeline dispatch
# ---------------------------------------------------------------------------
class PipelinePathError(ValueError):
    """Raised when a client-supplied pipeline path fails confinement.

    Carries a short, client-safe message (``"Invalid input path"`` etc.); each
    server maps it onto its own transport's error response.
    """


def parse_reference_langs(raw):
    """Normalize the cross-language reference argument.

    ``'auto'`` (default) -> None (all reviewed langs), ``'off'`` -> [] (no
    reference), an explicit list -> that list (manual selection).
    """
    if raw == "off":
        return []
    return raw if isinstance(raw, list) else None


def resolve_pipeline_paths(input_path, output_path, config_path, task_path, root_dir):
    """Validate & normalize the client-supplied pipeline paths.

    ``input``/``output`` are confined to the task directory; ``config`` may also
    live under ``root_dir`` (the project's shared config files). Returns the
    normalized ``(input_path, output_path, config_path)`` tuple, or raises
    :class:`PipelinePathError` with a client-safe message on the first bad path.
    """
    if input_path:
        safe = confine(input_path, task_path)
        if not safe:
            raise PipelinePathError("Invalid input path")
        input_path = str(safe)
    if output_path:
        safe = confine(output_path, task_path)
        if not safe:
            raise PipelinePathError("Invalid output path")
        output_path = str(safe)
    if config_path:
        p = Path(config_path)
        if not p.is_absolute():
            p = Path(root_dir) / p
        safe_cfg = resolve_within(str(p), [root_dir, task_path])
        if not safe_cfg or not safe_cfg.is_file():
            raise PipelinePathError("Invalid config path")
        config_path = str(safe_cfg)
    return input_path, output_path, config_path


async def run_pipeline(*, cmd, task_path, log: Callable[[str, str], None],
                       config_path=None, target_lang=None, overwrite=False,
                       start_index=None, reference_langs=None,
                       input_path=None, output_path=None) -> None:
    """Run one pipeline command (``extract`` / ``translate`` / ``render`` / ``run``).

    Transport-agnostic: progress/results are reported only through ``log(kind, msg)``
    where ``kind`` is ``"status"`` or ``"error"`` (the per-line logging stream is
    wired up by the caller, since that part is concurrency-model specific). Paths are
    expected to already be validated via :func:`resolve_pipeline_paths`.
    """
    # The pipeline needs the full package — import lazily so the editor itself can
    # start without manga_translator_lite (and its ML deps) installed.
    try:
        from manga_translator_lite.config import Config
        from manga_translator_lite.pipeline.extract import run_extract
        from manga_translator_lite.pipeline.translate import run_translate
        from manga_translator_lite.pipeline.render import run_render
    except ImportError as e:
        log("error", f"Pipeline unavailable: manga_translator_lite is not importable ({e}). "
                     f"The editor runs standalone, but extract/translate/render need the full package.")
        return

    cfg = Config.load(config_path or None)
    if target_lang:
        cfg.translator.target_lang = target_lang

    try:
        if cmd == "extract":
            await run_extract(input_path or str(task_path / "in"), task_path, cfg, overwrite=overwrite)
        elif cmd == "translate":
            await run_translate(task_path, cfg, overwrite=overwrite, target_lang=target_lang,
                                start_index=start_index, reference_langs=reference_langs)
        elif cmd == "render":
            await run_render(task_path, output_path or str(task_path / "out"), cfg)
        elif cmd == "run":
            await run_extract(input_path or str(task_path / "in"), task_path, cfg, overwrite=overwrite)
            await run_translate(task_path, cfg, overwrite=overwrite, target_lang=target_lang,
                                reference_langs=reference_langs)
            await run_render(task_path, output_path or str(task_path / "out"), cfg)
        log("status", "--- Pipeline Finished ---")
    except Exception as e:
        log("error", f"Error: {str(e)}")
