"""Step 2: read pages.json, call the LLM, write translations back.

The user is encouraged to open pages.json afterwards and edit the
``translation`` fields by hand before running render.

This step iterates over all task subdirectories under the given work_dir
and translates each independently.
"""
from __future__ import annotations

import os
import json
import asyncio
import re
import unicodedata
from typing import List, Optional

from ..config import Config, LLMProvider
from ..translators import TranslationItem, build_translator
from ..utils import get_logger
from .schema import (
    Block, Page, Workspace, Translation,
    discover_tasks, load_workspace, load_translations, save_translations, get_translations_dir,
    safe_workspace_path
)

logger = get_logger('translate')

# Sentinel: caller did not specify reference_langs, so fall back to config. This is
# distinct from None (auto), [] (off) and an explicit list (manual).
_USE_CONFIG = object()


def _resolve_reference_lang_codes(root: str, target_lang: str,
                                  reference_langs: Optional[List[str]]) -> List[str]:
    """Resolve the reference-langs trichotomy into concrete language codes.

    None  → auto: every other language with a ``<lang>.reviewed`` marker (human-finalised).
    []    → off:  no references.
    [...] → manual: exactly those codes (target excluded), regardless of review state.
    """
    if reference_langs is None:
        tdir = get_translations_dir(root)
        codes: List[str] = []
        if os.path.isdir(tdir):
            for fn in sorted(os.listdir(tdir)):
                if fn.endswith('.reviewed'):
                    code = fn[:-len('.reviewed')]
                    if code and code != target_lang:
                        codes.append(code)
        return codes
    return [c for c in reference_langs if c and c != target_lang]


def _load_reference_maps(root: str, codes: List[str]) -> dict[str, dict[str, str]]:
    """Load each reference language as a read-only {block_id: text} map."""
    maps: dict[str, dict[str, str]] = {}
    for code in codes:
        tr = load_translations(root, code)
        m = {bid: t.text for bid, t in tr.items() if t.text and t.text.strip()}
        if m:
            maps[code] = m
        else:
            logger.warning(f"Reference language '{code}' has no usable translations; ignoring.")
    return maps


def _has_real_text(s: Optional[str]) -> bool:
    """True if the string contains at least one letter or number in any script
    (CJK ideographs and kana count). A string that is only punctuation, symbols
    or whitespace (e.g. "…", "!?", "♪", "") returns False — its source carries no
    translatable content, which is the reliable signal for a broken / missing OCR
    source (hand-added or partial-recognition blocks)."""
    if not s:
        return False
    return any(unicodedata.category(ch)[0] in ('L', 'N') for ch in s)


def _pick_pivot(bid: str, ref_codes: List[str],
                ref_maps: dict[str, dict[str, str]]) -> Optional[tuple[str, str]]:
    """For a block with no usable source, choose an existing translation to
    translate *from* (pivot). Picks by reference priority order: explicit
    --reference-lang order, else the auto/reviewed order. Returns (lang_code, text)
    or None when no reference has this block."""
    for code in ref_codes:
        m = ref_maps.get(code)
        if m and bid in m:
            return code, m[bid]
    return None


def _load_story_description(root_dir: str) -> Optional[str]:
    """Search and load overall story description from text files or metadata."""
    # Look for files: story.txt, script.txt, description.txt, overview.txt, synopsis.txt
    candidates = ["story.txt", "script.txt", "description.txt", "overview.txt", "synopsis.txt"]
    for c in candidates:
        p = os.path.join(root_dir, c)
        if os.path.isfile(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if content:
                    logger.info(f"Found story description in file: {c}")
                    return content
            except Exception as e:
                logger.warning(f"Failed to read story description file {c}: {e}")

    # Check pages.json as fallback
    pages_json_path = os.path.join(root_dir, "pages.json")
    if os.path.isfile(pages_json_path):
        try:
            with open(pages_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for k in ["story", "description", "story_description"]:
                if k in data and data[k]:
                    logger.info(f"Found story description in pages.json under key '{k}'")
                    return str(data[k]).strip()
        except Exception:
            pass

    return None


def _build_story_script(workspace: Workspace, pages: Optional[List[Page]] = None) -> str:
    lines = []
    for page in pages or workspace.pages:
        if page.no_text:
            continue
        page_lines = [b.text.strip() for b in page.blocks if b.text and b.text.strip()]
        if page_lines:
            lines.append(f"--- Page {page.index} ---")
            lines.extend(page_lines)
    return "\n".join(lines)


def _page_source_text(page: Page) -> str:
    return "\n".join(b.text.strip() for b in page.blocks if b.text and b.text.strip())


def _guess_chapter_starts(workspace: Workspace) -> List[int]:
    starts = [0]
    title_re = re.compile(
        r"(第\s*[0-9０-９一二三四五六七八九十百零〇]+\s*[话話章回]|chapter\s*\d+|ch\.?\s*\d+|episode\s*\d+)",
        re.IGNORECASE,
    )
    for i, page in enumerate(workspace.pages[1:], 1):
        haystack = f"{page.name}\n{page.original}\n{_page_source_text(page)[:300]}"
        if title_re.search(haystack):
            starts.append(i)
    return starts


def _chapter_ranges(workspace: Workspace) -> tuple[List[dict], bool]:
    manual = [i for i, page in enumerate(workspace.pages) if page.chapter_start]
    starts = manual or (_guess_chapter_starts(workspace) if len(workspace.pages) >= 30 else [0])
    if 0 not in starts:
        starts.insert(0, 0)
    starts = sorted(set(i for i in starts if 0 <= i < len(workspace.pages)))
    chapters = []
    for n, start in enumerate(starts, 1):
        end = starts[n] if n < len(starts) else len(workspace.pages)
        page = workspace.pages[start]
        chapters.append({"name": page.chapter_name.strip() or f"CH{n}", "pages": workspace.pages[start:end]})
    return chapters, bool(manual)


def _chapter_story_path(workspace: Workspace, name: str) -> str:
    safe = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("._") or "chapter"
    return os.path.join(workspace.root, "story", f"{safe[:80]}.txt")


def _load_text_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text or None
    except Exception as e:
        logger.warning(f"Failed to read story context {path}: {e}")
        return None


def _load_review_story_context(workspace: Workspace) -> Optional[str]:
    parts = []
    global_story = _load_story_description(workspace.root)
    if global_story:
        parts.append(global_story)
    story_dir = os.path.join(workspace.root, "story")
    if os.path.isdir(story_dir):
        for name in sorted(os.listdir(story_dir)):
            if not name.lower().endswith(".txt"):
                continue
            text = _load_text_file(os.path.join(story_dir, name))
            if text:
                parts.append(f"[{name}]\n{text}")
    return "\n\n".join(parts) if parts else None


async def _story_contexts_by_chapter(
    workspace: Workspace,
    translator,
    global_story: Optional[str],
) -> tuple[List[dict], dict[int, str]]:
    chapters, has_manual_markers = _chapter_ranges(workspace)
    contexts: dict[int, str] = {}

    if len(chapters) <= 1:
        story = global_story
        if not story:
            story = await _generate_story_description(
                workspace,
                translator,
                story_path=os.path.join(workspace.root, "story.txt"),
            )
        if story:
            for page in workspace.pages:
                contexts[page.index] = story
        return chapters, contexts

    if has_manual_markers:
        logger.info(f"[task: {workspace.task_name}] Using {len(chapters)} user-marked chapter(s).")
    else:
        logger.info(f"[task: {workspace.task_name}] No user chapter markers; using temporary OCR/title guesses for {len(chapters)} chapter(s).")

    for chapter in chapters:
        chapter_story = None
        path = _chapter_story_path(workspace, chapter["name"])
        if has_manual_markers:
            chapter_story = _load_text_file(path)
        if not chapter_story:
            chapter_story = await _generate_story_description(
                workspace,
                translator,
                pages=chapter["pages"],
                story_path=path if has_manual_markers else None,
            )
        parts = [p for p in [global_story, chapter_story] if p]
        if parts:
            for page in chapter["pages"]:
                contexts[page.index] = "\n\n".join(parts)
    return chapters, contexts


def _page_image_path(workspace: Workspace, page: Page) -> str:
    # page.clean comes from client-writable pages.json; confine it to the workspace
    # so a tampered path can't leak arbitrary files to the LLM via vision.
    return safe_workspace_path(workspace.root, page.clean) or ""


async def _generate_story_description(
    workspace: Workspace,
    translator,
    pages: Optional[List[Page]] = None,
    story_path: Optional[str] = None,
) -> Optional[str]:
    if not hasattr(translator, "summarize_story"):
        return None

    script = _build_story_script(workspace, pages)
    if not script:
        return None

    try:
        logger.info(f"[task: {workspace.task_name}] Generating story context from dialogue...")
        story_description = (await translator.summarize_story(script)).strip()
        if not story_description:
            return None
        if story_path:
            os.makedirs(os.path.dirname(story_path), exist_ok=True)
            with open(story_path, "w", encoding="utf-8") as f:
                f.write(story_description)
                f.write("\n")
            logger.info(f"[task: {workspace.task_name}] Auto-generated {os.path.relpath(story_path, workspace.root)}")
        return story_description
    except Exception as e:
        logger.warning(f"[task: {workspace.task_name}] Failed to auto-generate story.txt: {e}")
        return None


async def _run_review_for_task(
    workspace: Workspace,
    translations: dict[str, Translation],
    cfg: Config,
    overwrite: bool = False,
) -> None:
    """Perform review and polish for translated text blocks in the task."""
    reviewed_marker_path = os.path.join(workspace.root, "translations", f"{workspace.target_lang}.reviewed")

    # Check if we should skip review
    if not overwrite and os.path.exists(reviewed_marker_path):
        logger.info(f"[task: {workspace.task_name}] Translation review already completed (marker file exists), skipping.")
        return

    # Check if there are any translations to review
    has_translations = any(t.text for t in translations.values())
    if not has_translations:
        logger.info(f"[task: {workspace.task_name}] No translated content to review, skipping.")
        return

    if cfg.translator.provider == LLMProvider.none:
        logger.warning(f"[task: {workspace.task_name}] Translator provider is 'none'; skipping review step.")
        return

    logger.info(f"[task: {workspace.task_name}] Starting translation review...")

    # Load story description
    story_description = _load_review_story_context(workspace)
    if not story_description:
        logger.warning(f"[task: {workspace.task_name}] No story description file (story.txt / script.txt / etc.) found in workspace. Polishing without specific story details.")

    # Gather items to review in reading order
    review_items = []
    for page in workspace.pages:
        if page.no_text:
            continue
        for block in page.blocks:
            t = translations.get(block.id)
            # Don't send hand-edited translations to the polisher — respect the user's edits.
            if t and t.text and not t.edited:
                review_items.append((block.id, block.text, t.text))

    if not review_items:
        logger.info(f"[task: {workspace.task_name}] No items found for review.")
        return

    translator = build_translator(cfg.translator)
    if not hasattr(translator, 'review'):
        logger.warning(f"Translator {translator.__class__.__name__} does not support review. Skipping.")
        return

    # Run the review
    polished_map = await translator.review(review_items, story_description or "")

    # Update translations in place
    updated_count = 0
    for bid, polished_text in polished_map.items():
        if bid in translations and translations[bid].text != polished_text:
            translations[bid].text = polished_text
            translations[bid].edited = False
            updated_count += 1

    if updated_count > 0:
        save_translations(workspace.root, workspace.target_lang, translations)
        logger.info(f"[task: {workspace.task_name}] {updated_count} translation(s) polished and updated.")
    else:
        logger.info(f"[task: {workspace.task_name}] Review complete. No translations changed.")

    # Create the marker file to ensure idempotency
    try:
        os.makedirs(os.path.dirname(reviewed_marker_path), exist_ok=True)
        with open(reviewed_marker_path, 'w', encoding='utf-8') as f:
            f.write("reviewed")
        logger.info(f"[task: {workspace.task_name}] Review marker created at {workspace.target_lang}.reviewed")
    except Exception as e:
        logger.warning(f"Failed to create review marker file: {e}")


async def _translate_task(
    workspace: Workspace,
    cfg: Config,
    overwrite: bool = False,
    start_index: Optional[int] = None,
    reference_langs: Optional[List[str]] = None,
) -> Workspace:
    """Translate all blocks in a single task workspace."""
    translator = build_translator(cfg.translator)
    if cfg.translator.provider == LLMProvider.none:
        logger.warning("Translator provider is 'none'; no API calls will be made.")
    story_description = _load_story_description(workspace.root)
    if story_description and hasattr(translator, "set_story_context"):
        translator.set_story_context(story_description)

    # Load existing translations for the target language
    translations = load_translations(workspace.root, workspace.target_lang)

    # Resolve cross-language references (other languages are read-only here).
    ref_codes = _resolve_reference_lang_codes(workspace.root, workspace.target_lang, reference_langs)
    ref_maps = _load_reference_maps(workspace.root, ref_codes) if ref_codes else {}
    if ref_maps:
        logger.info(f"[task: {workspace.task_name}] referencing {list(ref_maps.keys())} "
                    f"→ {workspace.target_lang}")

    # Collect all blocks that need translation across all pages
    all_items: List[TranslationItem] = []
    block_map: dict[str, tuple[Page, Block]] = {}

    for page in workspace.pages:
        # Skip no_text pages — they have no blocks to translate
        if page.no_text:
            continue

        # Logic for start_index and overwrite:
        # 1. If page is before start_index, it's context-only. No translation allowed.
        is_before_start = start_index is not None and page.index < start_index
        if is_before_start:
            page_translations = []
            for b in page.blocks:
                t = translations.get(b.id)
                if t and t.text:
                    page_translations.append(f"{b.text} => {t.text}")
            if page_translations:
                translator.add_context_page(page_translations)
            continue

        # 2. If no overwrite, we can skip fully translated pages
        page_translations = []
        for b in page.blocks:
            t = translations.get(b.id)
            if t and t.text:
                page_translations.append(f"{b.text} => {t.text}")
        is_fully_translated = len(page_translations) == len(page.blocks) and len(page.blocks) > 0
        if not overwrite and is_fully_translated:
            # Fully translated, add to context and skip
            translator.add_context_page(page_translations)
            continue

        # 3. Otherwise, add blocks to the translation queue
        for blk in page.blocks:
            # Skip if already translated and we are not overwriting. Hand-edited
            # translations (edited=True) are preserved even under overwrite.
            t = translations.get(blk.id)
            if t and t.text and (not overwrite or t.edited):
                continue
            refs = {code: m[blk.id] for code, m in ref_maps.items() if blk.id in m}
            if not _has_real_text(blk.text):
                # Source is empty / symbol-only (hand-added or partial-recognition
                # block). If any reference language has this block, translate FROM
                # that translation (pivot) instead of an empty source; the chosen
                # pivot language is dropped from refs (it's now the source). With no
                # pivot available, fall through to the normal path (translates the
                # raw source as before — e.g. a "…" block stays "…").
                pivot = _pick_pivot(blk.id, ref_codes, ref_maps)
                if pivot is not None:
                    pcode, ptext = pivot
                    refs.pop(pcode, None)
                    all_items.append(TranslationItem(id=blk.id, text=ptext,
                                                     references=refs, pivot_lang=pcode,
                                                     page_index=page.index,
                                                     image_path=_page_image_path(workspace, page)))
                    block_map[blk.id] = (page, blk)
                    continue
            all_items.append(TranslationItem(id=blk.id, text=blk.text, references=refs,
                                             page_index=page.index,
                                             image_path=_page_image_path(workspace, page)))
            block_map[blk.id] = (page, blk)

    if all_items:
        chapters, story_contexts = await _story_contexts_by_chapter(workspace, translator, story_description)

        total_blocks = sum(len(p.blocks) for p in workspace.pages)
        no_text_pages = sum(1 for p in workspace.pages if p.no_text)
        logger.info(f"[task: {workspace.task_name}] {len(workspace.pages)} page(s) "
                    f"({no_text_pages} no-text), {total_blocks} total block(s)")
        logger.info(f"[task: {workspace.task_name}] Queued for translation: "
                    f"{len(all_items)} block(s) → {workspace.target_lang}")

        # Perform translation
        translated_ids = set()
        for chapter in chapters:
            page_indexes = {p.index for p in chapter["pages"]}
            chapter_items = [item for item in all_items if item.page_index in page_indexes]
            if not chapter_items:
                continue
            if hasattr(translator, "set_story_context"):
                translator.set_story_context(story_contexts.get(chapter_items[0].page_index, ""))
            logger.info(f"[task: {workspace.task_name}] Translating {chapter['name']}: {len(chapter_items)} block(s)")
            await translator.translate(chapter_items)
            translated_ids.update(item.id for item in chapter_items)
        remaining_items = [item for item in all_items if item.id not in translated_ids]
        if remaining_items:
            if hasattr(translator, "set_story_context"):
                translator.set_story_context(story_description or "")
            await translator.translate(remaining_items)

        # Write back translations
        for item in all_items:
            if item.translation:
                translations[item.id] = Translation(text=item.translation, edited=False)

        # Final save
        save_translations(workspace.root, workspace.target_lang, translations)
        logger.info(f"[task: {workspace.task_name}] Translations written to {workspace.target_lang}.json")
    else:
        logger.info(f"[task: {workspace.task_name}] All blocks already translated, skipping translation phase.")

    # Perform translation review
    await _run_review_for_task(workspace, translations, cfg, overwrite=overwrite)
    return workspace


async def run_translate(
    work_dir: str,
    cfg: Config,
    overwrite: bool = False,
    target_lang: Optional[str] = None,
    start_index: Optional[int] = None,
    reference_langs=_USE_CONFIG,
    concurrency: Optional[int] = None,
) -> List[Workspace]:
    """Translate all tasks under work_dir.

    reference_langs: None = auto (all reviewed languages), [] = off, [codes] = manual.
    Left unset (sentinel) → fall back to cfg.translator.reference_langs.

    Returns a list of updated Workspace objects.
    """
    if reference_langs is _USE_CONFIG:
        reference_langs = cfg.translator.reference_langs

    work_dir = os.path.abspath(os.path.expanduser(work_dir))
    tasks = discover_tasks(work_dir)

    if not tasks:
        raise FileNotFoundError(f"No task subdirectories found under {work_dir}")

    logger.info(f"Found {len(tasks)} task(s) to translate: {tasks}")

    # Concurrency is per TASK, never within a task: each task builds its own
    # translator and keeps its full cross-page context, so parallel tasks don't
    # split or interleave any single work's context. CLI -j overrides the config.
    conc = concurrency if concurrency is not None else getattr(cfg.translator, "concurrency", 1)
    conc = max(1, int(conc or 1))

    async def _run_one(task_name: str) -> Optional[Workspace]:
        task_dir = os.path.join(work_dir, task_name)
        try:
            workspace = load_workspace(task_dir)
        except FileNotFoundError:
            logger.warning(f"[task: {task_name}] No pages.json found, skipping.")
            return None
        # Per-task copy of the config so concurrent tasks never race on the shared
        # cfg.translator.target_lang (each work may target a different language).
        task_cfg = cfg.model_copy(deep=True)
        if target_lang:
            workspace.target_lang = target_lang
            task_cfg.translator.target_lang = target_lang
        else:
            task_cfg.translator.target_lang = workspace.target_lang
        return await _translate_task(workspace, task_cfg, overwrite=overwrite,
                                     start_index=start_index, reference_langs=reference_langs)

    results: List[Workspace] = []
    if conc == 1:
        # Sequential (default) — unchanged behavior, one task at a time.
        for task_name in tasks:
            ws = await _run_one(task_name)
            if ws is not None:
                results.append(ws)
    else:
        logger.info(f"Translating up to {conc} task(s) concurrently")
        sem = asyncio.Semaphore(conc)

        async def _worker(name: str) -> Optional[Workspace]:
            async with sem:
                try:
                    return await _run_one(name)
                except Exception as e:
                    # Isolate failures: one task erroring out must not abort the rest.
                    logger.error(f"[task: {name}] translation failed: {e.__class__.__name__}: {e}")
                    return None

        gathered = await asyncio.gather(*(_worker(n) for n in tasks))
        results = [ws for ws in gathered if ws is not None]

    logger.info(f"Translation complete for {len(results)} task(s).")
    return results
