"""Step 2: read pages.json, call the LLM, write translations back.

The user is encouraged to open pages.json afterwards and edit the
``translation`` fields by hand before running render.

This step iterates over all task subdirectories under the given work_dir
and translates each independently.
"""
from __future__ import annotations

import os
import json
from typing import List, Optional

from ..config import Config, LLMProvider
from ..translators import LLMTranslator, NoneTranslator, TranslationItem, build_translator
from ..utils import get_logger
from .schema import (
    Block, Page, Workspace, Translation,
    discover_tasks, load_workspace, save_workspace,
    load_translations, save_translations
)

logger = get_logger('translate')


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
    story_description = _load_story_description(workspace.root)
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
) -> Workspace:
    """Translate all blocks in a single task workspace."""
    translator = build_translator(cfg.translator)
    if cfg.translator.provider == LLMProvider.none:
        logger.warning("Translator provider is 'none'; no API calls will be made.")

    # Load existing translations for the target language
    translations = load_translations(workspace.root, workspace.target_lang)

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
                    page_translations.append(t.text)
            if page_translations:
                translator.add_context_page(page_translations)
            continue

        # 2. If no overwrite, we can skip fully translated pages
        page_translations = []
        for b in page.blocks:
            t = translations.get(b.id)
            if t and t.text:
                page_translations.append(t.text)
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
            all_items.append(TranslationItem(id=blk.id, text=blk.text))
            block_map[blk.id] = (page, blk)

    if all_items:
        total_blocks = sum(len(p.blocks) for p in workspace.pages)
        no_text_pages = sum(1 for p in workspace.pages if p.no_text)
        logger.info(f"[task: {workspace.task_name}] {len(workspace.pages)} page(s) "
                    f"({no_text_pages} no-text), {total_blocks} total block(s)")
        logger.info(f"[task: {workspace.task_name}] Queued for translation: "
                    f"{len(all_items)} block(s) → {workspace.target_lang}")

        # Perform translation
        await translator.translate(all_items)

        # Write back translations
        for item in all_items:
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
) -> List[Workspace]:
    """Translate all tasks under work_dir.

    Returns a list of updated Workspace objects.
    """
    work_dir = os.path.abspath(os.path.expanduser(work_dir))
    tasks = discover_tasks(work_dir)

    if not tasks:
        raise FileNotFoundError(f"No task subdirectories found under {work_dir}")

    logger.info(f"Found {len(tasks)} task(s) to translate: {tasks}")
    results: List[Workspace] = []

    for task_name in tasks:
        task_dir = os.path.join(work_dir, task_name)
        try:
            workspace = load_workspace(task_dir)
        except FileNotFoundError:
            logger.warning(f"[task: {task_name}] No pages.json found, skipping.")
            continue

        if target_lang:
            workspace.target_lang = target_lang
            cfg.translator.target_lang = target_lang
        else:
            cfg.translator.target_lang = workspace.target_lang

        ws = await _translate_task(workspace, cfg, overwrite=overwrite, start_index=start_index)
        results.append(ws)

    logger.info(f"Translation complete for {len(results)} task(s).")
    return results
