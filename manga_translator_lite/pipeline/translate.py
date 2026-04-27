"""Step 2: read pages.json, call the LLM, write translations back.

The user is encouraged to open pages.json afterwards and edit the
``translation`` fields by hand before running render.

This step iterates over all task subdirectories under the given work_dir
and translates each independently.
"""
from __future__ import annotations

import os
from typing import List, Optional

from ..config import Config, LLMProvider
from ..translators import LLMTranslator, NoneTranslator, TranslationItem, build_translator
from ..utils import get_logger
from .schema import Block, Workspace, discover_tasks, load_workspace, save_workspace

logger = get_logger('translate')


async def _translate_task(workspace: Workspace, cfg: Config, overwrite: bool = False) -> Workspace:
    """Translate all blocks in a single task workspace."""
    translator = build_translator(cfg.translator)
    if cfg.translator.provider == LLMProvider.none:
        logger.warning("Translator provider is 'none'; no API calls will be made.")

    # Collect all blocks that need translation across all pages
    all_items: List[TranslationItem] = []
    block_map: dict[str, Block] = {}

    for page in workspace.pages:
        # Skip no_text pages — they have no blocks to translate
        if page.no_text:
            continue
        for blk in page.blocks:
            if blk.translation and not overwrite:
                continue
            all_items.append(TranslationItem(id=blk.id, text=blk.text))
            block_map[blk.id] = blk

    if not all_items:
        logger.info(f"[task: {workspace.task_name}] All blocks already translated, skipping.")
        return workspace

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
        blk = block_map.get(item.id)
        if blk:
            blk.translation = item.translation

    # Update rolling context for future runs (per page)
    for page in workspace.pages:
        page_translations = [b.translation for b in page.blocks if b.translation]
        if page_translations:
            translator.add_context_page(page_translations)

    # Final save
    save_workspace(workspace)
    logger.info(f"[task: {workspace.task_name}] Translations written: {workspace.pages_json_path}")
    logger.info("Open pages.json to review/edit translations before running render.")
    return workspace


async def run_translate(
    work_dir: str,
    cfg: Config,
    overwrite: bool = False,
    target_lang: Optional[str] = None,
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

        ws = await _translate_task(workspace, cfg, overwrite=overwrite)
        results.append(ws)

    logger.info(f"Translation complete for {len(results)} task(s).")
    return results
