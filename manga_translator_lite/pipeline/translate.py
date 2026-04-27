"""Step 2: read pages.json, call the LLM, write translations back.

The user is encouraged to open pages.json afterwards and edit the
``translation`` fields by hand before running render.
"""
from __future__ import annotations

import os
from typing import Optional

from ..config import Config, LLMProvider
from ..translators import LLMTranslator, NoneTranslator, TranslationItem, build_translator
from ..utils import get_logger
from .schema import Workspace, load_workspace, save_workspace

logger = get_logger('translate')


async def run_translate(
    work_dir: str,
    cfg: Config,
    overwrite: bool = False,
    target_lang: Optional[str] = None,
) -> Workspace:
    work_dir = os.path.abspath(os.path.expanduser(work_dir))
    workspace = load_workspace(work_dir)

    if target_lang:
        workspace.target_lang = target_lang
        cfg.translator.target_lang = target_lang
    else:
        # honour whatever is in the workspace already
        cfg.translator.target_lang = workspace.target_lang

    translator = build_translator(cfg.translator)
    if cfg.translator.provider == LLMProvider.none:
        logger.warning("Translator provider is 'none'; no API calls will be made.")

    # 1. Collect all blocks that need translation across all pages
    all_items: List[TranslationItem] = []
    block_map: dict[str, Block] = {}

    for page in workspace.pages:
        for blk in page.blocks:
            if blk.translation and not overwrite:
                continue
            all_items.append(TranslationItem(id=blk.id, text=blk.text))
            block_map[blk.id] = blk

    if not all_items:
        logger.info("All blocks already translated, skipping.")
        return workspace

    total_blocks = sum(len(p.blocks) for p in workspace.pages)
    logger.info(f"Translating workspace: {len(workspace.pages)} page(s), {total_blocks} total block(s)")
    logger.info(f"Queued for translation: {len(all_items)} block(s) → {workspace.target_lang}")

    # 2. Perform translation (LLMTranslator handles batching internally based on cfg.translator.batch_chars)
    await translator.translate(all_items)

    # 3. Write back translations
    for item in all_items:
        blk = block_map.get(item.id)
        if blk:
            blk.translation = item.translation

    # 4. Update rolling context for future runs (per page)
    for page in workspace.pages:
        page_translations = [b.translation for b in page.blocks if b.translation]
        if page_translations:
            translator.add_context_page(page_translations)

    # Final save
    save_workspace(workspace)
    logger.info(f"Translations written: {workspace.pages_json_path}")
    logger.info("Open pages.json to review/edit translations before running render.")
    return workspace
