"""Unified LLM translator.

Sends batches of numbered text blocks to a third-party LLM API and returns
the translated text aligned with the input order. Supports OpenAI-compatible
HTTP endpoints (OpenAI, DeepSeek, OpenRouter, Groq, local Ollama, ...) and
Google Gemini.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence
from rich.table import Table
from rich.console import Console

console = Console()

from ..config import LLMProvider, TranslatorConfig
from ..utils import get_logger
from .common import (
    InvalidServerResponse,
    MissingAPIKeyException,
    VALID_LANGUAGES,
)
from .keys import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_HTTP_PROXY,
    OPENAI_MODEL,
)

logger = get_logger('llm')

SYSTEM_PROMPT = (
    "You are a professional manga / comic translator. Translate the provided "
    "lines into {to_lang}. Preserve the meaning, tone, and casual register of "
    "the original. Keep onomatopoeia recognisable in the target language. "
    "Do not censor, omit, or add commentary."
)

USER_PROMPT_HEADER = (
    "Translate the following numbered lines into {to_lang}. "
    "Reply with the translations only, in the same order, in the form "
    "<|i|>translation, one per line. Do not add explanations."
)


@dataclass
class TranslationItem:
    """One block of source text with a stable id."""
    id: str
    text: str
    translation: str = ""


@dataclass
class TranslationBatch:
    items: List[TranslationItem] = field(default_factory=list)
    char_count: int = 0


def _normalise_lang(lang: str) -> str:
    if lang in VALID_LANGUAGES:
        return VALID_LANGUAGES[lang]
    return lang


def make_batches(
    items: Sequence[TranslationItem],
    batch_chars: int,
) -> List[TranslationBatch]:
    """Split items into batches whose summed text length stays near batch_chars.

    A batch never goes empty; a single item that exceeds the limit is sent on
    its own.
    """
    batches: List[TranslationBatch] = []
    current = TranslationBatch()
    for item in items:
        item_len = len(item.text) + 8  # + tag overhead
        if current.items and current.char_count + item_len > batch_chars:
            batches.append(current)
            current = TranslationBatch()
        current.items.append(item)
        current.char_count += item_len
    if current.items:
        batches.append(current)
    return batches


def _build_prompt(
    items: Sequence[TranslationItem],
    to_lang_human: str,
    context: Optional[str] = None,
    extra_instructions: Optional[str] = None,
) -> str:
    parts = [USER_PROMPT_HEADER.format(to_lang=to_lang_human)]
    if extra_instructions:
        parts.append(extra_instructions.strip())
    if context:
        parts.append("Recent translated context (for tone reference only, do not retranslate):")
        parts.append(context.strip())
    parts.append("Lines to translate:")
    for i, item in enumerate(items, 1):
        parts.append(f"<|{i}|>{item.text}")
    return "\n".join(parts)


def _parse_response(text: str, count: int) -> List[str]:
    """Parse <|i|>... blocks from an LLM response."""
    pieces = re.split(r"<\|(\d+)\|>", text)
    # split returns [pre, idx1, content1, idx2, content2, ...]
    out: dict[int, str] = {}
    for i in range(1, len(pieces) - 1, 2):
        try:
            idx = int(pieces[i])
        except ValueError:
            continue
        out[idx] = pieces[i + 1].strip().rstrip()
    if not out:
        # The model may have returned plain lines; try one-per-line fallback.
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if len(lines) == count:
            return lines
        raise InvalidServerResponse(
            f"Could not parse <|i|> entries from response (got {len(lines)} lines, expected {count})."
        )
    return [out.get(i, "") for i in range(1, count + 1)]


class LLMTranslator:
    """Drives an LLM endpoint to translate a list of TranslationItems."""

    def __init__(self, cfg: TranslatorConfig):
        self.cfg = cfg
        self.target_human = _normalise_lang(cfg.target_lang)
        self._context_pages: List[List[str]] = []  # translated lines per past page

    # ---- Public API ----

    def add_context_page(self, translated_lines: Iterable[str]) -> None:
        lines = [ln.strip() for ln in translated_lines if ln and ln.strip()]
        if lines:
            self._context_pages.append(lines)
        # keep only the most recent N pages
        n = max(0, self.cfg.context_pages)
        if len(self._context_pages) > n:
            self._context_pages = self._context_pages[-n:]

    async def translate(self, items: Sequence[TranslationItem]) -> None:
        """Translate items in place by batching them."""
        if not items:
            return
        if self.cfg.provider == LLMProvider.none:
            return
        batches = make_batches(items, self.cfg.batch_chars)
        logger.info(
            f"Translating {len(items)} blocks in {len(batches)} batch(es) "
            f"(~{self.cfg.batch_chars} chars each) to {self.target_human}"
        )
        for batch_no, batch in enumerate(batches, 1):
            logger.info(f"Batch {batch_no}/{len(batches)}: {len(batch.items)} blocks, ~{batch.char_count} chars")
            try:
                translations = await self._request(batch, len(batch.items))

                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Source Text", style="cyan", width=50)
                table.add_column(f"Translation ({self.target_human})", style="green")

                for item, trans in zip(batch.items, translations):
                    item.translation = trans
                    table.add_row(item.text.replace("\n", " "), trans.replace("\n", " "))

                console.print(table)
                print() # Add a newline after the table
            except InvalidServerResponse as e:
                logger.error(f"Batch {batch_no} failed after {self.cfg.max_retries} attempts: {e}. Skipping this batch.")
                continue

    # ---- Provider dispatch ----

    async def _request(self, batch: TranslationBatch, expected: int) -> List[str]:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            ctx = self._context_text()
            prompt = _build_prompt(
                batch.items,
                self.target_human,
                context=ctx,
                extra_instructions=self.cfg.extra_instructions,
            )
            try:
                if self.cfg.provider == LLMProvider.openai:
                    text = await self._request_openai(prompt)
                elif self.cfg.provider == LLMProvider.gemini:
                    text = await self._request_gemini(prompt)
                else:
                    raise ValueError(f"Unsupported provider: {self.cfg.provider}")
                return _parse_response(text, expected)
            except Exception as e:
                last_err = e
                logger.warning(f"LLM request attempt {attempt}/{self.cfg.max_retries} failed: {e}")
                
                if "403" in str(e) or getattr(e, 'status_code', None) == 403 or getattr(e, 'code', None) == 403:
                    logger.warning("Encountered 403 error. Clearing context and retrying...")
                    self._context_pages.clear()

                if attempt < self.cfg.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 10))
        raise InvalidServerResponse(f"All {self.cfg.max_retries} attempts failed: {last_err}")

    async def _request_openai(self, prompt: str) -> str:
        try:
            import openai
        except ImportError as e:
            raise RuntimeError("openai package is required for the openai provider") from e

        api_key = self.cfg.api_key or OPENAI_API_KEY
        if not api_key:
            raise MissingAPIKeyException("API key not set (config.api_key or OPENAI_API_KEY)")
        api_base = self.cfg.api_base or OPENAI_API_BASE
        model = self.cfg.model or OPENAI_MODEL

        client_kwargs = {"api_key": api_key, "base_url": api_base}
        if OPENAI_HTTP_PROXY:
            from httpx import AsyncClient
            client_kwargs["http_client"] = AsyncClient(
                proxies={"all://": f"http://{OPENAI_HTTP_PROXY}"}
            )
        client = openai.AsyncOpenAI(**client_kwargs)

        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(to_lang=self.target_human)},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            ),
            timeout=self.cfg.timeout,
        )
        return resp.choices[0].message.content or ""

    async def _request_gemini(self, prompt: str) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise RuntimeError("google-genai package is required for the gemini provider") from e

        api_key = self.cfg.api_key or GEMINI_API_KEY
        if not api_key:
            raise MissingAPIKeyException("API key not set (config.api_key or GEMINI_API_KEY)")
        model = self.cfg.model or GEMINI_MODEL

        client = genai.Client(api_key=api_key)
        cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT.format(to_lang=self.target_human),
            temperature=0.3,
        )

        def _call() -> str:
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            return resp.text or ""

        return await asyncio.wait_for(asyncio.to_thread(_call), timeout=self.cfg.timeout)

    # ---- Helpers ----

    def _context_text(self) -> Optional[str]:
        if not self._context_pages:
            return None
        chunks: List[str] = []
        for page_idx, page in enumerate(self._context_pages, 1):
            for line in page:
                chunks.append(f"<|p{page_idx}|>{line}")
        return "\n".join(chunks) if chunks else None
