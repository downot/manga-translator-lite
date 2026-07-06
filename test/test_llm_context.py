import asyncio

from manga_translator_lite.config import LLMProvider, TranslatorConfig
from manga_translator_lite.translators.llm import (
    LLMTranslator,
    TranslationItem,
    _build_prompt,
    _parse_response,
    make_batches,
)


def test_build_prompt_includes_story_context():
    prompt = _build_prompt(
        [TranslationItem(id="p0001_b000", text="やあ", page_index=1)],
        "English",
        story_context="A quiet school romance with formal senpai/kouhai speech.",
    )

    assert "Overall story context" in prompt
    assert "formal senpai/kouhai speech" in prompt
    assert "--- Page 1 ---" in prompt
    assert "<|p0001_b000|>やあ" in prompt


def test_parse_response_uses_block_ids():
    parsed = _parse_response(
        "<|p0001_b001|>Second\n<|p0001_b000|>First",
        ["p0001_b000", "p0001_b001"],
    )

    assert parsed == ["First", "Second"]


def test_make_batches_prefers_page_boundary_near_limit():
    items = [
        TranslationItem(id="p0001_b000", text="abcd", page_index=1),
        TranslationItem(id="p0002_b000", text="ef", page_index=2),
    ]

    batches = make_batches(items, batch_chars=15)

    assert [[item.id for item in batch.items] for batch in batches] == [
        ["p0001_b000"],
        ["p0002_b000"],
    ]


def test_context_pages_zero_disables_context():
    cfg = TranslatorConfig(
        provider=LLMProvider.openai,
        target_lang="ENG",
        context_pages=0,
    )
    translator = LLMTranslator(cfg)

    translator.add_context_page(["A => T:A"])

    assert translator._context_text() is None


def test_translate_adds_successful_batch_as_source_translation_context():
    class FakeTranslator(LLMTranslator):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.seen_contexts = []

        async def _request(self, batch):
            self.seen_contexts.append(self._context_text())
            return [f"T:{item.text}" for item in batch.items]

    cfg = TranslatorConfig(
        provider=LLMProvider.openai,
        target_lang="ENG",
        batch_chars=9,
        context_pages=1,
    )
    translator = FakeTranslator(cfg)

    items = [
        TranslationItem(id="p0001_b000", text="A"),
        TranslationItem(id="p0001_b001", text="B"),
    ]
    asyncio.run(translator.translate(items))

    assert translator.seen_contexts == [None, "<|p1|>A => T:A"]


def test_summarize_story_uses_dialogue_script():
    class FakeTranslator(LLMTranslator):
        async def _request_openai(self, prompt, system_prompt=None):
            self.prompt = prompt
            self.system_prompt = system_prompt
            return "summary"

    cfg = TranslatorConfig(provider=LLMProvider.openai, target_lang="ENG")
    translator = FakeTranslator(cfg)

    result = asyncio.run(translator.summarize_story("--- Page 1 ---\nやあ"))

    assert result == "summary"
    assert "やあ" in translator.prompt
    assert "manga story analyst" in translator.system_prompt
