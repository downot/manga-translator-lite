"""Pass-through translator that emits empty translations.

Useful for prepping clean images without spending API credits, or for review
workflows where translations will be filled in entirely by hand.
"""
from typing import Sequence

from .llm import TranslationItem


class NoneTranslator:
    async def translate(self, items: Sequence[TranslationItem]) -> None:
        for item in items:
            item.translation = ""

    def add_context_page(self, _lines) -> None:  # noqa: D401 - mirror LLMTranslator
        return None
