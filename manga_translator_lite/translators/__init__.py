from ..config import LLMProvider, TranslatorConfig
from .common import (
    ISO_639_1_TO_VALID_LANGUAGES,
    InvalidServerResponse,
    LanguageUnsupportedException,
    MissingAPIKeyException,
    VALID_LANGUAGES,
)
from .llm import LLMTranslator, TranslationItem, make_batches
from .none import NoneTranslator


def build_translator(cfg: TranslatorConfig):
    """Construct the right translator object for the given config."""
    if cfg.provider == LLMProvider.none:
        return NoneTranslator()
    return LLMTranslator(cfg)


__all__ = [
    "ISO_639_1_TO_VALID_LANGUAGES",
    "InvalidServerResponse",
    "LLMTranslator",
    "LanguageUnsupportedException",
    "MissingAPIKeyException",
    "NoneTranslator",
    "TranslationItem",
    "VALID_LANGUAGES",
    "build_translator",
    "make_batches",
]
