"""Text preprocessing for Wikipedia article bodies.

The pipeline produces three parallel representations — raw tokens, Porter
stems, WordNet lemmas — so the TF-IDF engine can be retargeted at any of
them via :class:`wiki_recommender.config.EngineConfig.text_column`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

_CITATION_BRACKETS = re.compile(r"\[\d+\]")
_WHITESPACE = re.compile(r"\s+")

# Tokens shorter than this are typically noise (single letters from formula
# fragments, OCR-like artifacts in Wikipedia footnotes). The original pipeline
# used >2; keeping that threshold to preserve corpus statistics across the rewrite.
_MIN_TOKEN_LENGTH = 3


@dataclass(frozen=True, slots=True)
class ProcessedText:
    """Output of :meth:`TextProcessor.process_text`.

    ``tokens``/``stems``/``lemmas`` are aligned by index where possible:
    ``stems[i]`` and ``lemmas[i]`` correspond to ``tokens[i]``.
    """

    original_text: str
    tokens: list[str]
    stems: list[str]
    lemmas: list[str]
    token_count: int


class TextProcessor:
    """Stateful preprocessor.

    Construct once per pipeline — the NLTK objects are cheap to reuse and
    expensive to re-create per-document.
    """

    _TREEBANK_TO_WORDNET: ClassVar[dict[str, str]] = {
        "J": wordnet.ADJ,
        "V": wordnet.VERB,
        "N": wordnet.NOUN,
        "R": wordnet.ADV,
    }

    def __init__(self) -> None:
        self._stemmer = PorterStemmer()
        self._lemmatizer = WordNetLemmatizer()
        self._stop_words: frozenset[str] = frozenset(stopwords.words("english"))

    @classmethod
    def _wordnet_pos(cls, treebank_tag: str) -> str:
        """Map a Penn-Treebank tag prefix to the WordNet POS family.

        WordNet only distinguishes four parts of speech; unknowns default to
        NOUN, which matches WordNetLemmatizer's own internal default.
        """
        if not treebank_tag:
            return wordnet.NOUN
        return cls._TREEBANK_TO_WORDNET.get(treebank_tag[0], wordnet.NOUN)

    @staticmethod
    def clean_text(text: str) -> str:
        """Strip Wikipedia footnote markers and collapse whitespace."""
        text = _CITATION_BRACKETS.sub("", text)
        text = _WHITESPACE.sub(" ", text)
        return text.strip()

    def process_text(self, text: str) -> ProcessedText:
        """Tokenize, filter, stem, and POS-aware-lemmatize one article body."""
        cleaned = self.clean_text(text)
        tokens = word_tokenize(cleaned.lower())

        filtered = [
            t for t in tokens
            if t.isalpha() and t not in self._stop_words and len(t) >= _MIN_TOKEN_LENGTH
        ]

        stems = [self._stemmer.stem(t) for t in filtered]

        # POS-tag the *filtered* tokens, not the raw stream — tagging stopwords
        # is wasted work, and dropping them after tagging changes the contexts
        # the tagger sees in subtle ways. The original pipeline did the same.
        pos_tags = nltk.pos_tag(filtered)
        lemmas = [
            self._lemmatizer.lemmatize(tok, pos=self._wordnet_pos(tag))
            for tok, tag in pos_tags
        ]

        return ProcessedText(
            original_text=cleaned,
            tokens=filtered,
            stems=stems,
            lemmas=lemmas,
            token_count=len(filtered),
        )
