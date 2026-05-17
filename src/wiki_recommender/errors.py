"""Domain exceptions surfaced to the CLI as exit-code-1 errors."""

from __future__ import annotations


class WikiRecError(Exception):
    """Base for errors the CLI is expected to handle and translate to exit code 1."""


class MissingDataError(WikiRecError):
    """Raised when the parquet corpus is referenced before it exists on disk."""


class EmptyCorpusError(WikiRecError):
    """Raised when the corpus loads successfully but is empty / has no usable text."""


class QueryNotFoundError(WikiRecError):
    """Raised when none of the requested article titles can be resolved in the corpus."""


class ModelNotBuiltError(WikiRecError):
    """Raised when downstream code asks for the TF-IDF matrix before it has been fitted."""
