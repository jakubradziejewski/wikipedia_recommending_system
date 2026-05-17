"""NLTK corpus bootstrap.

Kept out of the package import path so importing ``wiki_recommender`` never
touches the network. The CLI calls :func:`ensure_nltk_data` once on the
``scrape`` command (and during Docker build) so subsequent calls are offline.
"""

from __future__ import annotations

import logging

import nltk

log = logging.getLogger(__name__)

# Tagger name changed between NLTK majors; the engine-tagged version is what
# WordNetLemmatizer needs alongside the universal tagger fallback.
_REQUIRED_PACKAGES: tuple[str, ...] = (
    "punkt",
    "punkt_tab",
    "wordnet",
    "stopwords",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
)


def ensure_nltk_data(packages: tuple[str, ...] = _REQUIRED_PACKAGES) -> None:
    """Download the NLTK corpora the pipeline depends on, idempotently."""
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except (OSError, ValueError) as exc:
            # Some legacy names (e.g. punkt_tab on older NLTK) raise ValueError;
            # we tolerate it and let downstream LookupError surface only when actually missing.
            log.debug("Skipping NLTK package %s (%s)", pkg, exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("Bootstrapping NLTK corpora...")
    ensure_nltk_data()
    log.info("Done.")
