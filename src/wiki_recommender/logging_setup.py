"""Logging configuration shared by every CLI entry point.

Scrapy ships with a verbose logger of its own; without dampening it the user
sees hundreds of INFO lines per crawl that drown out our pipeline messages.
"""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(verbose: int = 0) -> None:
    """Configure the root logger.

    Args:
        verbose: 0 = WARNING root + INFO for ``wiki_recommender.*``,
                 1 = INFO root, 2+ = DEBUG root. Scrapy is held one step
                 below the root unless the user asks for DEBUG explicitly.
    """
    if verbose >= 2:
        root_level = logging.DEBUG
        scrapy_level = logging.INFO
    elif verbose == 1:
        root_level = logging.INFO
        scrapy_level = logging.WARNING
    else:
        root_level = logging.WARNING
        scrapy_level = logging.WARNING

    logging.basicConfig(
        level=root_level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        force=True,
    )

    logging.getLogger("wiki_recommender").setLevel(
        logging.DEBUG if verbose >= 2 else logging.INFO
    )
    for noisy in ("scrapy", "twisted", "asyncio", "filelock"):
        logging.getLogger(noisy).setLevel(scrapy_level)
