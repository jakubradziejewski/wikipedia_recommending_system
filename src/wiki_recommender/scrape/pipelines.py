"""Scrapy item pipelines: NLP preprocessing then parquet export."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from wiki_recommender.nlp import TextProcessor

log = logging.getLogger(__name__)

_DEFAULT_PARQUET_PATH = Path("data/wikipedia_articles.parquet")


class TextProcessingPipeline:
    """Run each scraped article through tokenize → stem → POS-aware lemmatize."""

    def __init__(self) -> None:
        self.processor = TextProcessor()

    def process_item(self, item: dict[str, Any], spider) -> dict[str, Any]:
        content = item.get("original_text", "") or ""
        if not content:
            return item

        processed = self.processor.process_text(content)
        item["tokens"] = " ".join(processed.tokens)
        item["stems"] = " ".join(processed.stems)
        item["lemmas"] = " ".join(processed.lemmas)
        item["token_count"] = processed.token_count
        item["text_length"] = len(processed.original_text)
        return item


class ParquetExportPipeline:
    """Buffer items in memory and write a single parquet file when the spider closes.

    Why in-memory: 5k articles × a few KB of lemma strings is tens of MB.
    Streaming via :class:`pyarrow.parquet.ParquetWriter` would require declaring
    the schema up front and managing chunked row groups — premature complexity
    for the bounded crawl size this project targets.
    """

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self._start_time = time.perf_counter()
        self.output_path: Path = _DEFAULT_PARQUET_PATH

    def open_spider(self, spider) -> None:
        # Runner injects the output path through Scrapy settings so the pipeline
        # stays decoupled from our config object.
        settings_path = spider.crawler.settings.get("WIKIREC_OUTPUT_PATH")
        if settings_path:
            self.output_path = Path(settings_path)

    def process_item(self, item: dict[str, Any], spider) -> dict[str, Any]:
        self.items.append(dict(item))
        return item

    def close_spider(self, spider) -> None:
        if not self.items:
            log.warning("Spider closed without scraping any items.")
            return

        df = pd.DataFrame(self.items)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path, engine="pyarrow", compression="snappy")

        elapsed = time.perf_counter() - self._start_time
        log.info(
            "Wrote %d articles to %s in %.1fs", len(df), self.output_path, elapsed
        )
