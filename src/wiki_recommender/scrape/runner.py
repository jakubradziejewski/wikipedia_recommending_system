"""Wrap Scrapy's CrawlerProcess so the CLI stays small.

CrawlerProcess owns the Twisted reactor and can only be started once per
process — fine here because each ``wikirec scrape`` invocation is a fresh
Python process. Pull it into a module-level helper so the CLI command can
stay focused on argument plumbing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from scrapy.crawler import CrawlerProcess
from scrapy.settings import Settings

from wiki_recommender.config import Paths, SpiderConfig
from wiki_recommender.scrape import settings as default_settings
from wiki_recommender.scrape.spider import WikipediaSpider

log = logging.getLogger(__name__)


def _build_settings(cfg: SpiderConfig, output_path: Path) -> Settings:
    settings = Settings()
    settings.setmodule(default_settings)
    settings.set("CONCURRENT_REQUESTS", cfg.concurrent_requests)
    settings.set("DOWNLOAD_DELAY", cfg.download_delay)
    settings.set("COOKIES_ENABLED", cfg.cookies_enabled)
    settings.set("ROBOTSTXT_OBEY", cfg.respect_robots)
    settings.set("WIKIREC_OUTPUT_PATH", str(output_path))
    return settings


def run_crawl(cfg: SpiderConfig, paths: Paths) -> Path:
    """Run the crawl synchronously and return the path of the produced parquet."""
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    output_path = paths.parquet_path

    log.info(
        "Starting crawl from %s (target: %d pages, %d links/page)",
        cfg.start_url, cfg.max_pages, cfg.max_links_per_page,
    )

    process = CrawlerProcess(settings=_build_settings(cfg, output_path))
    process.crawl(WikipediaSpider, cfg=cfg)
    process.start()  # Blocks until the spider closes; reactor shuts down after.

    return output_path
