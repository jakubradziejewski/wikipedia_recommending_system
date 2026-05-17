"""Wikipedia BFS spider.

Starts from a single seed article and expands breadth-first, picking up to
``max_links_per_page`` links from each visited page. Link shuffling avoids the
topical bias that a deterministic depth-priority traversal would have on a
seed article whose first paragraph repeats the same handful of categories.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import scrapy
from scrapy.exceptions import CloseSpider
from scrapy.http import Response

from wiki_recommender.config import SpiderConfig
from wiki_recommender.scrape.items import WikipediaArticleItem

log = logging.getLogger(__name__)

# Wikipedia internal links to skip: namespaces (`Talk:`, `User:`, etc.), the
# main portal, and same-page fragments. Matches the original filter set.
_SKIP_SUBSTRINGS: tuple[str, ...] = (":", "Main_Page", "#")

_PROGRESS_EVERY = 50


class WikipediaSpider(scrapy.Spider):
    """BFS spider over Wikipedia article pages."""

    name = "wikipedia_spider"
    allowed_domains = ["en.wikipedia.org"]

    def __init__(self, cfg: SpiderConfig | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cfg: SpiderConfig = cfg or SpiderConfig()
        self.start_urls: list[str] = [self.cfg.start_url]
        self.visited_urls: set[str] = set()
        self.pages_scraped: int = 0
        self._rng = random.Random(self.cfg.seed)

    def parse(self, response: Response, **_: Any):
        url = response.url
        if url in self.visited_urls:
            return

        if self.pages_scraped >= self.cfg.max_pages:
            # CloseSpider is the documented escape hatch; raising it stops scheduling
            # without dropping the items already in flight.
            raise CloseSpider("max_pages reached")

        self.visited_urls.add(url)
        self.pages_scraped += 1

        if self.pages_scraped % _PROGRESS_EVERY == 0:
            log.info(
                "Crawling %s/%s — current: %s",
                self.pages_scraped, self.cfg.max_pages, url,
            )

        title = response.css("h1.firstHeading::text").get()
        if not title:
            title = url.rsplit("/", 1)[-1].replace("_", " ")

        content = self._extract_body(response)
        if content:
            item = WikipediaArticleItem()
            item["url"] = url
            item["title"] = title
            item["original_text"] = content
            yield item

        if self.pages_scraped < self.cfg.max_pages:
            yield from self._enqueue_next(response)

    def _extract_body(self, response: Response) -> str:
        """Pull article body text.

        The main Wikipedia layout has paragraphs nested directly under
        ``.mw-parser-output``; some pages (stubs, redirects, soft-deletes)
        use a flatter ``#mw-content-text p`` structure. We try both, in that
        order, and use ``.//text()`` so text inside ``<a>`` / ``<b>`` is kept.
        """
        primary = response.css("#mw-content-text .mw-parser-output > p")
        content = self._join_paragraphs(primary)
        if content:
            return content

        fallback = response.css("#mw-content-text p")
        return self._join_paragraphs(fallback)

    @staticmethod
    def _join_paragraphs(paragraphs) -> str:
        parts: list[str] = []
        for para in paragraphs:
            text_nodes = para.xpath(".//text()").getall()
            joined = " ".join(text_nodes).strip()
            if joined:
                parts.append(joined)
        return " ".join(parts).strip()

    def _enqueue_next(self, response: Response):
        # Deduplicate while collecting — `dict` preserves insertion order and is
        # measurably faster than `list(set(...))` for the sizes we see (~hundreds
        # of links per page).
        seen: dict[str, None] = {}
        for href in response.css("#mw-content-text a::attr(href)").getall():
            if not href.startswith("/wiki/"):
                continue
            if any(skip in href for skip in _SKIP_SUBSTRINGS):
                continue
            full_url = response.urljoin(href)
            if full_url in self.visited_urls:
                continue
            seen.setdefault(full_url, None)

        candidates = list(seen)
        self._rng.shuffle(candidates)

        for next_url in candidates[: self.cfg.max_links_per_page]:
            yield scrapy.Request(next_url, callback=self.parse)
