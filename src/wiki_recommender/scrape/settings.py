"""Scrapy settings module.

Loaded by :mod:`wiki_recommender.scrape.runner` via ``Settings.setmodule``;
field values can be further overridden at runtime from a
:class:`wiki_recommender.config.SpiderConfig`.
"""

from __future__ import annotations

BOT_NAME = "wikipedia_spider"
SPIDER_MODULES = ["wiki_recommender.scrape"]

ROBOTSTXT_OBEY = True
CONCURRENT_REQUESTS = 32
DOWNLOAD_DELAY = 0.1
COOKIES_ENABLED = False
TELNETCONSOLE_ENABLED = False

# BFS ordering: depth-priority + FIFO queues mean shallower links are scheduled
# first, so a crawl rooted at one seed expands breadth-first rather than depth-first.
DEPTH_PRIORITY = 1
SCHEDULER_DISK_QUEUE = "scrapy.squeues.PickleFifoDiskQueue"
SCHEDULER_MEMORY_QUEUE = "scrapy.squeues.FifoMemoryQueue"

ITEM_PIPELINES = {
    "wiki_recommender.scrape.pipelines.TextProcessingPipeline": 300,
    "wiki_recommender.scrape.pipelines.ParquetExportPipeline": 400,
}

# Required by Scrapy 2.7+; pinning here so the choice survives spider rewrites.
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
