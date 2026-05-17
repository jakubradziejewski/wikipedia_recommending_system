"""Scrapy item shape produced by the spider and consumed by the pipelines."""

from __future__ import annotations

import scrapy


class WikipediaArticleItem(scrapy.Item):
    # Spider-populated
    url = scrapy.Field()
    title = scrapy.Field()
    original_text = scrapy.Field()

    # Pipeline-populated by TextProcessingPipeline
    tokens = scrapy.Field()
    stems = scrapy.Field()
    lemmas = scrapy.Field()
    token_count = scrapy.Field()
    text_length = scrapy.Field()
