### This file defines the Wikipedia spider (and its most important parse() method) which crawls articles starting from Madagascar.
### It uses BFS strategy to explore links and yields items for processing.

import scrapy
import random
from datetime import datetime
from scrapy.exceptions import CloseSpider
from src.spider.items import WikipediaArticleItem

class WikipediaSpider(scrapy.Spider):
    name = 'wikipedia_spider'
    allowed_domains = ['en.wikipedia.org']
    start_urls = ['https://en.wikipedia.org/wiki/Madagascar']

    # Custom settings ensure critical settings are applied even if main.py doesn't load settings.py
    custom_settings = {
        'LOG_LEVEL': 'ERROR',
        'ITEM_PIPELINES': {
            'src.spider.pipelines.TextProcessingPipeline': 300,
            'src.spider.pipelines.ParquetExportPipeline': 400,
        }
    }

    def __init__(self, *args, **kwargs):
        super(WikipediaSpider, self).__init__(*args, **kwargs)
        self.visited_urls = set()
        self.max_pages = 5000
        self.max_links_per_page = 10
        self.pages_scraped = 0
        self.start_time = datetime.now()

    def parse(self, response):
        # 1. Check Limits
        if self.pages_scraped >= self.max_pages:
            raise CloseSpider('Reached maximum pages limit')

        url = response.url
        if url in self.visited_urls:
            return

        self.visited_urls.add(url)
        self.pages_scraped += 1

        # 2. Progress Logging
        if self.pages_scraped % 50 == 0:
            print(f"\r[{self.pages_scraped}/{self.max_pages}] Scraping: {url}", end='', flush=True)

        # 3. Extract Title
        title = response.css('h1.firstHeading::text').get()
        if not title:
            title = url.split('/')[-1].replace('_', ' ')

        # 4. Extract Content (Deep Extraction)
        # We use XPath .//text() to get text from <p> AND its children (<a>, <b>, etc)
        # This ensures text "hidden" inside hyperlinks is captured.
        paragraphs = response.css('#mw-content-text .mw-parser-output > p')
        content_parts = []
        
        for para in paragraphs:
            # Get all text nodes within this paragraph recursively
            text_nodes = para.xpath('.//text()').getall()
            # Join them to form the full sentence(s)
            para_text = ' '.join(text_nodes)
            if para_text.strip():
                content_parts.append(para_text)
        
        content = ' '.join(content_parts)

        # Fallback for different Wiki layouts if main selector fails
        if not content.strip():
            alt_paragraphs = response.css('#mw-content-text p')
            content_parts = [' '.join(p.xpath('.//text()').getall()) for p in alt_paragraphs]
            content = ' '.join(content_parts)

        # 5. Yield Item
        # We yield the item immediately. The Pipeline will handle the 'TextProcessor' logic.
        if content.strip():
            item = WikipediaArticleItem()
            item['url'] = url
            item['title'] = title
            item['original_text'] = content
            yield item

        # 6. BFS Navigation
        if self.pages_scraped < self.max_pages:
            all_links = response.css('#mw-content-text a::attr(href)').getall()
            
            valid_links = []
            for link in all_links:
                # Filter for valid article links
                if link.startswith('/wiki/') and not any(x in link for x in [':', 'Main_Page', '#']):
                    full_url = response.urljoin(link)
                    if full_url not in self.visited_urls:
                        valid_links.append(full_url)
            
            valid_links = list(set(valid_links))
            random.shuffle(valid_links)
            
            # Schedule next requests
            for next_url in valid_links[:self.max_links_per_page]:
                yield scrapy.Request(next_url, callback=self.parse)