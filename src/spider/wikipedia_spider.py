import random
import pandas as pd
import scrapy
from datetime import datetime
from scrapy.exceptions import CloseSpider
from src.processing.text_processor import TextProcessor

class WikipediaSpider(scrapy.Spider):
    """Spider for crawling Wikipedia pages with BFS strategy"""

    name = 'wikipedia_spider'
    allowed_domains = ['en.wikipedia.org']
    start_urls = ['https://en.wikipedia.org/wiki/Madagascar']

    custom_settings = {
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 16,
        'DOWNLOAD_DELAY': 0.25,
        'COOKIES_ENABLED': False,
        'RETRY_TIMES': 2,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'DEPTH_PRIORITY': 1,
        'SCHEDULER_DISK_QUEUE': 'scrapy.squeues.PickleFifoDiskQueue',
        'SCHEDULER_MEMORY_QUEUE': 'scrapy.squeues.FifoMemoryQueue',
        'DNSCACHE_ENABLED': True,
        'REDIRECT_ENABLED': True,
        'AJAXCRAWL_ENABLED': False,
        'LOG_LEVEL': 'ERROR',  # Suppress unnecessary messages
    }

    def __init__(self, *args, **kwargs):
        super(WikipediaSpider, self).__init__(*args, **kwargs)
        self.visited_urls = set()
        self.max_pages = 5000
        self.max_links_per_page = 10
        self.pages_scraped = 0
        self.data = []
        self.text_processor = TextProcessor()
        self.start_time = datetime.now()

    def parse(self, response):
        """Parse Wikipedia page and extract content"""

        # Force close spider if limit reached
        if self.pages_scraped > self.max_pages:
            from scrapy.exceptions import CloseSpider
            raise CloseSpider('Reached maximum pages limit')

        url = response.url

        if url in self.visited_urls:
            return

        self.visited_urls.add(url)
        self.pages_scraped += 1

        # Progress indicator
        if self.pages_scraped % 50 == 0 or self.pages_scraped == 1:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.pages_scraped / elapsed if elapsed > 0 else 0
            remaining = (self.max_pages - self.pages_scraped) / rate if rate > 0 else 0
            print(
                f"\r[{self.pages_scraped}/{self.max_pages}] Progress: {self.pages_scraped / self.max_pages * 100:.1f}% | "
                f"Rate: {rate:.1f} pages/sec | ETA: {remaining / 60:.1f} min", end='', flush=True)

        # Extract title
        title = response.css('h1.firstHeading::text').get()
        if not title:
            title = response.url.split('/')[-1].replace('_', ' ')

        # Extract main content including text inside links
        # Get all text from paragraphs, including linked text
        paragraphs = response.css('#mw-content-text .mw-parser-output > p')
        content_parts = []
        for para in paragraphs:
            para_text = para.css('::text').getall()
            content_parts.append(' '.join(para_text))

        content = ' '.join(content_parts)

        if not content:
            paragraphs_alt = response.css('#mw-content-text p')
            content_parts = []
            for para in paragraphs_alt:
                para_text = para.css('::text').getall()
                content_parts.append(' '.join(para_text))
            content = ' '.join(content_parts)

        # Process the text
        if content.strip():
            processed = self.text_processor.process_text(content)

            self.data.append({
                'url': url,
                'title': title,
                'original_text': processed['original_text'],
                'tokens': ' '.join(processed['tokens']),
                'stems': ' '.join(processed['stems']),
                'lemmas': ' '.join(processed['lemmas']),
                'token_count': processed['token_count'],
                'text_length': len(processed['original_text'])
            })

        # Extract links for BFS - only if we haven't reached the limit
        if self.pages_scraped < self.max_pages:
            # Extract all links from the content area
            all_links = response.css('#mw-content-text a::attr(href)').getall()

            valid_links = []
            for link in all_links:
                # Must start with /wiki/
                if not link.startswith('/wiki/'):
                    continue
                # Skip special pages
                if any(skip in link for skip in [':', 'Main_Page', '#']):
                    continue
                # Convert to absolute URL
                full_url = response.urljoin(link)
                # Skip if already visited
                if full_url not in self.visited_urls:
                    valid_links.append(full_url)
            # Remove duplicates
            valid_links = list(set(valid_links))
            # Randomly select links to follow
            random.shuffle(valid_links)
            selected_urls = valid_links[:self.max_links_per_page]
            # Yield requests for selected URLs
            for new_url in selected_urls:
                yield scrapy.Request(
                    new_url,
                    callback=self.parse,
                    priority=0,
                    dont_filter=False
                )

    def closed(self, reason):
        """Called when spider is closed - save data to parquet"""

        if self.data:
            df = pd.DataFrame(self.data)

            # Save to parquet (overwrite existing)
            output_file = 'data/wikipedia_articles.parquet'
            df.to_parquet(output_file, engine='pyarrow', compression='snappy')

            elapsed = (datetime.now() - self.start_time).total_seconds()

            print("\n" + "=" * 70)
            print("SCRAPING COMPLETED")
            print("=" * 70)
            print(f"✓ Total pages scraped: {len(self.data)}")
            print(f"✓ Time elapsed: {elapsed / 60:.2f} minutes ({elapsed:.1f} seconds)")
            print(f"✓ Average rate: {len(self.data) / elapsed:.2f} pages/second")
            print(f"✓ Output file: {output_file}")
            print("=" * 70)
        else:
            print("⚠ Warning: No data collected!")