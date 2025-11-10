import scrapy
from scrapy.crawler import CrawlerProcess
import pandas as pd
import random
import re
import os
from datetime import datetime
from collections import Counter
import logging

# NLTK imports for text processing
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass


class TextProcessor:
    """Handles text preprocessing including tokenization, stemming, and lemmatization"""

    def __init__(self):
        self.porter = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Remove unwanted characters and normalize text"""
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def process_text(self, text):
        """Process text: tokenize, remove stopwords, and apply stemming/lemmatization"""
        cleaned_text = self.clean_text(text)

        # Tokenize
        tokens = word_tokenize(cleaned_text.lower())

        # Filter: keep only alphabetic tokens that are not stopwords
        filtered_tokens = [
            token for token in tokens
            if token.isalpha() and token not in self.stop_words and len(token) > 2
        ]

        # Apply stemming
        stems = [self.porter.stem(token) for token in filtered_tokens]

        # Apply lemmatization
        lemmas = [self.lemmatizer.lemmatize(token, pos='v') for token in filtered_tokens]

        return {
            'original_text': cleaned_text,
            'tokens': filtered_tokens,
            'stems': stems,
            'lemmas': lemmas,
            'token_count': len(filtered_tokens)
        }


class WikipediaSpider(scrapy.Spider):
    """Optimized Spider for crawling Wikipedia pages with BFS strategy"""

    name = 'wikipedia_spider'
    allowed_domains = ['en.wikipedia.org']
    start_urls = ['https://en.wikipedia.org/wiki/Madagascar']

    custom_settings = {
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 8,  # Increased for faster scraping
        'DOWNLOAD_DELAY': 0.25,  # Reduced delay
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
        self.max_pages = 1250
        self.max_links_per_page = 20
        self.pages_scraped = 0
        self.data = []
        self.text_processor = TextProcessor()
        self.start_time = datetime.now()

    def parse(self, response):
        """Parse Wikipedia page and extract content"""

        # Force close spider if limit reached
        if self.pages_scraped >= self.max_pages:
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

        # Extract main content
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

        # Extract links for BFS
        if self.pages_scraped < self.max_pages:
            all_links = response.css('#mw-content-text a::attr(href)').getall()

            valid_links = []
            for link in all_links:
                if not link.startswith('/wiki/'):
                    continue

                if any(skip in link for skip in [':', 'Main_Page', '#']):
                    continue

                full_url = response.urljoin(link)

                if full_url not in self.visited_urls:
                    valid_links.append(full_url)

            valid_links = list(set(valid_links))
            random.shuffle(valid_links)
            selected_urls = valid_links[:self.max_links_per_page]

            for new_url in selected_urls:
                yield scrapy.Request(
                    new_url,
                    callback=self.parse,
                    priority=0,
                    dont_filter=False
                )

    def closed(self, reason):
        """Called when spider is closed - save data to parquet"""
        print("\n")  # New line after progress bar

        if self.data:
            df = pd.DataFrame(self.data)

            # Save to parquet (overwrite existing)
            output_file = 'wikipedia_articles.parquet'
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


def perform_text_analysis(parquet_file='wikipedia_articles.parquet'):
    """Perform comprehensive text analysis on scraped data"""

    print("\n" + "=" * 70)
    print("TEXT PROCESSING & ANALYSIS")
    print("=" * 70)

    if not os.path.exists(parquet_file):
        print(f"✗ Error: {parquet_file} not found!")
        return

    # Load data
    print(f"\n📊 Loading data from {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    print(f"✓ Loaded {len(df)} articles")

    # Basic statistics
    print("\n" + "-" * 70)
    print("1. BASIC STATISTICS")
    print("-" * 70)
    print(f"Total articles: {len(df)}")
    print(f"Total tokens: {df['token_count'].sum():,}")
    print(f"Total characters: {df['text_length'].sum():,}")
    print(f"\nTokens per article:")
    print(f"  - Mean: {df['token_count'].mean():.0f}")
    print(f"  - Median: {df['token_count'].median():.0f}")
    print(f"  - Min: {df['token_count'].min()}")
    print(f"  - Max: {df['token_count'].max()}")
    print(f"\nText length per article:")
    print(f"  - Mean: {df['text_length'].mean():.0f} characters")
    print(f"  - Median: {df['text_length'].median():.0f} characters")

    # Vocabulary analysis
    print("\n" + "-" * 70)
    print("2. VOCABULARY ANALYSIS")
    print("-" * 70)

    all_tokens = ' '.join(df['tokens'].dropna()).split()
    all_stems = ' '.join(df['stems'].dropna()).split()
    all_lemmas = ' '.join(df['lemmas'].dropna()).split()

    print(f"Unique tokens (vocabulary size): {len(set(all_tokens)):,}")
    print(f"Unique stems: {len(set(all_stems)):,}")
    print(f"Unique lemmas: {len(set(all_lemmas)):,}")
    print(f"Total word occurrences: {len(all_tokens):,}")

    # Most common words
    token_freq = Counter(all_tokens)
    print(f"\nTop 20 most frequent tokens:")
    for i, (word, count) in enumerate(token_freq.most_common(20), 1):
        print(f"  {i:2d}. {word:15s} ({count:5d} occurrences)")

    # Stem analysis
    print("\n" + "-" * 70)
    print("3. STEMMING ANALYSIS")
    print("-" * 70)
    stem_freq = Counter(all_stems)
    print(f"Top 15 most frequent stems:")
    for i, (stem, count) in enumerate(stem_freq.most_common(15), 1):
        print(f"  {i:2d}. {stem:15s} ({count:5d} occurrences)")

    # Lemma analysis
    print("\n" + "-" * 70)
    print("4. LEMMATIZATION ANALYSIS")
    print("-" * 70)
    lemma_freq = Counter(all_lemmas)
    print(f"Top 15 most frequent lemmas:")
    for i, (lemma, count) in enumerate(lemma_freq.most_common(15), 1):
        print(f"  {i:2d}. {lemma:15s} ({count:5d} occurrences)")

    # Compression analysis
    print("\n" + "-" * 70)
    print("5. TEXT NORMALIZATION EFFICIENCY")
    print("-" * 70)
    vocab_reduction_stem = (1 - len(set(all_stems)) / len(set(all_tokens))) * 100
    vocab_reduction_lemma = (1 - len(set(all_lemmas)) / len(set(all_tokens))) * 100
    print(f"Vocabulary reduction through stemming: {vocab_reduction_stem:.1f}%")
    print(f"Vocabulary reduction through lemmatization: {vocab_reduction_lemma:.1f}%")

    # Article samples
    print("\n" + "-" * 70)
    print("6. SAMPLE ARTICLES")
    print("-" * 70)
    print("First 10 articles:")
    for i, (idx, row) in enumerate(df.head(10).iterrows(), 1):
        print(f"  {i:2d}. {row['title']} ({row['token_count']} tokens)")

    print("\nLast 10 articles:")
    for i, (idx, row) in enumerate(df.tail(10).iterrows(), 1):
        print(f"  {len(df) - 10 + i:2d}. {row['title']} ({row['token_count']} tokens)")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


def run_spider():
    """Run the Wikipedia spider"""
    print("=" * 70)
    print("WIKIPEDIA BFS CRAWLER")
    print("=" * 70)
    print("Configuration:")
    print(f"  - Start URL: https://en.wikipedia.org/wiki/Madagascar")
    print(f"  - Target pages: 1250")
    print(f"  - Links per page: 20 (random selection)")
    print(f"  - Strategy: Breadth-First Search (BFS)")
    print(f"  - Concurrent requests: 8")
    print("=" * 70)
    print("\nStarting scraper...\n")

    # Configure logging to suppress Scrapy messages
    logging.getLogger('scrapy').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)

    process = CrawlerProcess(settings={
        'FEEDS': {},
        'LOG_LEVEL': 'ERROR',
        'LOG_ENABLED': True,
    })

    process.crawl(WikipediaSpider)
    process.start()


if __name__ == '__main__':
    # Run the scraper
    run_spider()

    # Perform text analysis
    perform_text_analysis()