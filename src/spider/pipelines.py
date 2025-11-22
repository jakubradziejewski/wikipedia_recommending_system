import pandas as pd
import os
from datetime import datetime
# Absolute import based on your src folder structure
from src.processing.text_processor import TextProcessor

class TextProcessingPipeline:
    """
    Intercepts the item, runs it through TextProcessor,
    and populates the linguistic fields.
    """
    def __init__(self):
        self.processor = TextProcessor()

    def process_item(self, item, spider):
        content = item.get('original_text', '')
        
        if content:
            # Process the text (Tokenize, Stem, Lemmatize)
            processed = self.processor.process_text(content)
            
            # Update the Item
            item['tokens'] = ' '.join(processed['tokens'])
            item['stems'] = ' '.join(processed['stems'])
            item['lemmas'] = ' '.join(processed['lemmas'])
            item['token_count'] = processed['token_count']
            item['text_length'] = len(processed['original_text'])
            
        return item

class ParquetExportPipeline:
    """
    Collects all items in memory and saves them to a Parquet file
    when the spider closes.
    """
    def __init__(self):
        self.items = []
        self.start_time = datetime.now()

    def process_item(self, item, spider):
        # Convert Scrapy Item to standard dict for DataFrame
        self.items.append(dict(item))
        return item

    def close_spider(self, spider):
        if self.items:
            df = pd.DataFrame(self.items)
            
            # Ensure data directory exists (as required by main.py Step #1)
            os.makedirs('data', exist_ok=True)
            output_file = 'data/wikipedia_articles.parquet'
            
            # Save to Parquet
            df.to_parquet(output_file, engine='pyarrow', compression='snappy')
            
            elapsed = (datetime.now() - self.start_time).total_seconds()
            print(f"\n✓ SCRAPING COMPLETE: {len(self.items)} articles saved to {output_file}")
            print(f"✓ Time elapsed: {elapsed:.1f}s")
        else:
            print("⚠ Warning: No data collected.")