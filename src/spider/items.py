### This code defines the data structure for storing Wikipedia article inside the scraper before saving to the file.

import scrapy

class WikipediaArticleItem(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    original_text = scrapy.Field()
    
    # Fields populated by the pipeline
    tokens = scrapy.Field()
    stems = scrapy.Field()
    lemmas = scrapy.Field()
    token_count = scrapy.Field()
    text_length = scrapy.Field()