### Scrapy settings for wikipedia_spider

BOT_NAME = 'wikipedia_spider'
SPIDER_MODULES = ['src.spider']

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests
CONCURRENT_REQUESTS = 32

# Configure a delay for requests for the same website (default: 0)
DOWNLOAD_DELAY = 0.1
COOKIES_ENABLED = False

# Disable Telnet Console 
TELNETCONSOLE_ENABLED = False

# BFS Queue Settings
DEPTH_PRIORITY = 1
SCHEDULER_DISK_QUEUE = 'scrapy.squeues.PickleFifoDiskQueue'
SCHEDULER_MEMORY_QUEUE = 'scrapy.squeues.FifoMemoryQueue'

# Pipelines (first TextProcessing, then ParquetExport)
ITEM_PIPELINES = {
   'src.spider.pipelines.TextProcessingPipeline': 300,
   'src.spider.pipelines.ParquetExportPipeline': 400,
}