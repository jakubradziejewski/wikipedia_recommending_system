import os
import logging
from scrapy.crawler import CrawlerProcess

# Local imports
from src.spider.wikipedia_spider import WikipediaSpider
from src.analysis.statistics import perform_text_analysis

from src.engine.similarity_engine import ArticleSimilarityEngine
from src.analysis.statistics import generate_model_statistics
from src.engine.strategy import compare_recommendation_strategies


def main():
    print("=" * 80)
    print("WIKIPEDIA ARTICLE PIPELINE — SCRAPE → ANALYZE → SIMILARITY")
    print("=" * 80)

    data_path = "data/wikipedia_articles.parquet"

    # Step 1: Check if parquet file exists
    if os.path.exists(data_path):
        print(f"✓ Found existing dataset: {data_path}")
    else:
        print("No dataset found.")
        print("--> Starting Wikipedia crawl to create dataset...\n")

        os.makedirs("data", exist_ok=True)
        logging.getLogger("scrapy").setLevel(logging.ERROR)
        process = CrawlerProcess(settings={"LOG_LEVEL": "ERROR"})
        process.crawl(WikipediaSpider)
        process.start()

    # Step 2: Perform basic text analysis
    perform_text_analysis(data_path)

    # Step 3: Build similarity engine
    print("Building Article Similarity Engine using Text after Lemmatization\n")
    engine = ArticleSimilarityEngine(data_path)
    engine.build_tfidf_model()

    # Step 4: Generate statistics
    generate_model_statistics(engine)

    # Step 5: Compare recommendation strategies, change number of trials to average results
    compare_recommendation_strategies(engine, num_articles=10, run_explainability=True, num_trials=100)

    print("Pipe completed successfully!")


if __name__ == "__main__":
    main()
