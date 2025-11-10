import os
import logging
from scrapy.crawler import CrawlerProcess

# Local imports
from src.spider.wikipedia_spider import WikipediaSpider
from src.analysis.text_analysis import perform_text_analysis

from src.engine.similarity_engine import ArticleSimilarityEngine
from src.engine.explainability import explain_similarity, visualize_similarity_explanation
from src.engine.statistics import generate_database_statistics
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
        print("⚠ No dataset found.")
        print("→ Starting Wikipedia crawl to create dataset...\n")

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
    engine.build_tfidf_model(max_features=20000)

    # Step 4: Generate statistics
    generate_database_statistics(engine)

    # Step 5: Compare recommendation strategies
    strategies = compare_recommendation_strategies(engine, num_articles=10)

    # Step 6: Explain and visualize similarity
    if strategies.get("random_titles"):
        target = strategies["random_titles"][0]
        explanation = explain_similarity(engine, strategies["random_titles"][:3], target)
        visualize_similarity_explanation(explanation, save_path="../plots/explanation_example.png")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
