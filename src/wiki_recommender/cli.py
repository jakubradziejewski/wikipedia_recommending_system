"""Typer CLI — the only user-facing entry point.

Five subcommands map onto the pipeline stages:

- ``scrape``   — run the BFS Wikipedia crawler.
- ``analyze``  — corpus statistics + TF-IDF matrix statistics.
- ``recommend``— direct top-K recommendations for one or more article titles.
- ``compare``  — random / similar / recursive strategy comparison.
- ``pipeline`` — convenience: chain scrape (if needed) → analyze → compare.

Imports of Scrapy and matplotlib are deferred to inside the command bodies
so ``wikirec --help`` stays fast.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from wiki_recommender import __version__
from wiki_recommender.config import EngineConfig, Settings, StrategyConfig
from wiki_recommender.errors import WikiRecError
from wiki_recommender.logging_setup import configure_logging

app = typer.Typer(
    add_completion=False,
    help="Content-based Wikipedia article recommender (TF-IDF + cosine similarity).",
    rich_markup_mode="rich",
)

log = logging.getLogger("wiki_recommender.cli")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"wikirec {__version__}")
        raise typer.Exit()


@app.callback()
def _global_options(
    verbose: Annotated[
        int,
        typer.Option("-v", "--verbose", count=True, help="-v for INFO, -vv for DEBUG."),
    ] = 0,
    _version: Annotated[
        Optional[bool],
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version and exit."),
    ] = None,
) -> None:
    configure_logging(verbose)


def _load_settings() -> Settings:
    """Build a fresh Settings instance; surfaces validation errors early."""
    try:
        return Settings()
    except Exception as exc:
        log.error("Configuration error: %s", exc)
        raise typer.Exit(1) from exc


def _handle(exc: WikiRecError) -> "typer.Exit":
    log.error("%s", exc)
    return typer.Exit(1)


@app.command(help="Crawl Wikipedia and write the article corpus to parquet.")
def scrape(
    start_url: Annotated[Optional[str], typer.Option("--start-url", help="Seed URL for the BFS crawl.")] = None,
    max_pages: Annotated[Optional[int], typer.Option("--max-pages", help="Cap on number of articles.")] = None,
    max_links_per_page: Annotated[Optional[int], typer.Option("--max-links-per-page", help="Outbound links to schedule per page.")] = None,
    output: Annotated[Optional[Path], typer.Option("--output", help="Override the destination parquet path.")] = None,
    seed: Annotated[Optional[int], typer.Option("--seed", help="Seed BFS link shuffling for reproducible crawls.")] = None,
) -> None:
    settings = _load_settings()
    spider_cfg = settings.spider.model_copy(
        update={
            k: v
            for k, v in {
                "start_url": start_url,
                "max_pages": max_pages,
                "max_links_per_page": max_links_per_page,
                "seed": seed,
            }.items()
            if v is not None
        }
    )
    if output is not None:
        settings = settings.model_copy(
            update={"paths": settings.paths.model_copy(update={
                "data_dir": output.parent,
                "parquet_name": output.name,
            })}
        )

    # Lazy import: pulls in Twisted and Scrapy, which is slow.
    from wiki_recommender.nlp.bootstrap import ensure_nltk_data
    from wiki_recommender.scrape.runner import run_crawl

    ensure_nltk_data()
    try:
        out_path = run_crawl(spider_cfg, settings.paths)
    except WikiRecError as exc:
        raise _handle(exc) from exc
    typer.echo(f"Corpus written to {out_path}")


@app.command(help="Print corpus statistics and TF-IDF model statistics.")
def analyze(
    input: Annotated[Optional[Path], typer.Option("--input", help="Override the parquet input path.")] = None,
    plots_dir: Annotated[Optional[Path], typer.Option("--plots-dir", help="Where to save generated plots.")] = None,
    text_column: Annotated[Optional[str], typer.Option("--text-column", help="lemmas | tokens | stems.")] = None,
    ngram_max: Annotated[Optional[int], typer.Option("--ngram-max", help="Upper bound on TF-IDF n-gram size.")] = None,
    min_df: Annotated[Optional[float], typer.Option("--min-df", help="Drop terms appearing in fewer than this many documents.")] = None,
    max_df: Annotated[Optional[float], typer.Option("--max-df", help="Drop terms appearing in more than this fraction of documents.")] = None,
    no_sublinear_tf: Annotated[bool, typer.Option("--no-sublinear-tf", help="Disable sublinear TF scaling.")] = False,
) -> None:
    settings = _load_settings()
    parquet_path = input or settings.paths.parquet_path
    plot_dir = plots_dir or settings.paths.plots_dir
    engine_cfg = _engine_cfg_with_overrides(
        settings.engine, text_column, ngram_max, min_df, max_df, no_sublinear_tf,
    )

    from wiki_recommender.analysis.corpus_stats import perform_text_analysis
    from wiki_recommender.analysis.model_stats import generate_model_statistics
    from wiki_recommender.engine.similarity import ArticleSimilarityEngine

    try:
        perform_text_analysis(parquet_path)
        engine = ArticleSimilarityEngine.load(parquet_path, engine_cfg)
        generate_model_statistics(engine, plot_dir, dpi=settings.viz.dpi)
    except WikiRecError as exc:
        raise _handle(exc) from exc


@app.command(help="Print top-K recommendations for one or more article titles.")
def recommend(
    query: Annotated[list[str], typer.Option("--query", "-q", help="Article title (repeatable).")],
    input: Annotated[Optional[Path], typer.Option("--input", help="Override the parquet input path.")] = None,
    top_k: Annotated[Optional[int], typer.Option("--top-k", "-k", help="Number of recommendations.")] = None,
    text_column: Annotated[Optional[str], typer.Option("--text-column", help="lemmas | tokens | stems.")] = None,
    ngram_max: Annotated[Optional[int], typer.Option("--ngram-max", help="Upper bound on TF-IDF n-gram size.")] = None,
    min_df: Annotated[Optional[float], typer.Option("--min-df", help="Drop terms appearing in fewer than this many documents.")] = None,
    max_df: Annotated[Optional[float], typer.Option("--max-df", help="Drop terms appearing in more than this fraction of documents.")] = None,
    no_sublinear_tf: Annotated[bool, typer.Option("--no-sublinear-tf", help="Disable sublinear TF scaling.")] = False,
) -> None:
    if not query:
        log.error("At least one --query is required.")
        raise typer.Exit(1)

    settings = _load_settings()
    parquet_path = input or settings.paths.parquet_path
    engine_cfg = _engine_cfg_with_overrides(
        settings.engine, text_column, ngram_max, min_df, max_df, no_sublinear_tf,
    )
    if top_k is not None:
        engine_cfg = engine_cfg.model_copy(update={"top_k": top_k})

    from wiki_recommender.analysis.reporting import print_recommendations
    from wiki_recommender.engine.similarity import ArticleSimilarityEngine

    try:
        engine = ArticleSimilarityEngine.load(parquet_path, engine_cfg)
        recs = engine.find_similar_articles(query)
    except WikiRecError as exc:
        raise _handle(exc) from exc

    print_recommendations(recs, label=f"Top {len(recs)} Recommendations for {query}")


@app.command(help="Compare the three query-construction strategies side by side.")
def compare(
    input: Annotated[Optional[Path], typer.Option("--input", help="Override the parquet input path.")] = None,
    plots_dir: Annotated[Optional[Path], typer.Option("--plots-dir", help="Where to save generated plots.")] = None,
    num_articles: Annotated[Optional[int], typer.Option("--num-articles", help="Query size for each strategy.")] = None,
    num_trials: Annotated[Optional[int], typer.Option("--num-trials", help="Number of trials to average over.")] = None,
    explain: Annotated[Optional[bool], typer.Option("--explain/--no-explain", help="Run explainability on the first trial.")] = None,
    text_column: Annotated[Optional[str], typer.Option("--text-column", help="lemmas | tokens | stems.")] = None,
) -> None:
    settings = _load_settings()
    parquet_path = input or settings.paths.parquet_path
    plot_dir = plots_dir or settings.paths.plots_dir
    strategy_cfg = _strategy_cfg_with_overrides(settings.strategy, num_articles, num_trials, explain)
    engine_cfg = settings.engine.model_copy(update={"text_column": text_column} if text_column else {})

    from wiki_recommender.analysis.explain import explainability_analysis
    from wiki_recommender.analysis.reporting import describe_strategy, summarize_comparison
    from wiki_recommender.engine.similarity import ArticleSimilarityEngine
    from wiki_recommender.engine.strategies import compare_strategies

    try:
        engine = ArticleSimilarityEngine.load(parquet_path, engine_cfg)
        trials = compare_strategies(engine, strategy_cfg)
    except WikiRecError as exc:
        raise _handle(exc) from exc

    # Verbose on first trial only — replicates the original "describe once, then accumulate" output.
    for result in trials[0]:
        describe_strategy(result)

    if strategy_cfg.run_explainability:
        for result in trials[0]:
            explainability_analysis(
                engine,
                result.query_titles,
                result.recommendations,
                strategy_name=result.name,
                plots_dir=plot_dir,
                min_enrichment=settings.explain.min_enrichment,
                top_n_terms=settings.explain.top_n_terms,
                top_distinctive_terms=settings.explain.top_distinctive_terms,
                dpi=settings.viz.dpi,
            )

    summarize_comparison(trials)


@app.command(help="Run the full pipeline: scrape (if needed) → analyze → compare.")
def pipeline(
    skip_scrape: Annotated[bool, typer.Option("--skip-scrape", help="Skip the scrape step even if the parquet is missing.")] = False,
) -> None:
    settings = _load_settings()
    parquet_path = settings.paths.parquet_path

    print("=" * 80)
    print("WIKIPEDIA ARTICLE PIPELINE — SCRAPE → ANALYZE → SIMILARITY")
    print("=" * 80)

    from wiki_recommender.analysis.corpus_stats import perform_text_analysis
    from wiki_recommender.analysis.explain import explainability_analysis
    from wiki_recommender.analysis.model_stats import generate_model_statistics
    from wiki_recommender.analysis.reporting import describe_strategy, summarize_comparison
    from wiki_recommender.engine.similarity import ArticleSimilarityEngine
    from wiki_recommender.engine.strategies import compare_strategies

    if parquet_path.exists():
        print(f"✓ Found existing dataset: {parquet_path}")
    elif skip_scrape:
        log.error("Dataset missing at %s and --skip-scrape was passed.", parquet_path)
        raise typer.Exit(1)
    else:
        print("No dataset found.")
        print("--> Starting Wikipedia crawl to create dataset...\n")
        from wiki_recommender.nlp.bootstrap import ensure_nltk_data
        from wiki_recommender.scrape.runner import run_crawl

        ensure_nltk_data()
        run_crawl(settings.spider, settings.paths)

    try:
        perform_text_analysis(parquet_path)
        engine = ArticleSimilarityEngine.load(parquet_path, settings.engine)
        generate_model_statistics(engine, settings.paths.plots_dir, dpi=settings.viz.dpi)
        trials = compare_strategies(engine, settings.strategy)
    except WikiRecError as exc:
        raise _handle(exc) from exc

    for result in trials[0]:
        describe_strategy(result)

    if settings.strategy.run_explainability:
        for result in trials[0]:
            explainability_analysis(
                engine,
                result.query_titles,
                result.recommendations,
                strategy_name=result.name,
                plots_dir=settings.paths.plots_dir,
                min_enrichment=settings.explain.min_enrichment,
                top_n_terms=settings.explain.top_n_terms,
                top_distinctive_terms=settings.explain.top_distinctive_terms,
                dpi=settings.viz.dpi,
            )

    summarize_comparison(trials)
    print("\nPipeline completed.")


def _engine_cfg_with_overrides(
    base: EngineConfig,
    text_column: str | None,
    ngram_max: int | None,
    min_df: float | None,
    max_df: float | None,
    no_sublinear_tf: bool,
) -> EngineConfig:
    updates: dict[str, object] = {}
    if text_column is not None:
        updates["text_column"] = text_column
    if ngram_max is not None:
        updates["ngram_max"] = ngram_max
    if min_df is not None:
        # min_df accepts either an int (absolute) or a float (fraction). Typer's
        # CLI is float by default; if the user passes 15 we want it as int.
        updates["min_df"] = int(min_df) if min_df.is_integer() and min_df >= 1 else min_df
    if max_df is not None:
        updates["max_df"] = max_df
    if no_sublinear_tf:
        updates["sublinear_tf"] = False
    return base.model_copy(update=updates)


def _strategy_cfg_with_overrides(
    base: StrategyConfig,
    num_articles: int | None,
    num_trials: int | None,
    explain: bool | None,
) -> StrategyConfig:
    updates: dict[str, object] = {}
    if num_articles is not None:
        updates["num_articles"] = num_articles
    if num_trials is not None:
        updates["num_trials"] = num_trials
    if explain is not None:
        updates["run_explainability"] = explain
    return base.model_copy(update=updates)


if __name__ == "__main__":
    app()
