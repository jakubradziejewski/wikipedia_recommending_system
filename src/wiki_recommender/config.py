"""Typed configuration for the whole pipeline.

Every knob lives here so the CLI, the Scrapy spider, and the analysis modules
share a single source of truth. Overrides flow CLI → env-var → field default;
nested fields are reached over env via ``WIKIREC_<NESTED>__<FIELD>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

TextColumn = Literal["lemmas", "tokens", "stems"]


class Paths(BaseModel):
    """Filesystem layout. Relative paths are resolved against the working directory."""

    data_dir: Path = Field(default=Path("data"))
    plots_dir: Path = Field(default=Path("plots"))
    parquet_name: str = Field(default="wikipedia_articles.parquet")

    @property
    def parquet_path(self) -> Path:
        return self.data_dir / self.parquet_name


class SpiderConfig(BaseModel):
    """Crawl parameters. Defaults reproduce the original pipeline's behavior."""

    start_url: str = Field(default="https://en.wikipedia.org/wiki/Madagascar")
    max_pages: int = Field(default=5000, gt=0)
    max_links_per_page: int = Field(default=10, gt=0)
    concurrent_requests: int = Field(default=32, gt=0)
    download_delay: float = Field(default=0.1, ge=0)
    cookies_enabled: bool = Field(default=False)
    respect_robots: bool = Field(default=True)
    seed: int | None = Field(
        default=None,
        description="If set, BFS link shuffling becomes deterministic. Unset by default — preserves the original behavior.",
    )


class EngineConfig(BaseModel):
    """TF-IDF and similarity-engine parameters."""

    text_column: TextColumn = Field(default="lemmas")
    ngram_max: int = Field(default=2, ge=1, le=4)
    min_df: int | float = Field(default=15)
    max_df: int | float = Field(default=0.60)
    sublinear_tf: bool = Field(default=True)
    top_k: int = Field(default=10, gt=0)

    @property
    def ngram_range(self) -> tuple[int, int]:
        return (1, self.ngram_max)

    @model_validator(mode="after")
    def _df_sanity(self) -> "EngineConfig":
        # Bare-minimum guard: if both are fractions, the lower bound must not exceed the upper.
        # Mixed int/float forms are validated by sklearn at fit time and we shouldn't second-guess it.
        if isinstance(self.min_df, float) and isinstance(self.max_df, float):
            if self.min_df > self.max_df:
                raise ValueError(
                    f"min_df={self.min_df} cannot exceed max_df={self.max_df}"
                )
        return self


class StrategyConfig(BaseModel):
    """Recommendation-strategy comparison parameters."""

    num_articles: int = Field(default=10, gt=0)
    num_trials: int = Field(default=1, gt=0)
    run_explainability: bool = Field(default=True)


class ExplainConfig(BaseModel):
    """Explainability analysis parameters."""

    min_enrichment: float = Field(default=2.0, ge=0)
    top_n_terms: int = Field(default=5, gt=0)
    top_distinctive_terms: int = Field(default=20, gt=0)


class VizConfig(BaseModel):
    """Matplotlib output parameters."""

    dpi: int = Field(default=300, gt=0)


class Settings(BaseSettings):
    """Top-level settings — every field is overridable via env vars."""

    model_config = SettingsConfigDict(
        env_prefix="WIKIREC_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    paths: Paths = Field(default_factory=Paths)
    spider: SpiderConfig = Field(default_factory=SpiderConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    explain: ExplainConfig = Field(default_factory=ExplainConfig)
    viz: VizConfig = Field(default_factory=VizConfig)
