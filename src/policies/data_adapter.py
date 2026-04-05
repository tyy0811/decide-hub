"""Dataset adapter protocol — generalizes data loading across datasets.

Each adapter loads a ratings dataset into a standardized DataFrame with
columns: user_id, item_id, rating, timestamp.
"""

from typing import Protocol, runtime_checkable

import polars as pl

from src.policies.data import load_ratings


@runtime_checkable
class DatasetAdapter(Protocol):
    """Protocol for dataset adapters."""

    @property
    def name(self) -> str: ...

    def load(self) -> pl.DataFrame: ...


class MovieLensAdapter:
    """Adapter for MovieLens 1M dataset."""

    @property
    def name(self) -> str:
        return "movielens-1m"

    def load(self) -> pl.DataFrame:
        df = load_ratings()
        # Rename movie_id to item_id for standardized interface
        return df.rename({"movie_id": "item_id"})


class AmazonBooksAdapter:
    """Adapter for Amazon Reviews (Books subset).

    Downloads a pre-filtered ~50K sample on first use.
    """

    @property
    def name(self) -> str:
        return "amazon-books"

    def load(self) -> pl.DataFrame:
        from pathlib import Path
        data_dir = Path("data/amazon-books")
        ratings_path = data_dir / "ratings.csv"

        if not ratings_path.exists():
            raise FileNotFoundError(
                f"Amazon Books dataset not found at {ratings_path}. "
                "Run: python scripts/download_amazon_reviews.py"
            )

        df = pl.read_csv(ratings_path)
        # Ensure standardized column names and types
        return df.select([
            pl.col("user_id").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
            pl.col("rating").cast(pl.Float64),
            pl.col("timestamp").cast(pl.Int64),
        ])
