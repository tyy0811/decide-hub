"""MovieLens 1M data loading and temporal split."""

import io
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

DATA_DIR = Path("data/ml-1m")


def download_movielens() -> None:
    """Download MovieLens 1M if not already present."""
    if (DATA_DIR / "ratings.dat").exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    print(f"Downloading MovieLens 1M from {url}...")
    resp = urllib.request.urlopen(url)
    z = zipfile.ZipFile(io.BytesIO(resp.read()))
    z.extractall("data")
    print("Done.")


def load_ratings() -> pl.DataFrame:
    """Load ratings.dat into a Polars DataFrame."""
    download_movielens()
    return pl.read_csv(
        DATA_DIR / "ratings.dat",
        separator="::",
        has_header=False,
        new_columns=["user_id", "movie_id", "rating", "timestamp"],
        schema={"user_id": pl.Int64, "movie_id": pl.Int64,
                "rating": pl.Float64, "timestamp": pl.Int64},
    )


def load_users() -> pl.DataFrame:
    """Load users.dat into a Polars DataFrame."""
    download_movielens()
    return pl.read_csv(
        DATA_DIR / "users.dat",
        separator="::",
        has_header=False,
        new_columns=["user_id", "gender", "age", "occupation", "zip_code"],
    )


def load_movies() -> pl.DataFrame:
    """Load movies.dat into a Polars DataFrame."""
    download_movielens()
    return pl.read_csv(
        DATA_DIR / "movies.dat",
        separator="::",
        has_header=False,
        new_columns=["movie_id", "title", "genres"],
        schema={"movie_id": pl.Int64, "title": pl.Utf8, "genres": pl.Utf8},
    )


def temporal_split(
    ratings: pl.DataFrame, n_test: int = 5,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split ratings by time: last n_test interactions per user as test set."""
    ranked = ratings.with_columns(
        pl.col("timestamp")
        .rank(method="ordinal", descending=True)
        .over("user_id")
        .alias("_rank")
    )
    train = ranked.filter(pl.col("_rank") > n_test).drop("_rank")
    test = ranked.filter(pl.col("_rank") <= n_test).drop("_rank")
    return train, test
