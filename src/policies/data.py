"""MovieLens 1M data loading and temporal split."""

import io
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "ml-1m"


_DOWNLOAD_TIMEOUT = 120  # seconds


def download_movielens() -> None:
    """Download MovieLens 1M if not already present."""
    if (DATA_DIR / "ratings.dat").exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    print(f"Downloading MovieLens 1M from {url}...")
    try:
        resp = urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT)
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        z.extractall(_PROJECT_ROOT / "data")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Failed to download MovieLens 1M: {e}\n"
            f"Download manually from {url}, extract to {DATA_DIR}"
        ) from e
    except TimeoutError:
        raise RuntimeError(
            f"MovieLens download timed out after {_DOWNLOAD_TIMEOUT}s.\n"
            f"Download manually from {url}, extract to {DATA_DIR}"
        )
    print("Done.")


def _read_dat(path: Path, columns: list[str], dtypes: dict[str, type]) -> pl.DataFrame:
    """Read a :: separated .dat file (Polars requires single-byte separators)."""
    text = path.read_text(encoding="latin-1")
    rows = [line.split("::") for line in text.strip().split("\n")]
    data = {col: [dtypes[col](row[i]) for row in rows] for i, col in enumerate(columns)}
    return pl.DataFrame(data)


def load_ratings() -> pl.DataFrame:
    """Load ratings.dat into a Polars DataFrame."""
    download_movielens()
    return _read_dat(
        DATA_DIR / "ratings.dat",
        columns=["user_id", "movie_id", "rating", "timestamp"],
        dtypes={"user_id": int, "movie_id": int, "rating": float, "timestamp": int},
    )


def load_users() -> pl.DataFrame:
    """Load users.dat into a Polars DataFrame."""
    download_movielens()
    return _read_dat(
        DATA_DIR / "users.dat",
        columns=["user_id", "gender", "age", "occupation", "zip_code"],
        dtypes={"user_id": int, "gender": str, "age": int, "occupation": int, "zip_code": str},
    )


def load_movies() -> pl.DataFrame:
    """Load movies.dat into a Polars DataFrame."""
    download_movielens()
    return _read_dat(
        DATA_DIR / "movies.dat",
        columns=["movie_id", "title", "genres"],
        dtypes={"movie_id": int, "title": str, "genres": str},
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
