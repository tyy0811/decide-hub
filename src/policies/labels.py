"""Label construction functions for different ranking objectives.

Each function transforms raw ratings into training labels for a specific
objective. Pure functions: DataFrame in, DataFrame out.
"""

import polars as pl


def compute_pltv_labels(
    ratings: pl.DataFrame,
    n_days: int = 30,
    max_timestamp: int | None = None,
) -> pl.DataFrame:
    """Compute predicted Lifetime Value labels.

    For each (user, item, timestamp) interaction, pLTV = sum of that user's
    ratings within n_days after this interaction's timestamp.

    Interactions where the n_days window extends past max_timestamp are
    discarded to prevent temporal leakage into the test set.

    Args:
        ratings: DataFrame with user_id, movie_id, rating, timestamp.
        n_days: Future window in timestamp units.
        max_timestamp: Cutoff — discard interactions where
                       timestamp + n_days > max_timestamp.

    Returns:
        DataFrame with original columns + "pltv" label column,
        filtered to valid rows only.
    """
    if max_timestamp is None:
        max_timestamp = int(ratings["timestamp"].max())

    # Filter to interactions that have a complete future window
    valid = ratings.filter(
        pl.col("timestamp") + n_days <= max_timestamp
    )

    # For each row, compute sum of user's future ratings
    pltv_values = []
    for row in valid.iter_rows(named=True):
        uid = row["user_id"]
        ts = row["timestamp"]
        future = ratings.filter(
            (pl.col("user_id") == uid)
            & (pl.col("timestamp") > ts)
            & (pl.col("timestamp") <= ts + n_days)
        )
        pltv_values.append(future["rating"].sum())

    return valid.with_columns(
        pl.Series(name="pltv", values=pltv_values),
    )
