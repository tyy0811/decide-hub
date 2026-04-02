"""Feature engineering for MovieLens — user and item features for LightGBM."""

import polars as pl


def build_features(ratings: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Build user and item feature DataFrames from ratings."""

    # User features: average rating, rating count, rating std
    user_features = ratings.group_by("user_id").agg(
        pl.col("rating").mean().alias("user_avg_rating"),
        pl.col("rating").count().alias("user_rating_count"),
        pl.col("rating").std().alias("user_rating_std"),
    ).with_columns(
        pl.col("user_rating_std").fill_null(0.0),
    )

    # Item features: average rating, popularity (count), rating std
    item_features = ratings.group_by("movie_id").agg(
        pl.col("rating").mean().alias("item_avg_rating"),
        pl.col("rating").count().alias("item_popularity"),
        pl.col("rating").std().alias("item_rating_std"),
    ).with_columns(
        pl.col("item_rating_std").fill_null(0.0),
    )

    return {
        "user_features": user_features,
        "item_features": item_features,
    }


def build_training_pairs(
    ratings: pl.DataFrame,
    user_features: pl.DataFrame,
    item_features: pl.DataFrame,
) -> pl.DataFrame:
    """Build (user_features, item_features) -> rating training pairs."""
    return (
        ratings
        .join(user_features, on="user_id", how="left")
        .join(item_features, on="movie_id", how="left")
    )
