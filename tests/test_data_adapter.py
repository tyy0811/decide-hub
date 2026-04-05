"""Tests for dataset adapter protocol and implementations."""

import polars as pl
import pytest

from src.policies.data_adapter import MovieLensAdapter, DatasetAdapter

REQUIRED_COLUMNS = {"user_id", "item_id", "rating", "timestamp"}
REQUIRED_TYPES = {"user_id": pl.Int64, "item_id": pl.Int64, "rating": pl.Float64, "timestamp": pl.Int64}


def test_movielens_adapter_is_dataset_adapter():
    """MovieLensAdapter satisfies DatasetAdapter protocol."""
    adapter = MovieLensAdapter()
    assert isinstance(adapter, DatasetAdapter)


def test_movielens_adapter_name():
    adapter = MovieLensAdapter()
    assert adapter.name == "movielens-1m"


def test_movielens_adapter_loads():
    adapter = MovieLensAdapter()
    df = adapter.load()
    assert isinstance(df, pl.DataFrame)
    assert len(df) > 0


def test_movielens_adapter_has_required_columns():
    """Contract test: required columns with correct types."""
    adapter = MovieLensAdapter()
    df = adapter.load()
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_movielens_adapter_column_types():
    adapter = MovieLensAdapter()
    df = adapter.load()
    for col, expected_type in REQUIRED_TYPES.items():
        actual_type = df[col].dtype
        assert actual_type == expected_type or actual_type in (pl.Int32, pl.Float32), \
            f"Column {col}: expected {expected_type}, got {actual_type}"
