"""Abstract ranking policy interface."""

from abc import ABC, abstractmethod
import polars as pl


class BasePolicy(ABC):
    """Base class for ranking policies.

    Interface: fit -> observe -> score -> evaluate.
    Outcome logging belongs to the telemetry layer (db.log_outcome),
    not the policy — policies decide what to recommend, not how to persist.
    """

    @abstractmethod
    def fit(self, train_data: pl.DataFrame) -> "BasePolicy":
        """Fit the policy on training data."""
        ...

    @abstractmethod
    def observe(self, context: dict) -> None:
        """Observe user context features before scoring."""
        ...

    @abstractmethod
    def score(self, items: list[int]) -> list[tuple[int, float]]:
        """Score candidate items. Returns [(item_id, score)] sorted descending."""
        ...

    @abstractmethod
    def evaluate(self, test_data: pl.DataFrame, k: int = 10) -> dict[str, float]:
        """Run offline evaluation on held-out test data."""
        ...
