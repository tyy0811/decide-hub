"""Abstract ranking policy interface."""

from abc import ABC, abstractmethod
import polars as pl


class BasePolicy(ABC):
    """Base class for ranking policies.

    Interface: observe -> score -> log_outcome -> evaluate
    Scoped to ranking only. Automation module has its own pipeline shape.
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
    def log_outcome(self, user_id: int, item_id: int, reward: float) -> None:
        """Log an observed outcome."""
        ...

    @abstractmethod
    def evaluate(self, test_data: pl.DataFrame, k: int = 10) -> dict[str, float]:
        """Run offline evaluation on held-out test data."""
        ...
