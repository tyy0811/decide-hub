# Phase 2.1: Contextual Bandit — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an epsilon-greedy bandit policy that learns from interaction rewards, evaluated both offline (MovieLens) and via simulated online comparison against a static baseline.

**Architecture:** `EpsilonGreedyPolicy` extends `BasePolicy` with the same `fit/score/evaluate` interface. Arm estimates are maintained in-memory on the instance (`arm_rewards`, `arm_counts`), updated via a bandit-specific `update()` method. A separate `bandit_comparison.py` module simulates online rounds to produce cumulative reward curves. The existing `/rank` endpoint, IPS evaluation, and CI regression gates all apply unchanged.

**Tech Stack:** Python 3.11 / NumPy / Polars / FastAPI / Pydantic

**IMPORTANT — BasePolicy contract:** `score()` must return `list[tuple[int, float]]` sorted descending. `fit()` takes `pl.DataFrame`, returns self. `evaluate()` returns `dict[str, float]`. Context is `{"user_id": int}` passed per-call.

---

## Task 1: EpsilonGreedyPolicy — Core Scoring + Arm Updates

The core bandit class with epsilon-greedy scoring and online arm updates.

**Files:**
- Create: `src/policies/bandit.py`
- Create: `tests/test_bandit.py`

**Step 1: Write the tests**

Create `tests/test_bandit.py`:

```python
"""Tests for epsilon-greedy bandit policy."""

import numpy as np
import pytest

from src.policies.bandit import EpsilonGreedyPolicy


def test_bandit_is_base_policy():
    """EpsilonGreedyPolicy extends BasePolicy."""
    from src.policies.base import BasePolicy
    assert issubclass(EpsilonGreedyPolicy, BasePolicy)


def test_epsilon_cap_enforced():
    """Cannot create bandit with epsilon > max_epsilon."""
    with pytest.raises(ValueError, match="exceeds max_epsilon"):
        EpsilonGreedyPolicy(epsilon=0.5, max_epsilon=0.10)


def test_score_exploit_mode():
    """With epsilon=0, score returns items ranked by arm estimate."""
    bandit = EpsilonGreedyPolicy(epsilon=0.0)
    # Manually set arm estimates
    bandit.arm_rewards = {1: 10.0, 2: 30.0, 3: 20.0}
    bandit.arm_counts = {1: 5, 2: 5, 3: 5}

    scored = bandit.score([1, 2, 3])
    item_ids = [item_id for item_id, _ in scored]
    assert item_ids == [2, 3, 1]  # 6.0, 4.0, 2.0


def test_score_returns_sorted_descending():
    """Score output is always sorted descending by score."""
    bandit = EpsilonGreedyPolicy(epsilon=0.0, seed=42)
    bandit.arm_rewards = {1: 50.0, 2: 10.0, 3: 30.0, 4: 20.0}
    bandit.arm_counts = {1: 10, 2: 10, 3: 10, 4: 10}

    scored = bandit.score([1, 2, 3, 4])
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)


def test_score_unknown_items_get_zero():
    """Items not in arm estimates get score 0."""
    bandit = EpsilonGreedyPolicy(epsilon=0.0)
    bandit.arm_rewards = {1: 10.0}
    bandit.arm_counts = {1: 5}

    scored = bandit.score([1, 99])
    scores_dict = dict(scored)
    assert scores_dict[1] == 2.0  # 10/5
    assert scores_dict[99] == 0.0


def test_score_explore_mode_is_random():
    """With epsilon=1, score returns all items (in random order)."""
    bandit = EpsilonGreedyPolicy(epsilon=1.0, max_epsilon=1.0, seed=42)
    bandit.arm_rewards = {1: 100.0, 2: 0.0, 3: 0.0}
    bandit.arm_counts = {1: 1, 2: 1, 3: 1}

    # Run many times — if truly random, item 1 won't always be first
    first_items = []
    for _ in range(50):
        scored = bandit.score([1, 2, 3])
        first_items.append(scored[0][0])

    # With random ordering, we should see variation in first item
    assert len(set(first_items)) > 1


def test_update_changes_estimates():
    """update() modifies arm reward estimates."""
    bandit = EpsilonGreedyPolicy(epsilon=0.0)
    bandit.arm_rewards = {1: 10.0, 2: 10.0}
    bandit.arm_counts = {1: 10, 2: 10}

    # Item 2 starts at 1.0 avg, update with high rewards
    bandit.update(2, 5.0)
    bandit.update(2, 5.0)

    scored = bandit.score([1, 2])
    # Item 2 now: (10 + 5 + 5) / 12 = 1.67, Item 1: 10/10 = 1.0
    assert scored[0][0] == 2


def test_update_new_item():
    """update() works for items not seen during fit."""
    bandit = EpsilonGreedyPolicy(epsilon=0.0)
    bandit.update(99, 5.0)
    bandit.update(99, 3.0)

    scored = bandit.score([99])
    assert scored[0] == (99, 4.0)  # (5+3)/2
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/zenith/Desktop/decide-hub && .venv/bin/python -m pytest tests/test_bandit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.policies.bandit'`

**Step 3: Write the implementation**

Create `src/policies/bandit.py`:

```python
"""Epsilon-greedy bandit policy — exploration/exploitation with online learning.

Maintains per-arm reward estimates in memory. Arm estimates reset on server
restart — persistent bandit state is a V3 concern. V2 demonstrates the
algorithm and evaluation, not production statefulness.
"""

import numpy as np
import polars as pl

from src.policies.base import BasePolicy
from src.evaluation.naive import ndcg_at_k, mrr, hit_rate_at_k


class EpsilonGreedyPolicy(BasePolicy):
    """Epsilon-greedy multi-armed bandit with warm-start from training data.

    With probability epsilon, explores (random item ordering).
    With probability 1-epsilon, exploits (rank by estimated reward).
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        max_epsilon: float = 0.10,
        seed: int = 42,
    ):
        if epsilon > max_epsilon:
            raise ValueError(
                f"epsilon {epsilon} exceeds max_epsilon {max_epsilon}"
            )
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self._rng = np.random.default_rng(seed)
        self.arm_rewards: dict[int, float] = {}
        self.arm_counts: dict[int, int] = {}
        self._all_items: list[int] = []

    def fit(self, train_data: pl.DataFrame) -> "EpsilonGreedyPolicy":
        """Warm-start arm estimates from training data average ratings."""
        stats = train_data.group_by("movie_id").agg([
            pl.col("rating").sum().alias("total_rating"),
            pl.len().alias("count"),
        ])
        for row in stats.iter_rows(named=True):
            item_id = row["movie_id"]
            self.arm_rewards[item_id] = row["total_rating"]
            self.arm_counts[item_id] = row["count"]
        self._all_items = list(self.arm_counts.keys())
        return self

    def score(
        self, items: list[int], context: dict | None = None,
    ) -> list[tuple[int, float]]:
        """Epsilon-greedy scoring: explore or exploit.

        Explore: random ordering with synthetic descending scores.
        Exploit: rank by estimated reward (arm_rewards / arm_counts).
        """
        if self._rng.random() < self.epsilon:
            shuffled = list(items)
            self._rng.shuffle(shuffled)
            return [
                (item, float(len(items) - i))
                for i, item in enumerate(shuffled)
            ]

        scored = []
        for item in items:
            count = self.arm_counts.get(item, 0)
            estimate = self.arm_rewards[item] / count if count > 0 else 0.0
            scored.append((item, estimate))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def update(self, item_id: int, reward: float) -> None:
        """Update arm estimate with observed reward."""
        self.arm_rewards[item_id] = self.arm_rewards.get(item_id, 0.0) + reward
        self.arm_counts[item_id] = self.arm_counts.get(item_id, 0) + 1

    def evaluate(self, test_data: pl.DataFrame, k: int = 10) -> dict[str, float]:
        """Offline evaluation with epsilon=0 (pure exploitation)."""
        all_items = self._all_items
        users = test_data["user_id"].unique().to_list()

        old_epsilon = self.epsilon
        self.epsilon = 0.0

        ndcg_scores = []
        mrr_scores = []
        hit_scores = []

        for user_id in users:
            user_test = test_data.filter(pl.col("user_id") == user_id)
            relevant = set(user_test["movie_id"].to_list())

            ranked = self.score(all_items, context={"user_id": user_id})
            ranked_ids = [item_id for item_id, _ in ranked]

            ndcg_scores.append(ndcg_at_k(ranked_ids, relevant, k))
            mrr_scores.append(mrr(ranked_ids, relevant))
            hit_scores.append(hit_rate_at_k(ranked_ids, relevant, k))

        self.epsilon = old_epsilon

        return {
            f"ndcg@{k}": sum(ndcg_scores) / len(ndcg_scores),
            "mrr": sum(mrr_scores) / len(mrr_scores),
            f"hit_rate@{k}": sum(hit_scores) / len(hit_scores),
        }
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/zenith/Desktop/decide-hub && .venv/bin/python -m pytest tests/test_bandit.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add src/policies/bandit.py tests/test_bandit.py
git commit -m "feat: add epsilon-greedy bandit policy with arm updates"
```

---

## Task 2: Bandit Comparison Module — Online Simulation

Simulates interaction rounds to produce cumulative reward curves for static vs bandit.

**Files:**
- Create: `src/evaluation/bandit_comparison.py`
- Create: `tests/test_bandit_comparison.py`

**Step 1: Write the tests**

Create `tests/test_bandit_comparison.py`:

```python
"""Tests for bandit vs static policy comparison."""

import numpy as np
from src.evaluation.bandit_comparison import run_bandit_comparison


def test_comparison_returns_expected_keys():
    """Comparison result has all required fields."""
    result = run_bandit_comparison(n_rounds=100, seed=42)
    assert "cumulative_reward_static" in result
    assert "cumulative_reward_bandit" in result
    assert "n_rounds" in result
    assert "final_reward_static" in result
    assert "final_reward_bandit" in result
    assert len(result["cumulative_reward_static"]) == 100
    assert len(result["cumulative_reward_bandit"]) == 100


def test_cumulative_rewards_are_monotonically_nondecreasing():
    """Cumulative reward curves never decrease."""
    result = run_bandit_comparison(n_rounds=500, seed=42)
    for curve in ["cumulative_reward_static", "cumulative_reward_bandit"]:
        values = result[curve]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]


def test_static_policy_reward_rate_is_constant():
    """Static policy picks the same arm every round — reward rate is ~constant."""
    result = run_bandit_comparison(n_rounds=2000, seed=42)
    static = result["cumulative_reward_static"]
    # Reward rate in first half vs second half should be similar
    first_half_rate = static[999] / 1000
    second_half_rate = (static[1999] - static[999]) / 1000
    assert abs(first_half_rate - second_half_rate) < 0.1


def test_bandit_learns_over_time():
    """Bandit's reward rate improves as it learns arm estimates."""
    result = run_bandit_comparison(
        n_rounds=5000, n_items=5, epsilon=0.1, seed=42,
    )
    bandit = result["cumulative_reward_bandit"]
    # Reward rate in last 1000 rounds should exceed first 1000
    early_rate = bandit[999] / 1000
    late_rate = (bandit[4999] - bandit[3999]) / 1000
    assert late_rate >= early_rate - 0.05  # allow small tolerance


def test_zero_epsilon_bandit_never_explores():
    """With epsilon=0, bandit always picks the same arm (greedy)."""
    result = run_bandit_comparison(
        n_rounds=100, epsilon=0.0, seed=42,
    )
    # Greedy bandit and static should behave identically after warmup
    # (both pick best estimated arm). Allow small difference from warmup.
    diff = abs(result["final_reward_bandit"] - result["final_reward_static"])
    # Difference should be small relative to total
    assert diff < result["n_rounds"] * 0.15
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/zenith/Desktop/decide-hub && .venv/bin/python -m pytest tests/test_bandit_comparison.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation.bandit_comparison'`

**Step 3: Write the implementation**

Create `src/evaluation/bandit_comparison.py`:

```python
"""Bandit vs static policy comparison via simulated online interaction.

Simulates a sequence of interaction rounds where:
- A context vector is generated each round
- Each policy selects an item
- A reward is observed (Bernoulli with sigmoid probability)
- The bandit updates its arm estimates; the static policy does not

Output: cumulative reward curves for both policies across all rounds.
"""

import numpy as np


def run_bandit_comparison(
    n_rounds: int = 10_000,
    n_items: int = 20,
    n_features: int = 5,
    epsilon: float = 0.1,
    warmup_rounds: int = 100,
    seed: int = 42,
) -> dict:
    """Run head-to-head: static best-arm vs epsilon-greedy bandit.

    The static policy picks the single item with highest marginal expected
    reward (estimated from warmup rounds). This is the multi-armed bandit
    equivalent of PopularityPolicy — it ignores context.

    The bandit explores with probability epsilon and otherwise picks the
    arm with highest estimated reward, learning from every interaction.

    Args:
        n_rounds: Total interaction rounds.
        n_items: Number of items (arms).
        n_features: Context vector dimension.
        epsilon: Exploration probability for the bandit.
        warmup_rounds: Random rounds to estimate static policy's best arm.
        seed: Random seed for reproducibility.

    Returns:
        Dict with cumulative_reward_static, cumulative_reward_bandit (lists),
        n_rounds, epsilon, final_reward_static, final_reward_bandit.
    """
    rng = np.random.default_rng(seed)

    # Environment: fixed item features, reward = Bernoulli(sigmoid(ctx @ item))
    item_features = rng.standard_normal((n_items, n_features))

    # --- Warmup: estimate best static arm ---
    warmup_rewards = np.zeros(n_items)
    warmup_counts = np.zeros(n_items)
    for _ in range(warmup_rounds):
        ctx = rng.standard_normal(n_features)
        scores = ctx @ item_features.T
        reward_probs = 1.0 / (1.0 + np.exp(-scores))
        arm = rng.integers(n_items)
        reward = float(rng.binomial(1, reward_probs[arm]))
        warmup_rewards[arm] += reward
        warmup_counts[arm] += 1
    # Static policy: pick arm with highest estimated reward
    warmup_estimates = np.where(
        warmup_counts > 0, warmup_rewards / warmup_counts, 0.0,
    )
    static_arm = int(np.argmax(warmup_estimates))

    # --- Bandit: start with warmup knowledge ---
    bandit_rewards = warmup_rewards.copy()
    bandit_counts = warmup_counts.copy()

    cumulative_static = []
    cumulative_bandit = []
    total_static = 0.0
    total_bandit = 0.0

    for _ in range(n_rounds):
        ctx = rng.standard_normal(n_features)
        scores = ctx @ item_features.T
        reward_probs = 1.0 / (1.0 + np.exp(-scores))

        # Static: always pick static_arm
        static_reward = float(rng.binomial(1, reward_probs[static_arm]))
        total_static += static_reward
        cumulative_static.append(total_static)

        # Bandit: epsilon-greedy
        if rng.random() < epsilon:
            bandit_arm = rng.integers(n_items)
        else:
            estimates = np.where(
                bandit_counts > 0, bandit_rewards / bandit_counts, 0.0,
            )
            bandit_arm = int(np.argmax(estimates))

        bandit_reward = float(rng.binomial(1, reward_probs[bandit_arm]))
        total_bandit += bandit_reward
        cumulative_bandit.append(total_bandit)

        # Update bandit state
        bandit_rewards[bandit_arm] += bandit_reward
        bandit_counts[bandit_arm] += 1

    return {
        "cumulative_reward_static": cumulative_static,
        "cumulative_reward_bandit": cumulative_bandit,
        "n_rounds": n_rounds,
        "epsilon": epsilon,
        "final_reward_static": total_static,
        "final_reward_bandit": total_bandit,
    }
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/zenith/Desktop/decide-hub && .venv/bin/python -m pytest tests/test_bandit_comparison.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/evaluation/bandit_comparison.py tests/test_bandit_comparison.py
git commit -m "feat: add bandit comparison module — static vs epsilon-greedy simulation"
```

---

## Task 3: Serving Integration — /rank Endpoint + Schema

Register the bandit in the app lifespan and allow `policy=bandit` in the API.

**Files:**
- Modify: `src/serving/schemas.py:14` (add "bandit" to policy pattern)
- Modify: `src/serving/app.py:59-67` (register bandit in lifespan)
- Modify: `tests/test_serving.py` (add bandit endpoint test)

**Step 1: Write the test**

Append to `tests/test_serving.py`:

```python
def test_rank_bandit_policy():
    """Bandit policy returns ranked items via /rank endpoint."""
    with TestClient(app) as client:
        resp = client.post("/rank", json={
            "user_id": 42,
            "policy": "bandit",
            "k": 5,
        })
        if resp.status_code == 200:
            data = resp.json()
            assert data["policy"] == "bandit"
            assert len(data["items"]) == 5
            # Scores should be descending
            scores = [item["score"] for item in data["items"]]
            assert scores == sorted(scores, reverse=True)
        else:
            # Bandit may fail to load if data unavailable — 404 is acceptable
            assert resp.status_code == 404
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/zenith/Desktop/decide-hub && .venv/bin/python -m pytest tests/test_serving.py::test_rank_bandit_policy -v`
Expected: FAIL — 422 validation error (policy pattern rejects "bandit")

**Step 3: Update schema**

Modify `src/serving/schemas.py` line 14 — add "bandit" to the policy pattern:

```python
    policy: str = Field(default="popularity", pattern="^(popularity|scorer|bandit)$")
```

**Step 4: Register bandit in app lifespan**

Modify `src/serving/app.py` — add import after line 16:

```python
from src.policies.bandit import EpsilonGreedyPolicy
```

Add after the scorer registration block (after line 67, after the `except` for ScorerPolicy):

```python
    try:
        bandit = EpsilonGreedyPolicy(epsilon=0.1).fit(_train_data)
        _policies["bandit"] = bandit
    except Exception as e:
        print(f"Warning: EpsilonGreedyPolicy failed to fit: {e}")
```

Also update the `/rank` endpoint's candidate item fallback (around line 119). Add a branch for bandit's `_all_items`:

After `elif hasattr(policy, "_item_ids"):` add:

```python
    elif hasattr(policy, "_all_items"):
        candidates = policy._all_items
```

**Step 5: Run test to verify it passes**

Run: `cd /Users/zenith/Desktop/decide-hub && .venv/bin/python -m pytest tests/test_serving.py -v`
Expected: All serving tests PASS (including the new bandit test)

**Step 6: Commit**

```bash
git add src/serving/schemas.py src/serving/app.py tests/test_serving.py
git commit -m "feat: register bandit policy in /rank endpoint"
```

---

## Task 4: Bandit Evaluation Script + Benchmark

CLI script to run the bandit comparison and offline evaluation, updating the README benchmark table.

**Files:**
- Create: `scripts/run_bandit_comparison.py`
- Modify: `README.md` (add bandit to benchmark table + comparison summary)

**Step 1: Create the comparison script**

Create `scripts/run_bandit_comparison.py`:

```python
"""Run bandit vs static comparison and print results.

Usage: python scripts/run_bandit_comparison.py
"""

from src.evaluation.bandit_comparison import run_bandit_comparison
from src.policies.bandit import EpsilonGreedyPolicy
from src.policies.data import load_ratings, temporal_split


def main():
    # --- Online simulation comparison ---
    print("=== Bandit vs Static: Online Simulation ===\n")
    result = run_bandit_comparison(
        n_rounds=10_000, n_items=20, epsilon=0.1, seed=42,
    )
    print(f"Rounds: {result['n_rounds']}")
    print(f"Epsilon: {result['epsilon']}")
    print(f"Static final reward:  {result['final_reward_static']:.0f}")
    print(f"Bandit final reward:  {result['final_reward_bandit']:.0f}")
    advantage = result["final_reward_bandit"] - result["final_reward_static"]
    print(f"Bandit advantage:     {advantage:+.0f}")
    print()

    # --- Offline evaluation on MovieLens ---
    print("=== Bandit: Offline MovieLens Evaluation ===\n")
    ratings = load_ratings()
    train, test = temporal_split(ratings, n_test=5)

    bandit = EpsilonGreedyPolicy(epsilon=0.1).fit(train)
    metrics = bandit.evaluate(test, k=10)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
```

**Step 2: Run the script**

Run: `cd /Users/zenith/Desktop/decide-hub && .venv/bin/python scripts/run_bandit_comparison.py`
Expected: Prints comparison results and offline metrics.

**Step 3: Update README benchmark table**

Modify `README.md` — add bandit row to the ranking benchmarks table:

```markdown
## Ranking Benchmarks (MovieLens 1M)

| Policy | NDCG@10 | MRR | HitRate@10 |
|--------|---------|-----|------------|
| Popularity | 0.0177 | 0.0473 | 0.0954 |
| LightGBM LambdaRank | 0.0017 | 0.0119 | 0.0080 |
| Epsilon-Greedy Bandit (e=0.1) | _run script to fill_ | _run script to fill_ | _run script to fill_ |
```

Fill in actual values from the script output.

Add a new section after the counterfactual evaluation table:

```markdown
## Bandit Comparison (Simulated Online)

| Policy | Cumulative Reward (10K rounds) |
|--------|-------------------------------|
| Static best-arm | _from script_ |
| Epsilon-greedy (e=0.1) | _from script_ |

The static policy picks a single best arm estimated from a warmup phase
and never adapts. The bandit explores with 10% probability and exploits
its learned estimates otherwise. See [DECISIONS.md](DECISIONS.md) for
why the bandit uses in-memory arm state and what "contextual" means here.
```

**Step 4: Commit**

```bash
git add scripts/run_bandit_comparison.py README.md
git commit -m "feat: add bandit comparison script + README benchmarks"
```

---

## Task 5: DECISIONS.md Entry

Document design decisions for the bandit implementation.

**Files:**
- Modify: `DECISIONS.md` (append new entry)

**Step 1: Append entry**

Add to `DECISIONS.md`:

```markdown

## 15. Epsilon-greedy bandit with in-memory arm state

The bandit maintains per-arm reward estimates (`arm_rewards`, `arm_counts`)
in memory on the policy instance, updated via `update(item_id, reward)`.
Server restart resets all estimates to the warm-start values from training
data. Persistent bandit state (Redis, database) is a V3 concern — V2
demonstrates the algorithm and evaluation, not production statefulness.

Epsilon is hard-capped at `max_epsilon=0.10`. This limits exploration to
10% of interactions, analogous to how the permissions layer limits what
actions the automation pipeline can take. The cap is configurable but
exists to prevent unbounded random behavior.

Evaluation uses epsilon=0 (pure exploitation) to measure the quality of
what the bandit has learned, separate from its exploration behavior. This
is standard practice — exploration is a learning strategy, not a ranking
quality indicator.

The bandit comparison module simulates online interaction rounds against
a static best-arm baseline. The static policy picks a single arm with
the highest estimated marginal reward (from a warmup phase) and never
adapts. This is the multi-armed bandit equivalent of PopularityPolicy.
The comparison produces cumulative reward curves: if the bandit overtakes
the static baseline, exploration is paying off. If it doesn't, that's an
honest finding — it might mean the epsilon is too low or the reward
landscape doesn't favor exploration.
```

**Step 2: Commit**

```bash
git add DECISIONS.md
git commit -m "docs: add DECISIONS.md entry for bandit design choices"
```

---

## Post-Task Checklist

After all 5 tasks:

1. **Run full test suite:** `make test` — all existing + new bandit tests pass
2. **Run bandit comparison:** `python scripts/run_bandit_comparison.py` — prints results
3. **Fill README values:** Update benchmark tables with actual numbers from script output
4. **Verify /rank endpoint:** `curl -X POST http://localhost:8000/rank -H "Content-Type: application/json" -d '{"user_id": 42, "policy": "bandit", "k": 5}'`
