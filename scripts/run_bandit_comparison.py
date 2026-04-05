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
