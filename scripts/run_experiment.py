"""Run A/B experiment: control (logging policy) vs treatment (scorer-like policy).

Uses the existing simulator to generate data. If bandit policy is available,
includes bandit vs scorer comparison for a richer demo.

Usage: python scripts/run_experiment.py
"""

import numpy as np

from src.evaluation.simulator import generate_logged_data, softmax
from src.evaluation.experiment import run_experiment, minimum_detectable_effect
from src.evaluation.kpi import value_proxy, retention_proxy
from src.evaluation.report import render_markdown


def main():
    # Generate logged data under logging policy (temperature=1.0)
    data = generate_logged_data(n_samples=10_000, seed=42)

    # Control: logging policy rewards (as-is)
    control_rewards = np.array(data["rewards"])

    # Treatment: simulate a "better" policy (lower temperature = more greedy)
    rng = np.random.default_rng(99)
    item_features = data["item_features"]
    treatment_rewards = []
    for ctx in data["contexts"]:
        scores = np.array(ctx) @ item_features.T
        probs = softmax(scores, temperature=0.5)  # more greedy
        action = rng.choice(data["n_items"], p=probs)
        reward_prob = 1.0 / (1.0 + np.exp(-scores[action]))
        treatment_rewards.append(float(rng.binomial(1, reward_prob)))
    treatment_rewards = np.array(treatment_rewards)

    # --- Raw reward experiment ---
    print("=" * 60)
    print("EXPERIMENT: Control (temp=1.0) vs Treatment (temp=0.5)")
    print("=" * 60)
    result = run_experiment(control_rewards, treatment_rewards, seed=42)
    result["mde"] = minimum_detectable_effect(
        n=len(control_rewards),
        baseline_std=float(control_rewards.std()),
    )
    print(render_markdown(result))

    # --- KPI: retention proxy ---
    print("=" * 60)
    print("KPI: Retention Proxy (engagement > 0.5)")
    print("=" * 60)
    control_retention = np.array(retention_proxy(control_rewards.tolist()))
    treatment_retention = np.array(retention_proxy(treatment_rewards.tolist()))
    retention_result = run_experiment(control_retention, treatment_retention, seed=42)
    print(render_markdown(retention_result))

    # --- Segmented analysis ---
    print("=" * 60)
    print("SEGMENTED: By reward quartile (proxy for user activity)")
    print("=" * 60)
    # Segment by context feature (first dimension sign → proxy for user type)
    contexts = data["contexts"]
    segments_c = ["high_affinity" if ctx[0] > 0 else "low_affinity" for ctx in contexts]
    segments_t = ["high_affinity" if ctx[0] > 0 else "low_affinity" for ctx in contexts]
    seg_result = run_experiment(
        control_rewards, treatment_rewards,
        segments_control=segments_c,
        segments_treatment=segments_t,
        seed=42,
    )
    print(render_markdown(seg_result))


if __name__ == "__main__":
    main()
