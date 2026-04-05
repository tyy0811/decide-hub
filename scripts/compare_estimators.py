"""Compare IPS vs Doubly Robust estimators on synthetic data.

Runs both estimators across 100 random seeds and reports mean, std, MSE.

Usage: python scripts/compare_estimators.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.evaluation.simulator import generate_logged_data, softmax
from src.evaluation.counterfactual import ips_estimate, clipped_ips_estimate
from src.evaluation.doubly_robust import dr_estimate


def main():
    ips_vals = []
    clipped_vals = []
    dr_vals = []

    for seed in range(100):
        data = generate_logged_data(n_samples=5000, seed=seed)

        # Fit reward model: logistic regression on (context, action) -> reward
        X_model = np.array([
            np.concatenate([ctx, data["item_features"][a]])
            for ctx, a in zip(data["contexts"], data["actions"])
        ])
        y_model = np.array(data["rewards"])
        lr = LogisticRegression(max_iter=200, random_state=seed)
        lr.fit(X_model, y_model)
        model_preds = lr.predict_proba(X_model)[:, 1].tolist()

        # Target policy: greedy (temperature=0.5)
        target_probs = []
        for ctx in data["contexts"]:
            scores = np.array(ctx) @ data["item_features"].T
            probs = softmax(scores, temperature=0.5)
            action = data["actions"][len(target_probs)]
            target_probs.append(float(probs[action]))

        ips_vals.append(ips_estimate(data["rewards"], data["propensities"], target_probs))
        clipped_vals.append(clipped_ips_estimate(data["rewards"], data["propensities"], target_probs))
        dr_vals.append(dr_estimate(data["rewards"], data["propensities"], target_probs, model_preds))

    print("=== Estimator Comparison (100 seeds) ===\n")
    for name, vals in [("IPS", ips_vals), ("Clipped IPS", clipped_vals), ("Doubly Robust", dr_vals)]:
        print(f"{name}:")
        print(f"  Mean:  {np.mean(vals):.4f}")
        print(f"  Std:   {np.std(vals):.4f}")
        print()

    # Variance reduction
    print(f"DR variance reduction vs IPS: {(1 - np.var(dr_vals) / np.var(ips_vals)) * 100:.1f}%")


if __name__ == "__main__":
    main()
