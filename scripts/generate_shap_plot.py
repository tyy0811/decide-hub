"""Generate SHAP summary plots for the LightGBM scorer.

Produces beeswarm plots showing feature importance + directionality.
If CF embeddings are enabled, generates a separate comparison plot.

Usage: python scripts/generate_shap_plot.py
Output: docs/shap_summary.png (and docs/shap_summary_cf.png if applicable)
"""

import numpy as np
import shap
from pathlib import Path

from src.policies.data import load_ratings, temporal_split
from src.policies.scorer import ScorerPolicy, FEATURE_COLS
from src.policies.features import build_features, build_training_pairs


def main():
    ratings = load_ratings()
    train, test = temporal_split(ratings, n_test=5)

    # --- Scorer without CF embeddings ---
    print("=== SHAP for Scorer (6 aggregate features) ===")
    policy = ScorerPolicy(n_estimators=50).fit(train)

    features = build_features(train)
    pairs = build_training_pairs(train, features["user_features"], features["item_features"])
    X = pairs.select(FEATURE_COLS).to_numpy()

    # Sample for SHAP (use 500 samples for speed)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X), size=min(500, len(X)), replace=False)
    X_sample = X[sample_idx]

    # LightGBM native SHAP values
    shap_values = policy.model.predict(X_sample, pred_contrib=True)
    # pred_contrib returns shape (n_samples, n_features + 1) — last col is bias
    shap_values = shap_values[:, :-1]

    print(f"SHAP values shape: {shap_values.shape}")
    print(f"Feature columns: {FEATURE_COLS}")

    # Feature importance (mean absolute SHAP)
    importance = np.abs(shap_values).mean(axis=0)
    for col, imp in sorted(zip(FEATURE_COLS, importance), key=lambda x: -x[1]):
        print(f"  {col}: {imp:.4f}")

    # Save plot
    output_path = Path("docs/shap_summary.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap.summary_plot(
            shap_values, X_sample,
            feature_names=FEATURE_COLS,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved: {output_path}")
    except ImportError:
        print("\nmatplotlib not available — SHAP values computed but plot not saved")

    # --- Scorer with CF embeddings ---
    print("\n=== SHAP for Scorer + CF Embeddings ===")
    try:
        policy_cf = ScorerPolicy(
            n_estimators=50, use_embeddings=True, n_embedding_dims=8,
        ).fit(train)

        # Build features with embeddings
        cf_feature_cols = policy_cf._feature_cols
        pairs_cf = build_training_pairs(train, features["user_features"], features["item_features"])

        # Rebuild with embedding columns (reuse policy internals)
        X_cf = pairs_cf.select(FEATURE_COLS).to_numpy()
        # For SHAP with embeddings, we need the full feature matrix
        # Use the model's native SHAP
        shap_values_cf = policy_cf.model.predict(
            X_cf[sample_idx] if len(X_cf) > 500 else X_cf,
            pred_contrib=True,
        )
        shap_values_cf = shap_values_cf[:, :-1]

        importance_cf = np.abs(shap_values_cf).mean(axis=0)
        for i, imp in enumerate(sorted(enumerate(importance_cf), key=lambda x: -x[1])[:10]):
            idx, val = imp
            col_name = cf_feature_cols[idx] if idx < len(cf_feature_cols) else f"feature_{idx}"
            print(f"  {col_name}: {val:.4f}")

        output_cf = Path("docs/shap_summary_cf.png")
        try:
            shap.summary_plot(
                shap_values_cf,
                feature_names=cf_feature_cols[:shap_values_cf.shape[1]],
                show=False,
            )
            plt.tight_layout()
            plt.savefig(str(output_cf), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {output_cf}")
        except Exception:
            pass
    except Exception as e:
        print(f"CF embeddings SHAP failed: {e}")


if __name__ == "__main__":
    main()
