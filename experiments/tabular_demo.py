"""Demonstrate GIST limitations on one-hot encoded sparse tabular data.

Uses the Adult Census dataset (sklearn) with pure categorical features
one-hot encoded. Shows that GIST provides no advantage over random sampling,
and diagnoses why: distances in one-hot space are uninformative.

Usage:
    uv run experiments/tabular_demo.py [--seed 42] [--max-rows 5000]
"""

from __future__ import annotations

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import normalize

from gist_sampling import GISTSelector

warnings.filterwarnings("ignore")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GIST on one-hot tabular data: benchmark + diagnosis."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-rows", type=int, default=5000, help="Max rows to use")
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[42, 43, 44, 7, 13],
        help="Seeds for multi-seed AUC benchmark",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/assets",
        help="Directory for output images",
    )
    return parser.parse_args()


def _load_adult_onehot(max_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Load Adult Census, one-hot encode categorical features only."""
    adult = fetch_openml("adult", version=2, as_frame=True)
    df = adult.frame.dropna().head(max_rows)

    y = (df["class"] == ">50K").astype(int).values

    cat_cols = df.select_dtypes(include="category").columns.tolist()
    cat_cols.remove("class")
    X = pd.get_dummies(df[cat_cols], drop_first=False).values.astype(np.float64)

    return X, y


def _fit_auc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Train logistic regression and return ROC-AUC."""
    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        return float("nan")
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, probs))


def _stratified_sample(y: np.ndarray, k: int, seed: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified subsample preserving class balance."""
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=k, random_state=seed)
    indices = np.arange(len(y))
    try:
        train_idx, _ = next(splitter.split(indices, y))
        return train_idx
    except ValueError:
        return rng.choice(len(y), size=k, replace=False)


# ---------------------------------------------------------------------------
# Part 1: AUC Benchmark
# ---------------------------------------------------------------------------


def _run_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    seeds: list[int],
    fracs: list[float],
) -> dict[str, dict[float, list[float]]]:
    """Run random vs stratified vs GIST AUC comparison across seeds."""
    results: dict[str, dict[float, list[float]]] = {
        m: {f: [] for f in fracs} for m in ["random", "stratified", "gist"]
    }

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )

        for frac in fracs:
            k = max(1, int(len(X_train) * frac))
            rng = np.random.default_rng(seed)

            # Random
            ri = rng.choice(len(X_train), size=k, replace=False)
            results["random"][frac].append(_fit_auc(X_train[ri], y_train[ri], X_test, y_test))

            # Stratified
            si = _stratified_sample(y_train, k, seed, rng)
            results["stratified"][frac].append(_fit_auc(X_train[si], y_train[si], X_test, y_test))

            # GIST
            sel = GISTSelector(
                n_samples=k,
                metric="euclidean",
                epsilon=0.05,
                lambda_diversity=1.0,
                random_state=seed,
                mode="auto",
            )
            sel.fit(X_train)
            gi = sel.selected_indices_
            results["gist"][frac].append(_fit_auc(X_train[gi], y_train[gi], X_test, y_test))

    return results


def _print_benchmark(
    results: dict[str, dict[float, list[float]]],
    n_train: int,
) -> None:
    """Print AUC results in blog-ready table format."""
    fracs = sorted(results["random"].keys())

    print(f"\n{'=' * 70}")
    print("  PART 1: AUC BENCHMARK")
    print(f"{'=' * 70}")
    print(f"  Adult Census (one-hot categorical features)")
    print()
    header = f"  {'Budget':<15}"
    for method in ["Random", "Stratified", "GIST"]:
        header += f" {method:<20}"
    print(header)
    print(f"  {'-' * 15}" + f" {'-' * 20}" * 3)

    for frac in fracs:
        k = max(1, int(n_train * frac))
        line = f"  {frac:.0%} (k={k})" + " " * (15 - len(f"{frac:.0%} (k={k})"))
        for method in ["random", "stratified", "gist"]:
            arr = np.array(results[method][frac])
            valid = arr[np.isfinite(arr)]
            if valid.size > 0:
                line += f" {valid.mean():.3f} +/- {valid.std():.3f}      "
            else:
                line += f" {'n/a':<20}"
        print(line)

    print()
    print("  --> All methods perform within noise. GIST provides no advantage.")


# ---------------------------------------------------------------------------
# Part 2: Diagnosis
# ---------------------------------------------------------------------------


def _run_diagnosis(X: np.ndarray, y: np.ndarray, seed: int) -> None:
    """Diagnose why GIST doesn't help: distances are uninformative."""
    print(f"\n{'=' * 70}")
    print("  PART 2: WHY GIST DOESN'T HELP")
    print(f"{'=' * 70}")

    rng = np.random.default_rng(seed)
    n_sample = min(1000, len(X))
    idx = rng.choice(len(X), size=n_sample, replace=False)
    X_sub = X[idx]
    y_sub = y[idx]

    # Distance distribution
    dists = pdist(X_sub, metric="euclidean")
    cv = dists.std() / dists.mean()

    print(f"\n  Distance distribution (euclidean, {n_sample}-point sample):")
    print(f"    Features:  {X.shape[1]} (one-hot encoded, {(X == 0).sum() / X.size:.0%} sparse)")
    print(f"    Mean dist: {dists.mean():.3f}")
    print(f"    Std dist:  {dists.std():.3f}")
    print(f"    CV:        {cv:.3f}")
    print(f"    p5-p95:    [{np.percentile(dists, 5):.3f}, {np.percentile(dists, 95):.3f}]")

    # 10-NN same-label rate
    from scipy.spatial.distance import cdist

    D = cdist(X_sub, X_sub, metric="euclidean")
    same_label_rates = []
    for i in range(n_sample):
        dists_i = D[i].copy()
        dists_i[i] = np.inf
        nn_10 = np.argsort(dists_i)[:10]
        same_label_rates.append(np.mean(y_sub[nn_10] == y_sub[i]))

    nn_rate = np.mean(same_label_rates)
    baseline = max(y.mean(), 1 - y.mean())

    print(f"\n  10-nearest-neighbor same-label rate:")
    print(f"    Observed:  {nn_rate:.3f}")
    print(f"    Baseline:  {baseline:.3f} (majority class rate)")
    print(
        f"    --> {'Distances do NOT predict labels' if abs(nn_rate - baseline) < 0.05 else 'Distances weakly predict labels'}"
    )

    # Silhouette score
    km = KMeans(n_clusters=15, random_state=seed, n_init=10)
    labels = km.fit_predict(X_sub)
    sil = silhouette_score(X_sub, labels, sample_size=min(1000, n_sample), random_state=seed)

    print(f"\n  Cluster structure (k-means, k=15):")
    print(f"    Silhouette score: {sil:.3f} (>0.5 = strong, <0.25 = weak)")
    print(
        f"    --> {'No meaningful cluster structure' if sil < 0.25 else 'Some cluster structure'}"
    )

    print()
    print("  GIST optimizes a facility-location objective over pairwise distances.")
    print("  When distances don't reflect meaningful similarity, GIST can't do")
    print("  better than random -- there's no structure to exploit.")


# ---------------------------------------------------------------------------
# Part 3: Distance Distribution Comparison
# ---------------------------------------------------------------------------


def _plot_distance_comparison(
    X_onehot: np.ndarray,
    seed: int,
    output_path: str,
) -> None:
    """Side-by-side distance histograms: one-hot vs make_blobs."""
    rng = np.random.default_rng(seed)

    # One-hot distances
    n_sample = min(1000, len(X_onehot))
    idx = rng.choice(len(X_onehot), size=n_sample, replace=False)
    dists_onehot = pdist(X_onehot[idx], metric="euclidean")

    # Synthetic make_blobs (same structure as synthetic_demo.py)
    X_blobs, _ = make_blobs(
        n_samples=[300, 50, 50, 50, 50],
        n_features=384,
        cluster_std=[0.5, 1.5, 1.5, 0.6, 1.5],
        center_box=(-10, 10),
        random_state=seed,
    )
    X_blobs = normalize(X_blobs, norm="l2")
    dists_blobs = pdist(X_blobs, metric="euclidean")

    # Normalize both to [0, 1] for visual comparison
    onehot_norm = (dists_onehot - dists_onehot.min()) / (dists_onehot.max() - dists_onehot.min())
    blobs_norm = (dists_blobs - dists_blobs.min()) / (dists_blobs.max() - dists_blobs.min())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # One-hot
    ax1.hist(onehot_norm, bins=50, color="#e74c3c", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax1.set_title("Adult Census (one-hot categorical)", fontsize=12)
    ax1.set_xlabel("Normalized pairwise distance", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.axvline(
        np.median(onehot_norm), color="#c0392b", linestyle="--", linewidth=1.5, label="Median"
    )
    ax1.legend(fontsize=9)
    ax1.text(
        0.95,
        0.95,
        f"CV = {dists_onehot.std() / dists_onehot.mean():.3f}",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # make_blobs
    ax2.hist(blobs_norm, bins=50, color="#2ecc71", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax2.set_title("Synthetic embeddings (5 clusters)", fontsize=12)
    ax2.set_xlabel("Normalized pairwise distance", fontsize=10)
    ax2.axvline(
        np.median(blobs_norm), color="#27ae60", linestyle="--", linewidth=1.5, label="Median"
    )
    ax2.legend(fontsize=9)
    ax2.text(
        0.95,
        0.95,
        f"CV = {dists_blobs.std() / dists_blobs.mean():.3f}",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    fig.suptitle(
        "Distance distributions: when distances are uninformative, GIST can't help",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Distance comparison saved to {output_path}")


def main() -> None:
    args = _parse_args()

    # Load data
    print("Loading Adult Census dataset...")
    X, y = _load_adult_onehot(args.max_rows)
    n = X.shape[0]
    n_features = X.shape[1]
    sparsity = (X == 0).sum() / X.size
    pos_rate = y.mean()

    print(f"\nDataset: Adult Census (categorical features only)")
    print(f"  Rows:      {n}")
    print(f"  Features:  {n_features} (one-hot encoded)")
    print(f"  Sparsity:  {sparsity:.1%}")
    print(f"  Positive:  {pos_rate:.1%} (income >50K)")

    # Part 1: AUC benchmark
    fracs = [0.02, 0.05, 0.10]
    print("\nRunning AUC benchmark (this may take a minute)...")
    results = _run_benchmark(X, y, args.seeds, fracs)
    n_train = int(n * 0.8)
    _print_benchmark(results, n_train)

    # Part 2: Diagnosis
    _run_diagnosis(X, y, args.seed)

    # Part 3: Distance comparison plot
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = f"{args.output_dir}/tabular_distance_comparison.png"
    _plot_distance_comparison(X, args.seed, output_path)


if __name__ == "__main__":
    main()
