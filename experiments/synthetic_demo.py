"""Synthetic social-media dataset: random vs GIST sampling demonstration.

Creates a synthetic dataset of 500 "social media posts" across 5 topic clusters
using sklearn.make_blobs, then shows how GIST covers all clusters while random
over-samples the dominant one.

No model downloads or API keys required -- runs with numpy, sklearn, matplotlib.

Usage:
    uv run experiments/synthetic_demo.py
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from gist_sampling import GISTSelector

# ---------------------------------------------------------------------------
# Cluster definitions
# ---------------------------------------------------------------------------

CLUSTERS = [
    {
        "name": "GoFundMe bot spam",
        "size": 300,
        "std": 0.5,
        "examples": [
            "Please help my friend Sarah reach her goal! Every dollar counts 🙏 gofundme.com/help-sarah-fight",
            "We're so close to our goal!! Please donate and share 🙏🙏 gofundme.com/save-our-school",
            "My cousin was in a terrible accident. Any amount helps. Please share! gofundme.com/help-mike-recover",
        ],
    },
    {
        "name": "Star Wars discussion",
        "size": 50,
        "std": 1.5,
        "examples": [
            "Honestly the prequels aged really well. The world-building is incredible compared to the sequels",
            "Hot take: Andor is the best Star Wars content since Empire Strikes Back",
            "Just rewatched the original trilogy with my kids. They loved it as much as I did 30 years ago",
        ],
    },
    {
        "name": "NBC Olympics coverage",
        "size": 50,
        "std": 1.5,
        "examples": [
            "Did you see the 100m final last night?? That finish was unbelievable",
            "NBC's commentary during the gymnastics was actually really good this year",
            "The opening ceremony was beautiful. Paris really went all out",
        ],
    },
    {
        "name": "Prize money scams",
        "size": 50,
        "std": 0.6,
        "examples": [
            "CONGRATULATIONS!! You've been selected to receive $5,000! Click here to claim your prize NOW",
            "🎉 WINNER ALERT 🎉 Your phone number has won our weekly $10,000 cash giveaway! Reply YES to claim",
            "URGENT: You have an unclaimed prize of $2,500. Call 1-800-555-0199 before it expires!",
        ],
    },
    {
        "name": "Movie recommendations",
        "size": 50,
        "std": 1.5,
        "examples": [
            "Just saw the new Marvel movie and honestly it's the best one in years. Go see it",
            "Can someone recommend a good thriller? I've watched everything on Netflix already",
            "Dune Part Two is a masterpiece. Denis Villeneuve doesn't miss",
        ],
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic dataset: random vs GIST sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        default="experiments/assets",
        help="Directory for output images",
    )
    return parser.parse_args()


def _generate_dataset(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic embeddings with make_blobs, then L2-normalize."""
    sizes = [c["size"] for c in CLUSTERS]
    stds = [c["std"] for c in CLUSTERS]
    n_features = 384  # match sentence-transformer embedding dimension

    X, y = make_blobs(
        n_samples=sizes,
        n_features=n_features,
        cluster_std=stds,
        center_box=(-10, 10),
        random_state=seed,
    )

    # L2-normalize to match real embedding conventions (unit sphere)
    X = normalize(X, norm="l2")

    return X, y


def _coverage_scores(indices: np.ndarray, sim_matrix: np.ndarray) -> np.ndarray:
    """For each item in the full dataset, max similarity to nearest selected item."""
    return sim_matrix[:, indices].max(axis=1)


def _cluster_distribution(indices: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    """Count how many selected items come from each cluster."""
    counts = {}
    for cluster in range(len(CLUSTERS)):
        counts[CLUSTERS[cluster]["name"]] = int(np.sum(labels[indices] == cluster))
    return counts


def _print_run(
    title: str,
    indices: np.ndarray,
    labels: np.ndarray,
    coverage: np.ndarray,
    n_total: int,
) -> None:
    """Print results for one sampling method."""
    dist = _cluster_distribution(indices, labels)
    clusters_hit = sum(1 for v in dist.values() if v > 0)
    clusters_missed = [name for name, v in dist.items() if v == 0]

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  Mean coverage: {coverage.mean():.3f}")
    print(f"  Clusters covered: {clusters_hit}/{len(CLUSTERS)}")
    print()
    print(f"  {'Cluster':<30} {'Picks':>6}")
    print(f"  {'-' * 30} {'-' * 6}")
    for name, count in dist.items():
        marker = "" if count > 0 else "  <-- MISSED"
        print(f"  {name:<30} {count:>6}{marker}")

    if clusters_missed:
        print(f"\n  Missed clusters -- example posts the sample would never see:")
        for name in clusters_missed:
            cluster_idx = [i for i, c in enumerate(CLUSTERS) if c["name"] == name][0]
            example = CLUSTERS[cluster_idx]["examples"][0]
            print(f'    [{name}] "{example}"')
    print()


def _plot_coverage_histogram(
    random_cov: np.ndarray,
    gist_cov: np.ndarray,
    k: int,
    n_total: int,
    output_path: str,
) -> None:
    """Overlapping histogram of per-item coverage scores."""
    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(0.0, 1.0, 40)
    ax.hist(
        random_cov,
        bins=bins,
        alpha=0.55,
        color="#e74c3c",
        label=f"Random (k={k})",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.hist(
        gist_cov,
        bins=bins,
        alpha=0.55,
        color="#2ecc71",
        label=f"GIST (k={k})",
        edgecolor="white",
        linewidth=0.5,
    )

    ax.axvline(
        random_cov.mean(),
        color="#c0392b",
        linestyle="--",
        linewidth=1.5,
        label=f"Random mean ({random_cov.mean():.3f})",
    )
    ax.axvline(
        gist_cov.mean(),
        color="#27ae60",
        linestyle="--",
        linewidth=1.5,
        label=f"GIST mean ({gist_cov.mean():.3f})",
    )

    ax.set_xlabel("Coverage (max cosine similarity to nearest selected item)", fontsize=11)
    ax.set_ylabel("Number of items in full dataset", fontsize=11)
    ax.set_title(
        f"How well does each {k}-item sample represent {n_total} posts?",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Histogram saved to {output_path}")


def _plot_cluster_picks(
    random_dist: dict[str, int],
    gist_dist: dict[str, int],
    k: int,
    output_path: str,
) -> None:
    """Side-by-side bar chart of cluster picks per method."""
    names = list(random_dist.keys())
    # Shorten names for display
    short = [
        n.replace(" discussion", "").replace(" coverage", "").replace(" recommendations", "")
        for n in names
    ]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(
        x - width / 2,
        [random_dist[n] for n in names],
        width,
        label="Random",
        color="#e74c3c",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        [gist_dist[n] for n in names],
        width,
        label="GIST",
        color="#2ecc71",
        alpha=0.8,
    )

    ax.set_xlabel("Topic Cluster", fontsize=11)
    ax.set_ylabel(f"Picks (of k={k})", fontsize=11)
    ax.set_title(f"Where does each method spend its {k} picks?", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=15, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(random_dist.values()), max(gist_dist.values())) + 1)

    # Add count labels on bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.1, str(int(h)), ha="center", fontsize=10
            )
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.1, str(int(h)), ha="center", fontsize=10
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Cluster picks chart saved to {output_path}")


def _run_comparison(
    k: int,
    X: np.ndarray,
    y: np.ndarray,
    sim_matrix: np.ndarray,
    seed: int,
    output_dir: str,
) -> None:
    """Run one k-value comparison."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    print(f"\n{'#' * 70}")
    print(f"  k = {k}")
    print(f"{'#' * 70}")

    # Random sample
    random_idx = rng.choice(n, size=k, replace=False)
    random_cov = _coverage_scores(random_idx, sim_matrix)

    # GIST sample
    selector = GISTSelector(
        n_samples=k,
        metric="cosine",
        epsilon=0.05,
        lambda_diversity=1.0,
        random_state=seed,
        mode="exact",
    )
    selector.fit(X)
    gist_idx = selector.selected_indices_
    gist_cov = _coverage_scores(gist_idx, sim_matrix)

    # Print results
    _print_run(f"RANDOM SAMPLE (k={k})", random_idx, y, random_cov, n)
    _print_run(f"GIST SAMPLE (k={k})", gist_idx, y, gist_cov, n)

    # Summary
    random_dist = _cluster_distribution(random_idx, y)
    gist_dist = _cluster_distribution(gist_idx, y)

    print(f"  {'Metric':<40} {'Random':>8} {'GIST':>8}")
    print(f"  {'-' * 40} {'-' * 8} {'-' * 8}")
    print(f"  {'Mean coverage':<40} {random_cov.mean():>8.3f} {gist_cov.mean():>8.3f}")
    print(
        f"  {'Clusters covered':<40} {sum(1 for v in random_dist.values() if v > 0):>8} {sum(1 for v in gist_dist.values() if v > 0):>8}"
    )
    r_blind = (random_cov < 0.5).sum()
    g_blind = (gist_cov < 0.5).sum()
    print(f"  {'Blind spots (coverage < 0.5)':<40} {r_blind:>8} {g_blind:>8}")

    # Multi-seed stability
    print(f"\n  Clusters covered across seeds:")
    for s in [42, 43, 44, 7, 13]:
        r = np.random.default_rng(s)
        ri = r.choice(n, size=k, replace=False)
        sel = GISTSelector(
            n_samples=k,
            metric="cosine",
            epsilon=0.05,
            lambda_diversity=1.0,
            random_state=s,
            mode="exact",
        )
        sel.fit(X)
        gi = sel.selected_indices_
        rd = _cluster_distribution(ri, y)
        gd = _cluster_distribution(gi, y)
        rc = sum(1 for v in rd.values() if v > 0)
        gc = sum(1 for v in gd.values() if v > 0)
        print(f"    seed={s}: random={rc}/5, GIST={gc}/5")

    # Plots
    os.makedirs(output_dir, exist_ok=True)
    _plot_coverage_histogram(
        random_cov, gist_cov, k, n, f"{output_dir}/synthetic_coverage_k{k}.png"
    )
    _plot_cluster_picks(random_dist, gist_dist, k, f"{output_dir}/synthetic_cluster_picks_k{k}.png")


def main() -> None:
    args = _parse_args()
    seed = args.seed

    # Generate dataset
    print("Generating synthetic dataset...")
    X, y = _generate_dataset(seed)
    n = X.shape[0]

    print(f"\nDataset: {n} synthetic social media posts")
    print(f"{'Cluster':<30} {'Size':>6} {'%':>6}")
    print(f"{'-' * 30} {'-' * 6} {'-' * 6}")
    for i, cluster in enumerate(CLUSTERS):
        size = (y == i).sum()
        print(f"{cluster['name']:<30} {size:>6} {100 * size / n:>5.1f}%")
    print()

    print("Example posts per cluster:")
    for cluster in CLUSTERS:
        print(f"  [{cluster['name']}]")
        for ex in cluster["examples"][:2]:
            print(f'    "{ex}"')
    print()

    # Cosine similarity matrix
    sim_matrix = cosine_similarity(X)

    # Run comparisons for both k values
    _run_comparison(k=10, X=X, y=y, sim_matrix=sim_matrix, seed=seed, output_dir=args.output_dir)
    _run_comparison(k=5, X=X, y=y, sim_matrix=sim_matrix, seed=seed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
