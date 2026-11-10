"""Time GISTSelector on synthetic clustered data of increasing size.

Prints a Markdown table of wall-clock times for ``fit`` with default settings.
Compilation is excluded: a small problem is fitted first to warm the JIT cache.

Usage:
    uv run experiments/speed_benchmark.py
    uv run experiments/speed_benchmark.py --sizes 1000 5000 --dims 64
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from gist_sampling import GISTSelector


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GISTSelector speed benchmark")
    parser.add_argument("--sizes", nargs="*", type=int, default=[1_000, 5_000, 20_000, 100_000])
    parser.add_argument("--dims", nargs="*", type=int, default=[64, 384])
    parser.add_argument("--sample-frac", type=float, default=0.01, help="n_samples / n")
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--repeats", type=int, default=3, help="Report the fastest of this many")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _clustered(n: int, d: int, seed: int) -> np.ndarray:
    """Float32 points in 20 Gaussian clusters of very unequal size."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3.0, size=(20, d))
    labels = rng.choice(20, size=n, p=rng.dirichlet(np.full(20, 0.5)))
    return (centers[labels] + rng.normal(size=(n, d))).astype(np.float32)


def main() -> None:
    args = _parse_args()

    warmup = _clustered(400, 8, args.seed)
    for n_neighbors in (None, 10):
        GISTSelector(n_samples=5, metric=args.metric, n_neighbors=n_neighbors).fit(warmup)

    print("| points | features | n_samples | similarities kept | selected | seconds |")
    print("| ---: | ---: | ---: | --- | ---: | ---: |")
    for d in args.dims:
        for n in args.sizes:
            X = _clustered(n, d, args.seed)
            k = max(1, int(n * args.sample_frac))
            seconds = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                selector = GISTSelector(n_samples=k, metric=args.metric).fit(X)
                seconds.append(time.perf_counter() - start)
            kept = (
                "all pairs" if selector.n_neighbors_ is None else f"{selector.n_neighbors_} nearest"
            )
            print(
                f"| {n:,} | {d} | {k:,} | {kept} | {len(selector.selected_indices_):,} "
                f"| {min(seconds):.2f} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
