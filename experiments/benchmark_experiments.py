"""Benchmark Experiments for GIST"""

from __future__ import annotations

import argparse
import hashlib
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from gist_sampling import GISTSelector

_WARNING_FILTERS = (
    "ignore::UserWarning:sklearn.utils.parallel",
    "ignore::RuntimeWarning:sklearn.utils.extmath",
)
os.environ.setdefault("PYTHONWARNINGS", ",".join(_WARNING_FILTERS))
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"sklearn\.utils\.extmath",
)


@dataclass
class DatasetSpec:
    name: str
    subset: str | None
    text_col: str
    label_col: str
    label_map: dict | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Experiments for GIST")
    parser.add_argument("--sample-fracs", nargs="*", type=float, default=[0.02, 0.05, 0.1])
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--gist-metric", default="cosine")
    parser.add_argument("--gist-epsilon", type=float, default=0.05)
    parser.add_argument(
        "--gist-lambda",
        type=float,
        default=1.0,
        help="Diversity weight (λ) for GIST objective f(S)=g(S)+λ·div(S)",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=128,
        help="Batch size for SentenceTransformer.encode",
    )
    parser.add_argument(
        "--embedding-device",
        default=None,
        help="Device for embeddings (e.g., cpu, cuda, mps). Default: auto",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        default=None,
        help="If set, cache embeddings in this directory",
    )
    parser.add_argument(
        "--gist-mode",
        choices=["auto", "exact", "approximate"],
        default="auto",
        help="GIST computation mode (auto/exact/approximate)",
    )
    parser.add_argument(
        "--approximate-threshold",
        type=int,
        default=2000,
        help="Row threshold for auto mode to switch to approximate",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=200,
        help="k-NN size for sparse similarity in approximate mode",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Max threads for algorithm parallelism",
    )
    parser.add_argument(
        "--normalize-embeddings",
        dest="normalize_embeddings",
        action="store_true",
        default=True,
        help="L2-normalize embeddings (recommended for cosine distance)",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        dest="normalize_embeddings",
        action="store_false",
        help="Disable L2-normalization of embeddings",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--seed-jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Parallel workers across seeds",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional dataset filters (e.g., tweet_eval/hate sms_spam)",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model",
    )
    return parser.parse_args()


def _select_k(n_rows: int, sample_frac: float) -> int:
    return max(1, min(int(n_rows * sample_frac), n_rows))


def _random_sample_indices(n_rows: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(n_rows, size=k, replace=False)


def _stratified_sample_indices(
    y: np.ndarray, k: int, random_state: int, rng: np.random.Generator
) -> np.ndarray:
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=k, random_state=random_state)
    indices = np.arange(len(y))
    try:
        train_idx, _ = next(splitter.split(indices, y))
        return train_idx
    except ValueError:
        return _random_sample_indices(len(y), k, rng)


def _fit_auc(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    if np.unique(y).size < 2 or np.unique(y_test).size < 2:
        return float("nan")
    model = LogisticRegression(max_iter=2000)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            model.fit(X, y)
            probs = model.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, probs))


def _load_dataset(
    spec: DatasetSpec, max_rows: int, random_state: int
) -> tuple[list[str], np.ndarray]:
    split = "train"
    if max_rows:
        split = f"train[:{max_rows}]"
    try:
        data = load_dataset(spec.name, spec.subset, split=split)
        data = data.shuffle(seed=random_state)
        texts = data[spec.text_col]
        labels = data[spec.label_col]
    except Exception:
        ds = load_dataset(spec.name, spec.subset) if spec.subset else load_dataset(spec.name)
        if "train" in ds:
            data = ds["train"]
        else:
            data = ds[list(ds.keys())[0]]
        data = data.shuffle(seed=random_state)
        if max_rows and len(data) > max_rows:
            data = data.select(range(max_rows))
        texts = data[spec.text_col]
        labels = data[spec.label_col]

    if spec.label_map is not None:
        labels = [spec.label_map.get(label, label) for label in labels]
    else:
        unique = set(labels)
        if len(unique) > 2 or not unique.issubset({0, 1}):
            labels = [1 if float(label) >= 0.5 else 0 for label in labels]

    y = np.asarray(labels, dtype=int)
    return texts, y


def _embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    normalize: bool,
    batch_size: int,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize,
    )
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if not np.isfinite(embeddings).all():
        embeddings = np.nan_to_num(
            embeddings,
            copy=False,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    return embeddings


def _embeddings_are_sane(embeddings: np.ndarray, normalize: bool) -> bool:
    if not np.isfinite(embeddings).all():
        return False
    if normalize:
        max_abs = float(np.max(np.abs(embeddings)))
        if max_abs > 10.0:
            return False
    return True


def _embedding_cache_path(
    cache_dir: str | None,
    spec: DatasetSpec,
    max_rows: int,
    model_name: str,
    normalize: bool,
) -> str | None:
    if not cache_dir:
        return None
    subset = spec.subset or "all"
    rows = str(max_rows) if max_rows else "all"
    key = f"{spec.name}:{subset}:{spec.text_col}:{rows}:{model_name}:{normalize}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    filename = f"{spec.name}_{subset}_n{rows}_{digest}.npy".replace("/", "-")
    return os.path.join(cache_dir, filename)


def _stratified_gist_indices(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    metric: str,
    epsilon: float,
    lambda_diversity: float,
    random_state: int,
    n_jobs: int,
    mode: str,
    approximate_threshold: int,
    k_neighbors: int,
) -> np.ndarray:
    indices = np.arange(len(y))
    classes, counts = np.unique(y, return_counts=True)
    if classes.size == 1:
        return indices[:k]
    target = {}
    total = counts.sum()
    for cls, count in zip(classes, counts, strict=False):
        target[cls] = max(1, int(round(k * (count / total))))
    current = sum(target.values())
    while current > k:
        largest = max(target, key=lambda cls: target[cls])
        if target[largest] > 1:
            target[largest] -= 1
            current -= 1
        else:
            break
    while current < k:
        largest = max(target, key=lambda cls: target[cls])
        target[largest] += 1
        current += 1

    selected = []
    for cls in classes:
        cls_idx = indices[y == cls]
        if target[cls] >= len(cls_idx):
            selected.extend(cls_idx.tolist())
            continue
        per_class_seed = random_state + int(cls)
        selector = GISTSelector(
            n_samples=target[cls],
            metric=metric,
            epsilon=epsilon,
            lambda_diversity=lambda_diversity,
            random_state=per_class_seed,
            n_jobs=n_jobs,
            mode=mode,
            approximate_threshold=approximate_threshold,
            k_neighbors=k_neighbors,
        )
        selector.fit(X[cls_idx])
        picked = cls_idx[selector.selected_indices_]
        selected.extend(picked.tolist())
    return np.asarray(selected, dtype=int)


def _summarize(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    return float(valid.mean()), float(valid.std(ddof=0)), int(valid.size)


def _evaluate_seed(
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    sample_fracs: list[float],
    gist_metric: str,
    gist_epsilon: float,
    gist_lambda: float,
    gist_mode: str,
    approximate_threshold: int,
    k_neighbors: int,
    n_jobs: int,
) -> tuple[float, dict[float, dict[str, float]], dict[float, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    full_auc = _fit_auc(X_train, y_train, X_test, y_test)
    results: dict[float, dict[str, float]] = {}
    balance: dict[float, dict[str, float]] = {}

    for frac in sample_fracs:
        k = _select_k(len(X_train), frac)

        gist_selector = GISTSelector(
            n_samples=k,
            metric=gist_metric,
            epsilon=gist_epsilon,
            lambda_diversity=gist_lambda,
            random_state=seed,
            n_jobs=n_jobs,
            mode=gist_mode,
            approximate_threshold=approximate_threshold,
            k_neighbors=k_neighbors,
        )
        gist_selector.fit(X_train)
        gist_idx = gist_selector.selected_indices_

        rand_idx = _random_sample_indices(len(X_train), k, rng)
        strat_idx = _stratified_sample_indices(y_train, k, seed, rng)
        strat_gist_idx = _stratified_gist_indices(
            X_train,
            y_train,
            k,
            gist_metric,
            gist_epsilon,
            gist_lambda,
            seed,
            n_jobs,
            gist_mode,
            approximate_threshold,
            k_neighbors,
        )

        results[frac] = {
            "gist": _fit_auc(X_train[gist_idx], y_train[gist_idx], X_test, y_test),
            "random": _fit_auc(X_train[rand_idx], y_train[rand_idx], X_test, y_test),
            "strat": _fit_auc(X_train[strat_idx], y_train[strat_idx], X_test, y_test),
            "strat_gist": _fit_auc(
                X_train[strat_gist_idx], y_train[strat_gist_idx], X_test, y_test
            ),
        }
        balance[frac] = {
            "gist": float(np.mean(y_train[gist_idx])),
            "random": float(np.mean(y_train[rand_idx])),
            "strat": float(np.mean(y_train[strat_idx])),
            "strat_gist": float(np.mean(y_train[strat_gist_idx])),
        }

    return full_auc, results, balance


def _run_dataset(
    spec: DatasetSpec,
    model: SentenceTransformer,
    args: argparse.Namespace,
) -> None:
    texts, y = _load_dataset(spec, args.max_rows, args.random_state)

    print(f"\nDataset: {spec.name}{'/' + spec.subset if spec.subset else ''}")
    print(f"Loaded rows: {len(texts)}")
    cache_path = _embedding_cache_path(
        args.embedding_cache_dir,
        spec,
        args.max_rows,
        args.model,
        args.normalize_embeddings,
    )
    if cache_path and os.path.exists(cache_path):
        X = np.load(cache_path)
        if _embeddings_are_sane(X, args.normalize_embeddings):
            print(f"Loaded embeddings from {cache_path}")
        else:
            print("Cached embeddings contained non-finite or extreme values; recomputing.")
            X = _embed_texts(
                model,
                texts,
                args.normalize_embeddings,
                args.embedding_batch_size,
            )
            if cache_path:
                os.makedirs(args.embedding_cache_dir, exist_ok=True)
                np.save(cache_path, X)
                print(f"Saved embeddings to {cache_path}")
    else:
        X = _embed_texts(
            model,
            texts,
            args.normalize_embeddings,
            args.embedding_batch_size,
        )
        if cache_path:
            os.makedirs(args.embedding_cache_dir, exist_ok=True)
            np.save(cache_path, X)
            print(f"Saved embeddings to {cache_path}")
    print("Embeddings computed.")

    full_aucs: list[float] = []
    results: dict[float, dict[str, list[float]]] = {}
    balance: dict[float, dict[str, list[float]]] = {}

    for frac in args.sample_fracs:
        results[frac] = {"gist": [], "random": [], "strat": [], "strat_gist": []}
        balance[frac] = {"gist": [], "random": [], "strat": [], "strat_gist": []}

    algo_jobs = args.n_jobs
    seed_jobs = max(1, min(args.seed_jobs, len(args.seeds)))
    if seed_jobs > 1 and algo_jobs == -1:
        cpu_count = os.cpu_count() or 1
        algo_jobs = max(1, cpu_count // seed_jobs)
    if seed_jobs > 1:
        with ProcessPoolExecutor(max_workers=seed_jobs) as executor:
            futures = [
                executor.submit(
                    _evaluate_seed,
                    seed,
                    X,
                    y,
                    list(args.sample_fracs),
                    args.gist_metric,
                    args.gist_epsilon,
                    args.gist_lambda,
                    args.gist_mode,
                    args.approximate_threshold,
                    args.k_neighbors,
                    algo_jobs,
                )
                for seed in args.seeds
            ]
            for future in as_completed(futures):
                full_auc, seed_results, seed_balance = future.result()
                full_aucs.append(full_auc)
                for frac in args.sample_fracs:
                    for method in results[frac].keys():
                        results[frac][method].append(seed_results[frac][method])
                        balance[frac][method].append(seed_balance[frac][method])
    else:
        for seed in args.seeds:
            full_auc, seed_results, seed_balance = _evaluate_seed(
                seed,
                X,
                y,
                list(args.sample_fracs),
                args.gist_metric,
                args.gist_epsilon,
                args.gist_lambda,
                args.gist_mode,
                args.approximate_threshold,
                args.k_neighbors,
                algo_jobs,
            )
            full_aucs.append(full_auc)
            for frac in args.sample_fracs:
                for method in results[frac].keys():
                    results[frac][method].append(seed_results[frac][method])
                    balance[frac][method].append(seed_balance[frac][method])

    full_mean, full_std, full_n = _summarize(full_aucs)
    print(f"Rows: {len(X)} | Full AUC: {full_mean:.4f} ± {full_std:.4f} (n={full_n})")

    for frac in args.sample_fracs:
        k = _select_k(int(len(X) * 0.8), frac)
        line = [f"  frac={frac:.2f} k={k}"]
        for method in ["gist", "random", "strat", "strat_gist"]:
            mean, std, n = _summarize(results[frac][method])
            bal_mean, bal_std, _ = _summarize(balance[frac][method])
            if np.isfinite(mean):
                line.append(f"{method}={mean:.4f}±{std:.4f} n={n} pos={bal_mean:.3f}±{bal_std:.3f}")
            else:
                line.append(f"{method}=n/a")
        print(" | ".join(line))


def main() -> None:
    args = _parse_args()
    if args.embedding_device:
        model = SentenceTransformer(args.model, device=args.embedding_device)
    else:
        model = SentenceTransformer(args.model)

    specs = [
        DatasetSpec(
            name="tweet_eval",
            subset="hate",
            text_col="text",
            label_col="label",
            label_map=None,
        ),
        DatasetSpec(
            name="tweet_eval",
            subset="offensive",
            text_col="text",
            label_col="label",
            label_map=None,
        ),
        DatasetSpec(
            name="sms_spam",
            subset=None,
            text_col="sms",
            label_col="label",
            label_map={"ham": 0, "spam": 1},
        ),
    ]

    filters = set(args.only or [])
    for spec in specs:
        key = f"{spec.name}/{spec.subset}" if spec.subset else spec.name
        if filters and key not in filters:
            continue
        _run_dataset(spec, model, args)


if __name__ == "__main__":
    main()
