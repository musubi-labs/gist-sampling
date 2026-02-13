"""Distance computation utilities."""

from gist_sampling.distance.lazy import LazyDistanceMatrix, compute_lazy_distance_matrix
from gist_sampling.distance.metrics import (
    SUPPORTED_METRICS,
    MetricType,
    compute_distance_matrix,
    compute_similarity_matrix,
)

__all__ = [
    "SUPPORTED_METRICS",
    "MetricType",
    "compute_distance_matrix",
    "compute_similarity_matrix",
    "LazyDistanceMatrix",
    "compute_lazy_distance_matrix",
]
