"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import SCIPY_METRIC


@pytest.fixture
def two_clusters() -> np.ndarray:
    """Forty 2-D points: two well-separated clusters of twenty."""
    rng = np.random.default_rng(42)
    return np.vstack([rng.normal(size=(20, 2)), rng.normal(size=(20, 2)) + 10])


@pytest.fixture(params=sorted(SCIPY_METRIC))
def metric(request) -> str:
    """Each metric computed from features."""
    return request.param
