"""Diversity-aware subset selection with GIST (Greedy Independent Set Thresholding)."""

from gist_sampling._selector import GISTSelector, gist_sample
from gist_sampling._version import __version__

__all__ = ["GISTSelector", "__version__", "gist_sample"]
