"""The examples in the README run as written."""

from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).parent.parent / "README.md"


def test_readme_examples_run():
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.DOTALL)
    assert len(blocks) >= 5

    namespace: dict = {}
    for block in blocks:
        exec(block, namespace)  # blocks build on one another, as a reader would run them

    assert len(namespace["X_small"]) == 100
    assert len(namespace["sample"]) == 50
    assert list(namespace["sample"].columns) == ["a", "b", "c", "note"]
    assert len(namespace["texts_to_label"]) == 100
    assert len(namespace["selected"]) == 90
    assert len(namespace["selector"].selected_indices_) == 20
