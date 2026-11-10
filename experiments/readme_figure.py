"""Draw the figure at the top of the README: a random sample next to GIST's selection.

Both panels show the same 2-D points and the same number of picks. The circle
around each pick has the same radius in both panels, half the smallest distance
between two of GIST's picks, so GIST's circles never overlap.

    uv run python experiments/readme_figure.py

The figure is written by hand as SVG, so this needs no plotting library. It
uses Musubi's colours and follows the viewer's light or dark colour scheme.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gist_sampling import GISTSelector

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "selection.svg"

N_PICKS = 14
SEED = 0

# (centre, spread, number of points): one flood, two ordinary groups, four rare ones.
GROUPS = [
    ((0.0, -0.4), 1.25, 700),
    ((5.6, 2.6), 0.80, 180),
    ((-5.4, 2.9), 0.70, 90),
    ((5.9, -3.2), 0.45, 22),
    ((-5.8, -3.0), 0.45, 18),
    ((0.6, 4.4), 0.40, 14),
    ((-1.6, -4.6), 0.35, 10),
]

WIDTH, HEIGHT = 872, 412
PANEL_WIDTH, PANEL_HEIGHT = 372, 300
PANEL_TOP = 88
PANEL_LEFT = (34, 466)

# Ink is kept for the data: no frame, axes or grid, and one saturated colour,
# spent on the picks alone.
STYLE = """
  :root { --paper: #f7f7f8; --ink: #4a4a65; --ink-2: #5e5d7e; --point: #ababbf;
          --point-opacity: 0.8; --pick: #f9553f; }
  @media (prefers-color-scheme: dark) {
    :root { --paper: #20202c; --ink: #ececf2; --ink-2: #ababbf; --point: #7e7ea0;
            --point-opacity: 0.7; --pick: #f9553f; }
  }
  text { font-family: "Neue Montreal", Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
  .paper { fill: var(--paper); }
  .title { fill: var(--ink); font-size: 20px; font-weight: 500; letter-spacing: -0.02em; }
  .note { fill: var(--ink-2); font-size: 13px; }
  .missed { fill: var(--ink-2); font-size: 12px; font-style: italic;
            stroke: var(--paper); stroke-width: 3; paint-order: stroke; }
  .point { fill: var(--point); opacity: var(--point-opacity); }
  .reach { fill: var(--pick); fill-opacity: 0.07; stroke: var(--pick); stroke-opacity: 0.55;
           stroke-width: 0.75; }
  .pick { fill: var(--pick); stroke: var(--paper); stroke-width: 2; }
"""


def make_points() -> tuple[np.ndarray, np.ndarray]:
    """The points and the group each belongs to."""
    rng = np.random.default_rng(SEED)
    points = [rng.normal(centre, spread, (count, 2)) for centre, spread, count in GROUPS]
    group = np.repeat(np.arange(len(GROUPS)), [count for *_, count in GROUPS])
    return np.concatenate(points), group


def describe(group: np.ndarray, picks: np.ndarray) -> str:
    """How the picks are spread over the groups, in words."""
    counts = np.bincount(group[picks], minlength=len(GROUPS))
    missed = int((counts == 0).sum())
    largest = f"{counts[0]} of {len(picks)} picks in the largest group"
    if missed == 0:
        return f"{largest}, every group reached"
    return f"{largest}, {missed} of {len(GROUPS)} groups missed"


def main() -> None:
    points, group = make_points()
    selector = GISTSelector(n_samples=N_PICKS, lambda_diversity=0.05 * len(points)).fit(points)
    gist = selector.selected_indices_
    random = np.random.default_rng(SEED).choice(len(points), size=len(gist), replace=False)

    # One scale for both axes, so that circles stay circles.
    low, high = points.min(axis=0), points.max(axis=0)
    scale = min(PANEL_WIDTH / (high[0] - low[0]), PANEL_HEIGHT / (high[1] - low[1]))
    offset = (np.array([PANEL_WIDTH, PANEL_HEIGHT]) - scale * (high - low)) / 2
    pixels = (points - low) * scale + offset
    pixels[:, 1] = PANEL_HEIGHT - pixels[:, 1]
    reach = selector.diversity_ / 2 * scale

    def missed_labels(picks: np.ndarray) -> str:
        """The word "missed" beside every group that has no pick."""
        labels = []
        for index, (_, spread, _) in enumerate(GROUPS):
            if (group[picks] == index).any():
                continue
            x, y = pixels[group == index].mean(axis=0)
            gap = 2.4 * spread * scale + 4
            if x > 0.7 * PANEL_WIDTH:  # no room on the right
                labels.append(
                    f'<text class="missed" x="{x - gap:.1f}" y="{y + 4:.1f}" '
                    f'text-anchor="end">missed</text>'
                )
            else:
                labels.append(
                    f'<text class="missed" x="{x + gap:.1f}" y="{y + 4:.1f}">missed</text>'
                )
        return "".join(labels)

    cloud = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.6"/>' for x, y in pixels)
    panels = []
    for left, title, picks in (
        (PANEL_LEFT[0], "Random sample", random),
        (PANEL_LEFT[1], "GIST", gist),
    ):
        chosen = pixels[picks]
        reaches = "".join(
            f'<circle class="reach" cx="{x:.1f}" cy="{y:.1f}" r="{reach:.1f}"/>' for x, y in chosen
        )
        dots = "".join(
            f'<circle class="pick" cx="{x:.1f}" cy="{y:.1f}" r="4.5"/>' for x, y in chosen
        )
        panels.append(
            f'<text class="title" x="{left}" y="46">{title}</text>'
            f'<text class="note" x="{left}" y="68">{describe(group, picks)}</text>'
            f'<g transform="translate({left} {PANEL_TOP})">'
            f'<use href="#cloud"/>{reaches}{dots}{missed_labels(picks)}</g>'
        )

    summary = (
        f"The same {len(points)} points twice, with {len(gist)} picks each. "
        f"Random sample: {describe(group, random)}. GIST: {describe(group, gist)}."
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'aria-labelledby="title desc">'
        f'<title id="title">A random sample compared with a GIST selection</title>'
        f'<desc id="desc">{summary}</desc>'
        f"<style>{STYLE}</style>"
        f'<rect class="paper" width="{WIDTH}" height="{HEIGHT}" rx="14"/>'
        f'<defs><g id="cloud" class="point">{cloud}</g></defs>'
        f"{''.join(panels)}</svg>\n"
    )
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(svg)
    print(summary)
    print(f"Wrote {OUTPUT} ({len(svg) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
