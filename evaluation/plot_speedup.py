#!/usr/bin/env python3
"""Plot per-graph solver speedup over cugraph from a yuclid JSON Lines result file.

Each (graph, solver) cell is reduced to the median of its repetitions, then
turned into a ratio against the cugraph median for the same graph:

    speedup = median(kernel, cugraph) / median(kernel, solver)

so taller is faster and the cugraph series sits flat at 1.00 by construction.

    ./plot_speedup.py 20261002-233322.yuclid.jsonl -o speedup.svg
"""

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- configuration ----------------------------------------------------------

BASELINE = "cugraph"          # series everything is normalized against
X_DIM, Z_DIM, METRIC = "graph", "solver", "kernel"

# Bar order, left to right within each graph; the baseline goes last.
SOLVER_ORDER = ["tropical_exact", "cusparse", "cugraph"]

# Categorical slots 1 and 2 of the validated palette for the compared solvers;
# the baseline is deliberately neutral, since it is the reference, not a result.
COLORS = {
    "tropical_exact": "#2a78d6",
    "cusparse": "#eb6834",
    "cugraph": "#b4b3ac",
}

SURFACE = "#fcfcfb"
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#8c8b85"

# Type sizes, in points.
TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, ANNOTATION_SIZE = 16, 14, 13, 13, 16

GEOMEAN_LABEL = "geomean"


# --- data -------------------------------------------------------------------


def load(path):
    """Read a yuclid JSON Lines file into a list of records, skipping blanks."""
    records = []
    with open(path) as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                sys.exit(f"{path}:{lineno}: malformed JSON ({exc.msg})")
    if not records:
        sys.exit(f"{path}: no records")
    return records


def medians(records):
    """Median metric value per (x, z) cell."""
    samples = defaultdict(list)
    for record in records:
        try:
            samples[record[X_DIM], record[Z_DIM]].append(float(record[METRIC]))
        except KeyError as exc:
            sys.exit(f"missing column {exc.args[0]!r}; expected "
                     f"{X_DIM!r}, {Z_DIM!r}, {METRIC!r}")
    return {cell: statistics.median(values) for cell, values in samples.items()}


def speedups(records):
    """(x values, {z: [speedup per x]}) with the geomean appended to each series."""
    median = medians(records)
    xs = sorted({x for x, _ in median})
    zs = [z for z in SOLVER_ORDER if any(z == cell[1] for cell in median)]
    zs += sorted({z for _, z in median} - set(zs))
    if BASELINE not in zs:
        sys.exit(f"baseline {BASELINE!r} not present in {Z_DIM!r}")

    series = {}
    for z in zs:
        ratios = []
        for x in xs:
            reference, value = median.get((x, BASELINE)), median.get((x, z))
            ratios.append(reference / value if reference and value else math.nan)
        finite = [r for r in ratios if not math.isnan(r)]
        geomean = math.exp(sum(map(math.log, finite)) / len(finite)) if finite else math.nan
        series[z] = ratios + [geomean]
    return xs + [GEOMEAN_LABEL], series


# --- plot -------------------------------------------------------------------


def plot(xs, series, out, digits):
    # The baseline is flat at 1.00 by construction; the parity line carries it
    # instead of a row of identical bars.
    series = {z: values for z, values in series.items() if z != BASELINE}
    group_width, n = 0.8, len(series)
    width = group_width / n
    positions = range(len(xs))

    fig, ax = plt.subplots(figsize=(1.0 * len(xs) + 2.0, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for index, (z, values) in enumerate(series.items()):
        offsets = [p - group_width / 2 + width * (index + 0.5) for p in positions]
        bars = ax.bar(
            offsets, values,
            width=width * 0.88,            # leaves a surface gap between bars
            color=COLORS.get(z, INK_MUTED),
            label=z, zorder=3,
        )
        for bar, value in zip(bars, values):
            if math.isnan(value):
                continue
            ax.annotate(
                f"{value:.{digits}f}",
                (bar.get_x() + bar.get_width() / 2, value),
                textcoords="offset points", xytext=(0, 3),
                ha="center", va="bottom",
                fontsize=ANNOTATION_SIZE, color=INK_SOFT, rotation=90, zorder=4,
            )

    # Parity with the baseline, and the divider in front of the summary group.
    ax.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2,
               label=BASELINE)
    ax.axvline(len(xs) - 1.5, color=INK_MUTED, linewidth=0.8, alpha=0.6, zorder=2)

    top = max(v for values in series.values() for v in values if not math.isnan(v))
    ax.set_ylim(0, top * 1.38)
    ax.set_xlim(-0.6, len(xs) - 0.4)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(xs, rotation=30, ha="right")
    ax.set_ylabel(f"Speedup over {BASELINE}", color=INK, fontsize=LABEL_SIZE)
    # Title inside the axes, in the headroom left above the tallest bar.
    ax.text(
        0.5, 0.97, f"Speedup over {BASELINE}",
        transform=ax.transAxes, ha="center", va="top",
        color=INK, fontsize=TITLE_SIZE, zorder=5,
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))

    ax.grid(axis="y", color=INK_MUTED, alpha=0.25, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SOFT, labelsize=TICK_SIZE)

    ax.legend(
        frameon=False, ncol=len(series) + 1, loc="upper left",
        bbox_to_anchor=(0, -0.34), fontsize=LEGEND_SIZE, labelcolor=INK_SOFT,
    )

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE, dpi=200)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", help="yuclid JSON Lines result file")
    parser.add_argument("-o", "--output", default="speedup.svg",
                        help="output path; extension picks the format (default: %(default)s)")
    parser.add_argument("-d", "--digits", type=int, default=2,
                        help="digits in the bar annotations (default: %(default)s)")
    parser.add_argument("--show", action="store_true", help="also open a window")
    args = parser.parse_args()

    xs, series = speedups(load(args.file))
    plot(xs, series, args.output, args.digits)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
