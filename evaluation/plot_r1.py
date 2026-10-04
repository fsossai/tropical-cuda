#!/usr/bin/env python3
"""Plot the per-graph speedup of minplus_spmv over cugraph (answers Q1).

Each (graph, solver) cell is reduced to the median of its repetitions, then
turned into a ratio against the cugraph median for the same graph:

    speedup = median(kernel, cugraph) / median(kernel, solver)

so taller is faster and the cugraph bars sit at 1.00 by construction.
plot_r2.py and plot_r3.py reuse this script for the other solvers.

    ./plot_r1.py time.yuclid.jsonl -o speedup_r1.svg
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

SOLVER = "minplus_spmv"       # series compared against the baseline in this plot

# Categorical slots 1 to 3 of the validated palette, one per solver so each keeps its color
# across plots; the baseline is deliberately neutral, since it is the reference, not a result.
SOLVER_COLORS = {"minplus_spmv": "#2a78d6", "minplus_spmmop": "#eb6834", "smoothmin_spmv": "#1baf7a"}
BASELINE_COLOR = "#b4b3ac"

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


def medians(records, solver):
    """Median metric value per (x, z) cell, for the solver and the baseline only."""
    samples = defaultdict(list)
    for record in records:
        if record.get(Z_DIM) not in (solver, BASELINE):
            continue
        try:
            samples[record[X_DIM], record[Z_DIM]].append(float(record[METRIC]))
        except KeyError as exc:
            sys.exit(f"missing column {exc.args[0]!r}; expected "
                     f"{X_DIM!r}, {Z_DIM!r}, {METRIC!r}")
    return {cell: statistics.median(values) for cell, values in samples.items()}


def speedups(records, solver):
    """(x values, {z: [speedup per x]}) with the geomean appended to each series."""
    median = medians(records, solver)
    xs = sorted({x for x, _ in median})
    zs = [solver, BASELINE]
    for z in zs:
        if not any(z == cell[1] for cell in median):
            sys.exit(f"{z!r} not present in {Z_DIM!r}")

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


def plot(xs, series, out, digits, parity_label=None):
    """Draw grouped bars per x value; parity_label names the 1x line in the legend."""
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
            color=BASELINE_COLOR if z == BASELINE else SOLVER_COLORS.get(z, INK_MUTED),
            label=z, zorder=3,
        )
        for position, bar, value in zip(positions, bars, values):
            if math.isnan(value):
                continue
            ax.annotate(
                f"{value:.{digits}f}",
                (bar.get_x() + bar.get_width() / 2, value),
                textcoords="offset points", xytext=(0, 3),
                ha="center", va="bottom",
                fontsize=ANNOTATION_SIZE, color=INK_SOFT, rotation=90, zorder=4,
                fontweight="bold" if xs[position] == GEOMEAN_LABEL else "normal",
            )

    # Parity with the baseline, and the divider in front of the summary group.
    ax.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2,
               label=parity_label)
    ax.axvline(len(xs) - 1.5, color=INK_MUTED, linewidth=0.8, alpha=0.6, zorder=2)

    top = max(v for values in series.values() for v in values if not math.isnan(v))
    ax.set_ylim(0, top * 1.38)
    ax.set_xlim(-0.6, len(xs) - 0.4)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(xs, rotation=30, ha="right")
    for label in ax.get_xticklabels():
        if label.get_text() == GEOMEAN_LABEL:
            label.set_fontweight("bold")
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
        frameon=False, ncol=len(series) + (parity_label is not None), loc="upper left",
        bbox_to_anchor=(0, -0.5), fontsize=LEGEND_SIZE, labelcolor=INK_SOFT,
    )

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE, dpi=200)
    print(f"wrote {out}")


def main(solver=SOLVER, output="speedup_r1.svg", description=__doc__):
    """Parse the command line and plot the speedup of one solver over the baseline."""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", help="yuclid JSON Lines result file")
    parser.add_argument("-o", "--output", default=output,
                        help="output path; extension picks the format (default: %(default)s)")
    parser.add_argument("-d", "--digits", type=int, default=2,
                        help="digits in the bar annotations (default: %(default)s)")
    parser.add_argument("--show", action="store_true", help="also open a window")
    args = parser.parse_args()

    xs, series = speedups(load(args.file), solver)
    plot(xs, series, args.output, args.digits)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
