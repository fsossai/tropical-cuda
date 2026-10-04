#!/usr/bin/env python3
"""Plot the per-graph speedup of every solver over cugraph in one figure.

Same computation as plot_r1.py, with one bar per solver and the cugraph
baseline drawn as the dashed line at 1x instead of as bars.

    ./plot_all.py time.yuclid.jsonl -o speedup_all.svg
"""

import argparse

import matplotlib.pyplot as plt

import plot_r1

SOLVERS = ["minplus_spmv", "minplus_spmmop", "smoothmin_spmv"]


def main():
    """Parse the command line and plot every solver against the baseline."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", help="yuclid JSON Lines result file")
    parser.add_argument("-o", "--output", default="speedup_all.svg",
                        help="output path; extension picks the format (default: %(default)s)")
    parser.add_argument("-d", "--digits", type=int, default=2,
                        help="digits in the bar annotations (default: %(default)s)")
    parser.add_argument("--show", action="store_true", help="also open a window")
    args = parser.parse_args()

    records = plot_r1.load(args.file)
    xs, series = None, {}
    for solver in SOLVERS:
        xs, solver_series = plot_r1.speedups(records, solver)
        series[solver] = solver_series[solver]

    plot_r1.plot(xs, series, args.output, args.digits, parity_label=plot_r1.BASELINE)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
