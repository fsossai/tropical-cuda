#!/usr/bin/env python3
"""Plot the per-graph speedup of minplus_spmmop over cugraph (answers Q2).

Same plot as plot_r1.py, with minplus_spmmop as the compared solver.

    ./plot_r2.py time.yuclid.jsonl -o speedup_r2.svg
"""

import plot_r1

if __name__ == "__main__":
    plot_r1.main(solver="minplus_spmmop", output="speedup_r2.svg", description=__doc__)
