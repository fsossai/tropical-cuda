#!/usr/bin/env python3
"""Plot the per-graph speedup of smoothmin_spmv over cugraph (answers Q3).

Same plot as plot_r1.py, with smoothmin_spmv as the compared solver.

    ./plot_r3.py time.yuclid.jsonl -o speedup_r3.svg
"""

import plot_r1

if __name__ == "__main__":
    plot_r1.main(solver="smoothmin_spmv", output="speedup_r3.svg", description=__doc__)
