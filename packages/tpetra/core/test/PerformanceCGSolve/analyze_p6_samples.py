#!/usr/bin/env python3
"""Summarize P6 rank-major CSV from one or more GPU runs.

No external Python dependencies. Each sample is the maximum elapsed wall time
across all MPI ranks for that repetition. Per-phase maxima are NOT additive.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def percentile(sorted_values, q):
    index = q * (len(sorted_values) - 1)
    lo = math.floor(index)
    hi = math.ceil(index)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (index - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    print("file,case,section,method,reps,ranks,min_us,p10_us,median_us,p90_us,max_us")
    for file in args.files:
        raw = defaultdict(dict)
        all_ranks = defaultdict(set)
        with file.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            expected = {"case", "section", "method", "iteration", "rank", "us"}
            if set(reader.fieldnames or []) != expected:
                raise ValueError(f"{file}: wrong CSV columns")
            for row in reader:
                key = row["case"], row["section"], row["method"]
                iteration, rank, us = int(row["iteration"]), int(row["rank"]), float(row["us"])
                if iteration < 0 or rank < 0 or us < 0 or not math.isfinite(us):
                    raise ValueError(f"{file}: invalid sample {row}")
                pos = (rank, iteration)
                if pos in raw[key]:
                    raise ValueError(f"{file}: duplicate {key} {pos}")
                raw[key][pos] = us
                all_ranks[key].add(rank)
        if not raw:
            raise ValueError(f"{file}: empty CSV")
        for (case, section, method), samples in sorted(raw.items()):
            ranks = all_ranks[(case, section, method)]
            if ranks != set(range(max(ranks) + 1)):
                raise ValueError(f"{file}: missing MPI rank in {case}/{section}/{method}")
            reps = max(iteration for _, iteration in samples) + 1
            maxima = []
            for iteration in range(reps):
                values = [samples.get((rank, iteration)) for rank in sorted(ranks)]
                if any(v is None for v in values):
                    raise ValueError(f"{file}: missing rank sample at iter={iteration}")
                maxima.append(max(values))
            maxima.sort()
            print(f"{file.name},{case},{section},{method},{reps},{len(ranks)},"
                  f"{maxima[0]:.6f},{percentile(maxima, .1):.6f},"
                  f"{percentile(maxima, .5):.6f},{percentile(maxima, .9):.6f},"
                  f"{maxima[-1]:.6f}")


if __name__ == "__main__":
    main()
