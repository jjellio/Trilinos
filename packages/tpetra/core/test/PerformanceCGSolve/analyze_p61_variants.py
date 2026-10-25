#!/usr/bin/env python3
"""Compare direct P6.1 per-iteration rank maxima for thread/warp/block.

Run on raw *_samples.csv files after an identical P6 sweep. Threads/warps/
blocks are matched by replacing their distinct filename token, including
matching pass number; never silently combine different pass/topology labels.
The phase `put_signal` medians are auxiliary and per-rank maxima cannot be
summed with other phase maxima.
"""
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

VARIANT_METHODS = {
    "thread": "nvshmem_packed_put",
    "warp": "nvshmem_packed_put_warp",
    "block": "nvshmem_packed_put_block",
}


def median(values):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("invalid timing values")
    ordered = sorted(values)
    half = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[half]
    return (ordered[half-1] + ordered[half]) / 2


def summarize(path):
    raw = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {"case", "section", "method", "iteration", "rank", "us"}:
            raise ValueError(f"wrong P6 CSV schema: {path}")
        for row in reader:
            key = (row["case"], row["section"], row["method"])
            rank, rep, us = int(row["rank"]), int(row["iteration"]), float(row["us"])
            if rank < 0 or rep < 0 or not math.isfinite(us) or us < 0:
                raise ValueError(f"invalid timing row in {path}: {row}")
            if (rank, rep) in raw[key]:
                raise ValueError(f"duplicate sample {key} rank={rank} rep={rep} in {path}")
            raw[key][(rank, rep)] = us
    summaries = {}
    for key, samples in raw.items():
        ranks = max(r for r, _ in samples) + 1
        reps = max(i for _, i in samples) + 1
        if len(samples) != ranks * reps:
            raise ValueError(f"missing samples {key} in {path}")
        ranks_seen = {r for r, _ in samples}
        if ranks_seen != set(range(ranks)):
            raise ValueError(f"missing rank in {path}")
        worst = [max(samples[(r, i)] for r in range(ranks)) for i in range(reps)]
        summaries[key] = {"median": median(worst), "ranks": ranks, "reps": reps}
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    data = []
    references = {}
    for path in args.files:
        name = path.name
        pattern = re.search(r"_(thread|warp|block)_", name)
        if not pattern:
            raise ValueError(f"file lacks explicit variant token: {name}")
        variant = pattern.group(1)
        pair_key = name[:pattern.start()] + "_VARIANT_" + name[pattern.end():]
        summary = summarize(path)
        cases = {case for case, sec, method in summary if sec == "DIRECT"}
        if not cases:
            raise ValueError(f"no direct comparison in {path}")
        for case in cases:
            t = summary[(case, "DIRECT", "tpetra_doImport")]
            p = summary[(case, "DIRECT", VARIANT_METHODS[variant])]
            if t["ranks"] != p["ranks"] or t["reps"] != p["reps"]:
                raise ValueError(f"unpaired direct data {path}")
            put_stage = summary.get((case, "PHASE", "put_signal"))
            row = {"file": name, "case": case, "pair_key": pair_key, "variant": variant,
                   "ranks": t["ranks"], "reps": t["reps"],
                   "tpetra": t["median"], "put": p["median"],
                   "put_stage": put_stage["median"] if put_stage else None}
            data.append(row)
            ref_key = (pair_key, case)
            if variant == "thread":
                if ref_key in references:
                    raise ValueError(f"duplicate thread reference for {ref_key}")
                references[ref_key] = row
    print("file,case,variant,ranks,reps,tpetra_median_us,put_median_us,put_signal_phase_median_us,tpetra_over_put,thread_over_variant")
    for row in sorted(data, key=lambda x:(x["pair_key"],x["case"],{"thread":0,"warp":1,"block":2}[x["variant"]])):
        base = references.get((row["pair_key"], row["case"]))
        if row["put"] <= 0:
            raise ValueError(f"nonpositive variant timing {row['file']}")
        thread_ratio = f"{base['put']/row['put']:.5f}" if base else "NA"
        stage = f"{row['put_stage']:.6f}" if row["put_stage"] is not None else "NA"
        print(f"{row['file']},{row['case']},{row['variant']},{row['ranks']},{row['reps']},"
              f"{row['tpetra']:.6f},{row['put']:.6f},{stage},"
              f"{row['tpetra']/row['put']:.5f},{thread_ratio}")


if __name__ == "__main__":
    main()
