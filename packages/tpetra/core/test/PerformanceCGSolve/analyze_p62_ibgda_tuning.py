#!/usr/bin/env python3
"""Analyze bracketing P6.2 IBGDA parameter experiments from manifest.tsv.

One independent MPI launch yields one median over per-iteration maximum-rank times.
Compare each candidate launch to the geometric mean of the two *adjacent*
IBGDA-enabled, unset/unset references for that workload and pass. Do not count
50 iterations as 50 independent launches. Python 3 standard library only.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
import math
from pathlib import Path
import re
import statistics
import sys

DIRECT_RE = re.compile(r"\[NVSHMEM P6\.2 DIRECT\][^\n]*")
BARRIER_RE = re.compile(r"\[NVSHMEM P6\.2 BARRIER\][^\n]*")
SETUP_RE = re.compile(r"\[NVSHMEM P6\.2 SETUP\][^\n]*")


def match_field(line: str, name: str, required=True):
    m = re.search(r"(?:^|\s)" + re.escape(name) + r"=([^\s]+)", line)
    if m is None:
        if required:
            raise ValueError(f"missing {name} in {line[:200]}")
        return None
    return m.group(1)


def sample_validation(log_path: Path, method: str, reported: float, num_ranks: int):
    """Recompute only one method's median from the existing raw rank CSV."""
    prefix = str(log_path)[:-4]
    path = Path(prefix + "_matrix_p62_samples.csv")
    if not path.exists():
        return "missing"
    grouped = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["section"] == "DIRECT" and row["method"] == method:
                grouped[int(row["iteration"])].append((int(row["rank"]), float(row["us"])))
    if not grouped:
        return "empty"
    max_per_iter = []
    for _, items in sorted(grouped.items()):
        if len(items) != num_ranks or len(set(r for r, _ in items)) != num_ranks:
            return "rank_mismatch"
        max_per_iter.append(max(x for _, x in items))
    actual = statistics.median(max_per_iter)
    return "ok" if abs(actual - reported) <= 0.05 else f"median_mismatch:{actual:.5f}"


def read_run(entry, *, num_ranks: int, validate_csv: bool):
    row = {k: entry[k] for k in entry}
    row["status"] = ""
    row["p62_us"] = None
    row["tpetra_us"] = None
    row["barrier_us"] = None
    row["warn_gpu_handler_fallback"] = 0
    row["csv_check"] = "skipped"
    path = Path(entry["log"])
    if entry["exit_code"] != "0":
        row["status"] = "skipped_" + entry["exit_code"]
        return row
    if not path.exists():
        row["status"] = "missing_log"
        return row
    text = path.read_text(errors="replace")
    if "[NVSHMEM P6.2] SUMMARY status=PASS" not in text:
        row["status"] = "no_PASS_summary"
        return row
    row["warn_gpu_handler_fallback"] = text.count("ibgda_nic_mem_gpu_map failed")
    if "chunkDoubles=" not in text:
        row["status"] = "missing_chunk_metadata"
        return row
    try:
        setup = SETUP_RE.search(text).group()
        row["global_descriptors"] = int(match_field(setup, "totalPutDescriptors"))
        row["global_cross_node_doubles"] = int(match_field(setup, "crossNodeDoubles"))
        methods = {}
        for line in DIRECT_RE.findall(text):
            m = match_field(line, "method")
            if m in ("descriptor_two_stream", "tpetra_doImport"):
                if m in methods:
                    raise ValueError(f"multiple {m} lines")
                methods[m] = float(match_field(line, "median"))
        row["p62_us"] = methods["descriptor_two_stream"]
        row["tpetra_us"] = methods["tpetra_doImport"]
        barrier_lines = BARRIER_RE.findall(text)
        if len(barrier_lines) != 1:
            raise ValueError(f"expected exactly one barrier line, got {len(barrier_lines)}")
        row["barrier_us"] = float(match_field(barrier_lines[0], "median"))
        if validate_csv:
            row["csv_check"] = sample_validation(path, "descriptor_two_stream", row["p62_us"], num_ranks)
            if row["csv_check"] != "ok":
                raise ValueError("raw rank CSV check: " + row["csv_check"])
        row["status"] = "PASS"
    except (AttributeError, KeyError, ValueError) as err:
        row["status"] = "parse_error:" + str(err).replace("\t", " ")
    return row


def write_csv(path, rows, columns):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fmt(num, digits=2):
    if num is None:
        return "n/a"
    return f"{num:.{digits}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_directory", type=Path)
    ap.add_argument("--no-raw-csv-check", action="store_true",
                    help="skip recomputation of each P6.2 launch median from rank samples")
    args = ap.parse_args()
    folder = args.output_directory.resolve()
    manifest_path = folder / "manifest.tsv"
    if not manifest_path.exists():
        ap.error(f"no manifest.tsv in {folder}")
    with manifest_path.open(newline="") as f:
        manifest = list(csv.DictReader(f, delimiter="\t"))
    if not manifest:
        ap.error("empty manifest")
    config = (folder / "config.env.txt").read_text()
    num_ranks = int(re.search(r"^np=(\d+)$", config, re.M).group(1))
    runs = [read_run(x, num_ranks=num_ranks, validate_csv=not args.no_raw_csv_check)
            for x in manifest]
    output_cols = ["run_id", "stage", "pass", "variant", "chunk", "role", "ibgda", "batch", "rc",
                   "exit_code", "status", "p62_us", "tpetra_us", "barrier_us",
                   "global_descriptors", "global_cross_node_doubles", "warn_gpu_handler_fallback", "csv_check", "log"]
    write_csv(folder / "runs.csv", runs, output_cols)
    grouped = defaultdict(list)
    for r in runs:
        grouped[(r["stage"], r["pass"], r["variant"], r["chunk"])].append(r)
    paired = []
    drift = []
    issues = []
    for (stage, passid, variant, chunk), group in sorted(grouped.items()):
        before = [r for r in group if r["role"] == "reference_before" and r["status"] == "PASS"]
        after = [r for r in group if r["role"] == "reference_after" and r["status"] == "PASS"]
        if len(before) != 1 or len(after) != 1:
            issues.append(f"missing reference pair for {stage} pass={passid} {variant}:{chunk}")
            continue
        baseline = math.sqrt(before[0]["p62_us"] * after[0]["p62_us"])
        drift_pct = (after[0]["p62_us"] / before[0]["p62_us"] - 1.0) * 100
        drift.append({"stage": stage, "pass": passid, "variant": variant,
                      "chunk": chunk, "reference_before_us": before[0]["p62_us"],
                      "reference_after_us": after[0]["p62_us"],
                      "reference_geomean_us": baseline, "drift_pct": drift_pct})
        for r in group:
            if r["role"] in ("candidate", "off_control") and r["status"] == "PASS":
                paired.append({"stage": stage, "pass": passid, "variant": variant,
                               "chunk": chunk, "role": r["role"], "ibgda": r["ibgda"],
                               "batch": r["batch"], "rc": r["rc"],
                               "candidate_us": r["p62_us"], "reference_us": baseline,
                               "improvement_pct": 100 * (baseline / r["p62_us"] - 1),
                               "tpetra_us": r["tpetra_us"],
                               "barrier_us": r["barrier_us"],
                               "reference_drift_pct": drift_pct,
                               "run_id": r["run_id"]})
    write_csv(folder / "paired.csv", paired,
              ["stage", "pass", "variant", "chunk", "role", "ibgda", "batch", "rc", "candidate_us",
               "reference_us", "improvement_pct", "tpetra_us", "barrier_us", "reference_drift_pct", "run_id"])
    write_csv(folder / "reference_drift.csv", drift,
              ["stage", "pass", "variant", "chunk", "reference_before_us", "reference_after_us",
               "reference_geomean_us", "drift_pct"])
    by_setting = defaultdict(list)
    for r in paired:
        by_setting[(r["stage"], r["variant"], r["chunk"], r["role"], r["batch"], r["rc"])].append(r)
    aggregates = []
    for key, records in sorted(by_setting.items()):
        stage, variant, chunk, role, batch, rc = key
        improvements = [r["improvement_pct"] for r in records]
        latencies = [r["candidate_us"] for r in records]
        aggregates.append({"stage": stage, "variant": variant, "chunk": chunk,
                           "role": role, "batch": batch, "rc": rc,
                           "launches": len(records), "median_p62_us": statistics.median(latencies),
                           "median_improvement_pct": statistics.median(improvements),
                           "min_improvement_pct": min(improvements),
                           "max_improvement_pct": max(improvements),
                           "median_tpetra_us": statistics.median(x["tpetra_us"] for x in records)})
    write_csv(folder / "summary.csv", aggregates,
              ["stage", "variant", "chunk", "role", "batch", "rc", "launches",
               "median_p62_us", "median_improvement_pct", "min_improvement_pct",
               "max_improvement_pct", "median_tpetra_us"])

    passed = sum(1 for r in runs if r["status"] == "PASS")
    failed = [r for r in runs if r["status"] != "PASS"]
    if failed:
        issues.append(f"{len(failed)} run(s) failed or had unreadable timings")
    if drift:
        worst = max(drift, key=lambda r: abs(r["drift_pct"]))
        worst_drift = f"{abs(worst['drift_pct']):.2f}% ({worst['stage']} {worst['variant']}:{worst['chunk']} pass {worst['pass']})"
    else:
        worst_drift = "unavailable"

    lines = [
        "# P6.2 IBGDA tuning: generated result summary", "",
        f"- Independent MPI launches parsed: **{len(runs)}**; completed PASS: **{passed}**; missing/failed: **{len(failed)}**.",
        "- Primary metric: median of per-iteration maximum-rank completed P6.2 Import latency (microseconds).",
        "- Paired reference: geometric mean of the two *unset-batch/unset-RC, IBGDA-on* MPI launches bracketing each candidate block.",
        "- Improvement = 100 × (bracketing-reference / candidate − 1): positive means the candidate is faster.",
        "- Each candidate is tested in independent launches; the 50 in-process iterations are not independent MPI experiments.",
        f"- Largest within-block reference drift: **{worst_drift}**.",
        "- The experimental MPI_Barrier occurs before the timed GPU Import; reported Import times exclude the barrier.",
        "- `IBGDA=0` off-controls are compared with nearby IBGDA-on default references; their positive improvement means *disabling IBGDA* was faster.",
        "", "## Paired parameter effects", "",
        "| Stage | Mode | Chunk | Candidate | MPI launches | P6.2 median (µs) | Paired improvement | Range across launches |",
        "|---|---|---:|---|---:|---:|---:|---:|"
    ]
    for a in aggregates:
        cand = "IBGDA off" if a["role"] == "off_control" else f"batch={a['batch']}, RC={a['rc']}"
        lines.append("| {} | {} | {} | {} | {} | {} | {:+.2f}% | {:+.2f}% to {:+.2f}% |".format(
            a["stage"], a["variant"], a["chunk"], cand, a["launches"],
            fmt(a["median_p62_us"]), a["median_improvement_pct"],
            a["min_improvement_pct"], a["max_improvement_pct"]))
    lines.extend(["", "## Interpretation rules", "",
                  "- Treat an improvement as promising only when it repeats across independent passes and is larger than control drift.",
                  "- RC and batching were swept independently; do not infer their combined optimum without the `combo` stage.",
                  "- Keep `unset/unset` as a distinct reference, even if explicitly setting the documented numeric defaults happens to yield similar results.",
                  "- Any missing, timed-out, or failed-correctness launch invalidates its candidate measurement.",
                  "- For small gains (around 1–3%) repeat a confirmatory alternated run instead of treating an unreplicated minimum as a winner.",
                  "", "## Data files", "",
                  "`runs.csv`: per-launch medians, PASS, fallback warnings and CSV validation;",
                  "`paired.csv`: each independent candidate versus its two adjacent references;",
                  "`reference_drift.csv`: within-block drift; `summary.csv`: median paired effects by setting.", ""])
    if issues:
        lines.extend(["## Issues", ""])
        lines.extend("- " + issue for issue in issues)
    (folder / "analysis.md").write_text("\n".join(lines) + "\n")
    print(f"Parsed {len(runs)} launches; PASS={passed}; failed={len(failed)}")
    print(f"Largest reference drift: {worst_drift}")
    for a in sorted(aggregates, key=lambda x: (x["stage"], x["variant"], int(x["chunk"]), -x["median_improvement_pct"])):
        if a["role"] == "candidate":
            print("{} {:5} k={:<5} batch={:<7} rc={:<7} runs={} median={:8.2f}us paired={:+6.2f}%".format(
                a["stage"], a["variant"], a["chunk"], a["batch"], a["rc"],
                a["launches"], a["median_p62_us"], a["median_improvement_pct"]))
    print("Output:", folder / "analysis.md")
    if issues:
        print("WARNING: " + "; ".join(issues), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
