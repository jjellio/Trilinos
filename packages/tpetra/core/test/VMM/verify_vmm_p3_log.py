#!/usr/bin/env python3
"""Validate P3 numerical results, matrix eligibility, and per-rank dispatch.

The matrix_generalized mode verifies M5.2 without prematurely assuming that
M5.4 removes raw Map-pointer dispatch checks.  Older profiles remain available.
"""

import argparse
from collections import Counter
from pathlib import Path
import re
import sys

SUMMARY = re.compile(
    r"Summary:\s*total\s*=\s*(\d+)\s*,\s*run\s*=\s*(\d+)\s*,"
    r"\s*passed\s*=\s*(\d+)\s*,\s*failed\s*=\s*(\d+)"
)
APPLY = re.compile(
    r'\[TPETRA_MATRIX\]\s+event=apply\s+path=(\w+)\s+'
    r'rank=(\d+)\s+label="([^"]+)"'
)
SKIP = re.compile(
    r'\[TPETRA_MATRIX\]\s+event=vmmSkip[^\n]*'
    r'label="([^"]+)"[^\n]*reason=(\w+)'
)
LABELS = (
    "P3_G01_Interleaved", "P3_G02_Permuted", "P3_G03_XDistinct",
    "P3_G03_YDistinct", "P3_G03_BothDistinct",
    "P3_G05_NonOneToOneDomain",
)


def check_output(content: str, vmm: bool, native: bool, stats: bool,
                 mode: str, ranks: int = 2):
    errors = []
    if "End Result: TEST PASSED" not in content:
        errors.append("Teuchos did not report End Result: TEST PASSED")
    summaries = [tuple(map(int, match.groups())) for match in SUMMARY.finditer(content)]
    if not any(total >= 5 and run == 5 and passed == 5 and failed == 0
               for total, run, passed, failed in summaries):
        errors.append(f"expected all five Map tests to run/pass, got {summaries}")

    found = APPLY.findall(content)
    if not stats:
        if found:
            errors.append(f"stats=0 unexpectedly reported {len(found)} apply events")
        return errors

    matrix_generalized = mode in ("matrix_generalized", "generalized")
    direct_matrix = vmm and native and matrix_generalized
    direct_distinct = vmm and native and mode == "generalized"
    want_paths = {
        "P3_G01_Interleaved": "vmm" if direct_matrix else "classic",
        "P3_G02_Permuted": "vmm" if direct_matrix else "classic",
        "P3_G03_XDistinct": "vmm" if direct_distinct else "classic",
        "P3_G03_YDistinct": "vmm" if direct_distinct else "classic",
        "P3_G03_BothDistinct": "vmm" if direct_distinct else "classic",
        # Non-1:1 domain is **never** valid for addressed CRS.
        "P3_G05_NonOneToOneDomain": "classic",
    }
    counts = Counter()
    for path, rank_str, label in found:
        rank = int(rank_str)
        if rank < 0 or rank >= ranks:
            errors.append(f"invalid rank {rank} for label={label!r}")
        elif label not in want_paths:
            errors.append(f"unexpected apply event label={label!r}")
        else:
            counts[(rank, label, path)] += 1
    for rank in range(ranks):
        for label, wanted in want_paths.items():
            for path in ("classic", "vmm", "alpha_zero"):
                observed = counts[(rank, label, path)]
                expected = int(path == wanted)
                if observed != expected:
                    errors.append(f"rank={rank} label={label} path={path} "
                                  f"observed={observed}, expected={expected}")

    if vmm:
        if native and mode != "baseline" and re.search(
            r"\[TPETRA_MV\]\s+event=vmmSkip[^\n]*"
            r"reason=(?:map_not_contiguous|map_not_one_to_one)\b", content
        ):
            errors.append("obsolete MultiVector Map-distribution rejection observed")
        found_skip = {label: reason for label, reason in SKIP.findall(content)
                      if label.startswith("P3_G")}
        expected_skips = {
            "P3_G01_Interleaved": None if matrix_generalized else "map_not_contiguous",
            "P3_G02_Permuted": None if matrix_generalized else "map_not_contiguous",
            "P3_G03_XDistinct": None,
            "P3_G05_NonOneToOneDomain": (
                "domain_map_not_one_to_one" if matrix_generalized else "map_not_contiguous"
            ),
        }
        for label, wanted in expected_skips.items():
            observed = found_skip.get(label)
            if observed != wanted:
                errors.append(f"matrix eligibility label={label} "
                              f"reason={observed!r}, expected={wanted!r}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--vmm", type=int, choices=(0, 1), required=True)
    parser.add_argument("--native", type=int, choices=(0, 1), required=True)
    parser.add_argument("--stats", type=int, choices=(0, 1), required=True)
    parser.add_argument("--mode", choices=("baseline", "allocation_only", "matrix_generalized", "generalized"),
                        default="matrix_generalized")
    parser.add_argument("--np", type=int, default=2)
    args = parser.parse_args()
    if args.np != 2:
        parser.error("P3 presently supports exactly NP=2")
    try:
        content = args.log.read_text(encoding="utf-8", errors="replace")
    except OSError as ex:
        print(f"[VMM_P3] result=FAIL error={ex}", file=sys.stderr)
        return 2
    errors = check_output(content, bool(args.vmm), bool(args.native),
                          bool(args.stats), args.mode, args.np)
    if errors:
        for error in errors:
            print(f"[VMM_P3] result=FAIL reason={error}", file=sys.stderr)
        return 1
    print(f"[VMM_P3] result=PASS mode={args.mode} vmm={args.vmm} "
          f"native={args.native} stats={args.stats} ranks={args.np} cases=5 "
          f"apply_events={args.np * len(LABELS) if args.stats else 0}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
