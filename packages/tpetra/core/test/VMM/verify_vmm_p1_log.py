#!/usr/bin/env python3
"""Validate P1 Teuchos correctness and *each rank's* selected SpMV path.

Stats-on runs inspect the event=apply messages emitted by CrsMatrix with
TPETRA_EXPERIMENTAL_VMM_STATISTICS=1; stats-off runs verify no such diagnostic
messages were produced.  This checker never infers dispatch from allocations.
"""

import argparse
from collections import Counter
from pathlib import Path
import re
import sys

# One nonzero-alpha apply in each of S01-S04; S06 has four nonzero-alpha
# applies plus two alpha==0 early exits that must bypass both kernels.
EXPECTED_APPLIES = {
    "P1_S01_Diagonal": (1, 0),
    "P1_S02_Boundary": (1, 0),
    "P1_S03_AllRemote": (1, 0),
    "P1_S04_Stencil": (1, 0),
    "P1_S06_AlphaBeta": (4, 2),
}
PATH_RE = re.compile(
    r'\[TPETRA_MATRIX\]\s+event=apply\s+path=(\w+)\s+'
    r'rank=(\d+)\s+label="([^"]+)"'
)
SUMMARY_RE = re.compile(
    r"Summary:\s*total\s*=\s*(\d+)\s*,\s*run\s*=\s*(\d+)\s*,"
    r"\s*passed\s*=\s*(\d+)\s*,\s*failed\s*=\s*(\d+)"
)


def check_output(content: str, expected: str, stats: bool, ranks: int):
    errors = []
    if "End Result: TEST PASSED" not in content:
        errors.append("Teuchos did not report End Result: TEST PASSED")
    summaries = [tuple(int(x) for x in m.groups()) for m in SUMMARY_RE.finditer(content)]
    ntests = len(EXPECTED_APPLIES)
    if not any(total >= ntests and run == ntests and passed == ntests and failed == 0
               for total, run, passed, failed in summaries):
        errors.append(f"expected Teuchos summary run=passed={ntests}, failed=0; got {summaries}")

    found = PATH_RE.findall(content)
    if not stats:
        if found:
            errors.append(f"stats=0 unexpectedly emitted {len(found)} apply-path events")
        return errors

    counts = Counter()
    for path, rank_text, label in found:
        rank = int(rank_text)
        if rank < 0 or rank >= ranks:
            errors.append(f"apply event reported invalid rank={rank}")
            continue
        if label not in EXPECTED_APPLIES:
            errors.append(f"unexpected apply event label={label!r}")
            continue
        if path not in ("vmm", "classic", "alpha_zero"):
            errors.append(f"unexpected apply path={path!r}")
            continue
        counts[(path, rank, label)] += 1

    for rank in range(ranks):
        for label, (active, early) in EXPECTED_APPLIES.items():
            for path in ("vmm", "classic", "alpha_zero"):
                expected_count = early if path == "alpha_zero" else (
                    active if path == expected else 0)
                actual_count = counts[(path, rank, label)]
                if actual_count != expected_count:
                    errors.append(
                        f"rank={rank} label={label}: path={path} count={actual_count}"
                        f" expected={expected_count}"
                    )

    if expected == "vmm" and "storage=vmm" not in content:
        errors.append("expected native VMM allocation evidence in statistics output")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--expected", choices=("classic", "vmm"), required=True)
    parser.add_argument("--stats", type=int, choices=(0, 1), required=True)
    parser.add_argument("--np", type=int, default=2)
    args = parser.parse_args()
    if args.np < 1:
        parser.error("--np must be >= 1")
    try:
        content = args.log.read_text(encoding="utf-8", errors="replace")
    except OSError as ex:
        print(f"[VMM_P1] result=FAIL reason=read_error error={ex}", file=sys.stderr)
        return 2
    errors = check_output(content, args.expected, bool(args.stats), args.np)
    if errors:
        for reason in errors:
            print(f"[VMM_P1] result=FAIL reason={reason}", file=sys.stderr)
        return 1
    print(f"[VMM_P1] result=PASS expected_path={args.expected} stats={args.stats}"
          f" ranks={args.np} cases={len(EXPECTED_APPLIES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
