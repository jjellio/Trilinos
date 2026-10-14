#!/usr/bin/env python3
"""Validate the P2 Teuchos suites and their rank-local classical dispatch.

P2 fallback tests intentionally do not take the direct VMM path, including
when both experimental VMM flags are enabled.  For transpose, the existing P1
instrumentation does not print path events, so we verify its analytical result
in C++ without pretending an apply trace exists.
"""

import argparse
from collections import Counter
from pathlib import Path
import re
import sys

SUMMARY_RE = re.compile(
    r"Summary:\s*total\s*=\s*(\d+)\s*,\s*run\s*=\s*(\d+)\s*,"
    r"\s*passed\s*=\s*(\d+)\s*,\s*failed\s*=\s*(\d+)"
)
PATH_RE = re.compile(
    r'\[TPETRA_MATRIX\]\s+event=apply\s+path=(\w+)\s+'
    r'rank=(\d+)\s+label="([^"]+)"'
)
# F05 transpose does not emit a P1 applyNonTranspose diagnostic; F07 runs
# only when the global VMM flag is disabled.  Every other fallback case
# emits *one* nontranspose path event per rank when statistics is enabled.
FALLBACK_EVENTS = {
    "P2_F01_OrdinaryInput": "classic",
    "P2_F02_ThreeColumns": "classic",
    "P2_F03_Exporter": "classic",
    "P2_F04_ReplicatedRange": "classic",
    "P2_F06_AlphaZero": "alpha_zero",
}


def check_output(content: str, suite: str, global_vmm: bool,
                 native_mv: bool, stats: bool, ranks: int):
    errors = []
    ntests = 7
    if "End Result: TEST PASSED" not in content:
        errors.append("Teuchos did not report End Result: TEST PASSED")
    summaries = [tuple(int(x) for x in match.groups())
                 for match in SUMMARY_RE.finditer(content)]
    if not any(total >= ntests and run == ntests and passed == ntests and failed == 0
               for total, run, passed, failed in summaries):
        errors.append(f"expected Teuchos run=passed={ntests}, failed=0; got {summaries}")

    found = PATH_RE.findall(content)
    if suite == "mv":
        if found:
            errors.append(f"MV suite unexpectedly launched {len(found)} SpMVs")
        if global_vmm and native_mv and stats and "storage=vmm" not in content:
            errors.append("full VMM MV suite lacks native allocation diagnostics")
        return errors

    if not stats:
        if found:
            errors.append(f"stats=0 unexpectedly emitted {len(found)} apply events")
        return errors

    expected_events = dict(FALLBACK_EVENTS)
    if not global_vmm:
        expected_events["P2_F07_GlobalDisabled"] = "classic"

    counts = Counter()
    for path, rank_string, label in found:
        rank = int(rank_string)
        if not 0 <= rank < ranks:
            errors.append(f"invalid rank={rank}")
        elif label not in expected_events:
            errors.append(f"unexpected P2 fallback apply label {label!r}")
        else:
            counts[(rank, label, path)] += 1

    for rank in range(ranks):
        for label, expected in expected_events.items():
            for path in ("classic", "vmm", "alpha_zero"):
                observed = counts[(rank, label, path)]
                wanted = int(path == expected)
                if observed != wanted:
                    errors.append(
                        f"rank={rank} label={label} path={path} "
                        f"observed={observed} expected={wanted}"
                    )
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--suite", choices=("mv", "fallback"), required=True)
    parser.add_argument("--vmm", type=int, choices=(0, 1), required=True)
    parser.add_argument("--native", type=int, choices=(0, 1), required=True)
    parser.add_argument("--stats", type=int, choices=(0, 1), required=True)
    parser.add_argument("--np", type=int, default=2)
    args = parser.parse_args()
    if args.np != 2:
        parser.error("P2 presently supports exactly NP=2")
    try:
        content = args.log.read_text(encoding="utf-8", errors="replace")
    except OSError as ex:
        print(f"[VMM_P2] result=FAIL reason=read_error error={ex}", file=sys.stderr)
        return 2
    errors = check_output(content, args.suite, bool(args.vmm),
                          bool(args.native), bool(args.stats), args.np)
    if errors:
        for error in errors:
            print(f"[VMM_P2] result=FAIL suite={args.suite} reason={error}",
                  file=sys.stderr)
        return 1
    print(f"[VMM_P2] result=PASS suite={args.suite} stats={args.stats} "
          f"vmm={args.vmm} native={args.native} ranks={args.np} cases=7")
    return 0


if __name__ == "__main__":
    sys.exit(main())
