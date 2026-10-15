#!/usr/bin/env python3
"""Verify P4 Teuchos completion, exact rank-local path/debug/timer evidence.

This verifier is intentionally independent of the test's C++ assertions.
A correct numerical result or successful MPI exit cannot mask wrong dispatch.
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
EMPTY = re.compile(r'\[VMM_P4\]\s+event=zeroRowsDispatch\s+rank=(\d+)\s+path=(\w+)')
DEBUG = re.compile(r'\[VMM_P4\]\s+event=debugCaught\s+case=(\w+)\s+rank=(\d+)\s+result=(\w+)')
DEBUG_SKIP = re.compile(r'\[VMM_P4\]\s+event=debugSkipped\s+case=(\w+)\s+rank=(\d+)\s+reason=(\w+)')
TIMER = re.compile(r'\[VMM_P4\]\s+event=timer\s+rank=(\d+)\s+path=(\w+)\s+direct_delta=(-?\d+)\s+classic_delta=(-?\d+)')
FABRIC = re.compile(r'\[VMM_P4\]\s+event=fabricProbe\s+rank=(\d+)\s+supported=([01])\s+reason=(\w+)')


def check(content: str, suite: str, vmm: bool, native: bool, stats: bool,
          mode: str, ranks: int = 2):
    errors = []
    expected_runs = {"boundary": 3, "debug": 9, "timer": 1, "probe": 1}[suite]
    if "End Result: TEST PASSED" not in content:
        errors.append("Teuchos failed or did not print the success trailer")
    summaries = [tuple(map(int, match.groups())) for match in SUMMARY.finditer(content)]
    if not any(run == expected_runs and passed == expected_runs and failed == 0
               and total >= run for total, run, passed, failed in summaries):
        errors.append(f"expected {expected_runs} completed tests, got {summaries}")

    found = APPLY.findall(content)
    counts = Counter((int(rank), label, path) for path, rank, label in found)
    wanted = {}
    if suite == "boundary":
        if stats:
            wanted["P4_B03_SkewedRemote"] = "vmm" if vmm and native else "classic"
            empty = EMPTY.findall(content)
            by_rank = Counter((int(rank), path) for rank, path in empty)
            if len(empty) != ranks:
                errors.append(f"expected {ranks} zero-row path markers, got {empty}")
            paths = []
            for rank in range(ranks):
                options = [path for r, path in empty if int(r) == rank]
                if len(options) != 1 or options[0] not in ("vmm", "classic"):
                    errors.append(f"rank={rank} invalid zero-row path marker {options}")
                else:
                    paths.append(options[0])
                    wanted["P4_B02_EmptyRank"] = paths[0]
                    if counts[(rank, "P4_B02_EmptyRank", paths[-1])] != 1:
                        errors.append(f"rank={rank} B02 apply path and marker disagree")
            if paths and len(set(paths)) != 1:
                errors.append(f"ranks disagreed on zero-row path: {paths}")
        elif EMPTY.findall(content):
            errors.append("stats=0 unexpectedly reported zero-row paths")
    elif suite == "debug":
        if not stats:
            errors.append("debug test requires statistics trace")
        old_cases = {"G06_InvalidX", "G06_InvalidY"}
        vmm_only = {"G07_NoImporterX", "G07_NoExporterY",
                    "G07_LocalOrdering", "G07_RankLocalX",
                    "G07_RankMixedClasses"}
        found_debug = DEBUG.findall(content)
        observed_debug = Counter((int(rank), case, result)
                                 for case, rank, result in found_debug)
        found_skips = DEBUG_SKIP.findall(content)
        observed_skips = Counter((int(rank), case, reason)
                                 for case, rank, reason in found_skips)
        expected_debug = old_cases | (vmm_only if vmm and native else set())
        expected_skips = set() if vmm and native else vmm_only
        for rank in range(ranks):
            for case in expected_debug:
                if observed_debug[(rank, case, "PASS")] != 1:
                    errors.append(f"rank={rank} missing validated exception {case}")
            for case in expected_skips:
                if observed_skips[(rank, case, "direct_vmm_unavailable")] != 1:
                    errors.append(f"rank={rank} missing classic-only skip {case}")
        if len(found_debug) != ranks * len(expected_debug):
            errors.append(f"unexpected debug exception markers: {found_debug}")
        if len(found_skips) != ranks * len(expected_skips):
            errors.append(f"unexpected debug skip markers: {found_skips}")

        # InvalidX is rejected before applyNonTranspose's path trace only
        # when native VMM debug validation is applicable. InvalidY uses an
        # Exporter, so the original DistObject error path remains intact.
        if not (vmm and native):
            wanted["P4_G06_InvalidX"] = "classic"
        wanted.update({
            "P4_G06_InvalidY": "classic",
            "P4_G06_EquivalentValid":
                "vmm" if vmm and native and mode == "generalized" else "classic",
            "P4_G07_EquivalentNoImporter":
                "vmm" if vmm and native and mode == "generalized" else "classic",
        })
    elif suite == "timer":
        if not stats:
            errors.append("timer test requires statistics trace")
        chosen = "vmm" if vmm and native else "classic"
        wanted["P4_T01_TimedRemote"] = chosen
        timers = TIMER.findall(content)
        count = Counter((int(rank), path, int(direct), int(classic))
                        for rank, path, direct, classic in timers)
        expected = (chosen, 1 if chosen == "vmm" else 0,
                    0 if chosen == "vmm" else 1)
        for rank in range(ranks):
            if count[(rank, *expected)] != 1:
                errors.append(f"rank={rank} missing expected timer {expected}")
        if len(timers) != ranks:
            errors.append(f"unexpected timer events: {timers}")
    elif suite == "probe":
        entries = FABRIC.findall(content)
        for rank in range(ranks):
            options = [(support, reason) for r, support, reason in entries if int(r) == rank]
            if len(options) != 1:
                errors.append(f"rank={rank} missing or duplicate Fabric probe {options}")
            elif (options[0][0] == "1") != (options[0][1] == "ok"):
                errors.append(f"rank={rank} inconsistent Fabric probe {options}")
        if len(entries) != ranks:
            errors.append(f"unexpected Fabric probe entries: {entries}")

    if not stats and found:
        errors.append(f"stats=0 unexpectedly reported apply events: {found}")
    elif stats:
        for rank in range(ranks):
            for label, expected in wanted.items():
                for path in ("classic", "vmm", "alpha_zero"):
                    got = counts[(rank, label, path)]
                    if got != (1 if path == expected else 0):
                        errors.append(f"rank={rank} label={label} path={path} "
                                      f"expected={int(path == expected)}, got={got}")
        if len(found) != ranks * len(wanted):
            errors.append(f"unexpected extra/missing apply events: {found}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--suite", choices=("boundary", "debug", "timer", "probe"),
                        required=True)
    parser.add_argument("--vmm", choices=(0, 1), type=int, required=True)
    parser.add_argument("--native", choices=(0, 1), type=int, required=True)
    parser.add_argument("--stats", choices=(0, 1), type=int, required=True)
    parser.add_argument("--mode", choices=("baseline", "allocation_only", "matrix_generalized", "generalized"),
                        default="generalized")
    parser.add_argument("--np", type=int, default=2)
    args = parser.parse_args()
    if args.np != 2:
        parser.error("P4 presently supports exactly NP=2")
    try:
        content = args.log.read_text(encoding="utf-8", errors="replace")
    except OSError as ex:
        print(f"[VMM_P4] result=FAIL error={ex}", file=sys.stderr)
        return 2
    errors = check(content, args.suite, bool(args.vmm), bool(args.native),
                   bool(args.stats), args.mode, args.np)
    for error in errors:
        print(f"[VMM_P4] result=FAIL reason={error}", file=sys.stderr)
    if errors:
        return 1
    print(f"[VMM_P4] result=PASS suite={args.suite} mode={args.mode} "
          f"vmm={args.vmm} native={args.native} stats={args.stats} "
          f"ranks={args.np}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
