#!/usr/bin/env python3
"""Summarize rank-local VMM apply dispatch and rejection reasons by matrix label.

Usage:
  python3 packages/tpetra/core/test/VMM/summarize_vmm_apply_rejections.py muelu.log
  python3 packages/tpetra/core/test/VMM/summarize_vmm_apply_rejections.py muelu.log --by-rank

Only event=vmmSkip phase=apply represents apply rejection; fillComplete
vmmSkip events and Belos pool diagnostics are deliberately excluded.
Transpose calls have no event=apply path and are counted separately.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re

PATH_RE = re.compile(
    r'\[TPETRA_MATRIX\]\s+event=apply\s+path=(\S+)\s+'
    r'rank=(\d+)\s+label="([^"]*)"'
)
REJECTION_RE = re.compile(
    r'\[TPETRA_MATRIX\]\s+event=vmmSkip\s+phase=apply\s+'
    r'reason=(\S+)\s+rank=(\d+)\s+object=\S+\s+label="([^"]*)"'
)


def parse(contents, by_rank=False):
    paths = defaultdict(Counter)
    reasons = defaultdict(Counter)
    for line in contents.splitlines():
        p = PATH_RE.search(line)
        if p:
            path, rank, label = p.groups()
            key = (label, int(rank)) if by_rank else (label,)
            paths[key][path] += 1
        r = REJECTION_RE.search(line)
        if r:
            reason, rank, label = r.groups()
            key = (label, int(rank)) if by_rank else (label,)
            reasons[key][reason] += 1
    return paths, reasons


def render(paths, reasons):
    keys = sorted(paths.keys() | reasons.keys())
    if not keys:
        return "No VMM apply statistics found. Enable TPETRA_EXPERIMENTAL_VMM_STATISTICS=1."
    lines = []
    header = f"{'Operator':<27} {'VMM':>7} {'Classic':>8} {'Alpha0':>7} {'Trans':>7}  Classic rejection reasons"
    lines.append(header)
    lines.append("-" * len(header))
    for key in keys:
        label = key[0] if key[0] else "<unlabeled>"
        if len(key) == 2:
            label += f" [rank={key[1]}]"
        p, r = paths[key], reasons[key]
        trans = r.get("transpose_unsupported", 0)
        classic_reasons = Counter({k: v for k, v in r.items()
                                   if k != "transpose_unsupported"})
        formatted = ", ".join(f"{reason}:{n}" for reason, n in
                              sorted(classic_reasons.items(), key=lambda x: (-x[1], x[0])))
        lines.append(f"{label:<27} {p['vmm']:>7} {p['classic']:>8} "
                     f"{p['alpha_zero']:>7} {trans:>7}  {formatted or '-'}")
        if p['classic'] != sum(classic_reasons.values()):
            lines.append("  WARNING: classic apply count != nontranspose rejections "
                         "(possible missing/interleaved output or different log versions)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="MueLu or Tpetra statistics log")
    parser.add_argument("--by-rank", action="store_true",
                        help="Report separate rows for each MPI rank")
    args = parser.parse_args()
    contents = args.log.read_text(encoding="utf-8", errors="replace")
    paths, reasons = parse(contents, args.by_rank)
    print(render(paths, reasons))


if __name__ == "__main__":
    main()
