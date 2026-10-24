#!/usr/bin/env python3
"""Run and validate one P5.2 GPU/cluster correctness case.

Only Python's standard library is required. No shell evaluation is used.
Exit nonzero on failed/absent summaries, missing ranks, wrong placement,
unexpected case skips, or a nonzero MPI launcher return code.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

TOPOLOGIES = {
    "smoke1": {"ranks": 1, "ppn": 1, "nodes": 1},
    "intra2": {"ranks": 2, "ppn": 2, "nodes": 1},
    "intra4": {"ranks": 4, "ppn": 4, "nodes": 1},
    "inter2": {"ranks": 2, "ppn": 1, "nodes": 2},
    "inter16": {"ranks": 16, "ppn": 8, "nodes": 2},
}
MIN_PES = {
    "matrix": 1, "identity": 1, "permute": 1, "sparse-mixed": 2,
    "one-way": 2, "uneven": 2, "zero-owner": 2, "all-empty": 1,
    "multi-owner": 3, "fanout": 3,
}
# This is the number of messages crossing the two physical nodes, *not*
# merely any remote traffic.  The one-way and zero-owner rank 0 <-> 1
# relationship is intra-node in inter16, and all-empty never transfers.
CROSSING = {
    "inter2": {"matrix", "sparse-mixed", "one-way", "uneven", "zero-owner"},
    "inter16": {"matrix", "sparse-mixed", "uneven", "multi-owner", "fanout"},
}

LINE_RE = re.compile(r"\[NVSHMEM P5\.2(?: (CASE))?\]\s*(.*)")
P0_RE = re.compile(r"\[NVSHMEM P0\]\s+mpi_rank=(\d+)\s+nvshmem_pe=(\d+)")
FIELD_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)=(\S+)")


def die(reason: str, log: Path | None = None) -> None:
    print("FAIL: " + reason, file=sys.stderr)
    if log and log.is_file():
        print("Last log lines (full log: " + str(log) + "):", file=sys.stderr)
        for line in log.read_text(errors="replace").splitlines()[-45:]:
            print(line, file=sys.stderr)
    raise SystemExit(1)


def parse_log(text: str):
    begin = []
    cases = []
    summaries = []
    ranks = {}
    for line in text.splitlines():
        match = P0_RE.search(line)
        if match:
            rank = int(match.group(1))
            ranks.setdefault(rank, set()).add(int(match.group(2)))
        m = LINE_RE.search(line)
        if m:
            fields = dict(FIELD_RE.findall(m.group(2)))
            if m.group(1) == "CASE":
                cases.append(fields)
            elif "SUMMARY" in m.group(2):
                summaries.append(fields)
            elif "BEGIN" in m.group(2):
                begin.append(fields)
    return begin, cases, summaries, ranks


def verify_log(text: str, topo: str, case: str, size: int, epochs: int, cycles: int):
    cfg = TOPOLOGIES[topo]
    n = cfg["ranks"]
    begin, reported, summaries, ranks = parse_log(text)
    expected_cases = {c for c, minimum in MIN_PES.items() if minimum <= n}
    expected_skips = len(MIN_PES) - len(expected_cases)
    if case != "all":
        expected_cases = {case}
        expected_skips = 0

    if len(begin) != 1 or len(summaries) != 1:
        raise ValueError(f"expected exactly one BEGIN and SUMMARY; got {len(begin)} / {len(summaries)}")
    b, s = begin[0], summaries[0]
    if b.get("ranks") != str(n) or b.get("physicalNodes") != str(cfg["nodes"]):
        raise ValueError(f"MPI node/rank topology incorrect: {b.get('ranks')} ranks, "
                         f"{b.get('physicalNodes')} nodes; expected {n}/{cfg['nodes']}")
    if b.get("perImportGlobalSync") != "NONE":
        raise ValueError("not the point-to-point P5.2 build (BEGIN sync flag wrong)")
    if s.get("status") != "PASS" or s.get("sync") != "POINT_TO_POINT_GENERATION_CREDIT":
        raise ValueError(f"P5.2 summary failed or wrong implementation: {s}")
    if s.get("globalSyncsInImport") != "0":
        raise ValueError("the transport unexpectedly reports a global Import sync")
    if s.get("casesPassed") != str(len(expected_cases)) or s.get("casesSkipped") != str(expected_skips):
        raise ValueError(f"incorrect case/skip accounting: {s}; expected {len(expected_cases)}/{expected_skips}")
    if len(reported) != len(expected_cases):
        raise ValueError(f"expected {len(expected_cases)} CASE records; got {len(reported)}")
    seen = [f.get("case") for f in reported]
    if set(seen) != expected_cases or len(set(seen)) != len(seen):
        raise ValueError(f"missing/extra/duplicate cases: actual {seen}; expected {sorted(expected_cases)}")
    for entry in reported:
        if entry.get("status") != "PASS" or entry.get("globalSyncsInImport") != "0":
            raise ValueError(f"case did not pass zero-world-collective gate: {entry}")
        if entry.get("sync") != "POINT_TO_POINT_GENERATION_CREDIT":
            raise ValueError(f"case uses wrong sync algorithm: {entry}")
    if set(ranks) != set(range(n)):
        raise ValueError(f"P0 GPU/NVSHMEM session rank reports missing or duplicated: {sorted(ranks)}")
    if any(len(pes) != 1 for pes in ranks.values()):
        raise ValueError(f"inconsistent rank-to-PE mapping reported: {ranks}")
    if {next(iter(pes)) for pes in ranks.values()} != set(range(n)):
        raise ValueError(f"MPI->NVSHMEM PE mapping is not a permutation: {ranks}")
    if "gpuPutCalls" not in s or "gpuPutElements" not in s:
        raise ValueError("missing actual GPU PUT accounting")
    transfers = int(s["gpuPutElements"])
    calls = int(s["gpuPutCalls"])
    ready = int(s.get("readyStreamWaits", "-1"))
    ack = int(s.get("gpuAckSignals", "-1"))
    if transfers < 0 or calls < 0 or calls != ready or calls != ack:
        raise ValueError("GPU PUT, ready wait, ACK counters do not reconcile")
    if any(c in expected_cases for c in CROSSING.get(topo, set())):
        if int(s.get("crossNodePutCalls", "0")) <= 0:
            raise ValueError("no cross-node GPU PUTs despite crossing cases")

    # Strong known-fixture checks: previously measured P5.1 miniFE size=100
    # at 16 ranks, for the same ordering/distribution and two physical nodes.
    if topo == "inter16" and size == 100 and case in ("matrix", "all"):
        repetitions = epochs * cycles
        expected_peers = 44 if case == "matrix" else 141
        expected_values = 309105 if case == "matrix" else 309205
        expected_cross_values = 20614 if case == "matrix" else 20640
        expected_cross_calls = 10 if case == "matrix" else 36
        if calls != expected_peers * repetitions or transfers != expected_values * repetitions:
            raise ValueError(f"16-PE miniFE known fixture mismatch: {calls} PUTs / {transfers} elements")
        if int(s.get("creditStreamWaits", "-1")) != expected_peers * cycles * (epochs-1):
            raise ValueError("incorrect previous-generation credit waits")
        if int(s.get("crossNodePutElements", "-1")) != expected_cross_values * repetitions:
            raise ValueError("cross-node PUT element counts differ from accepted P5.1 fixture")
        if int(s.get("crossNodePutCalls", "-1")) != expected_cross_calls * repetitions:
            raise ValueError("cross-node PUT call counts differ from accepted P5.1 fixture")
    return {
        "status": "PASS", "topology": topo, "case": case,
        "casesPassed": len(expected_cases), "casesSkipped": expected_skips,
        "gpuPutCalls": calls, "gpuPutElements": transfers,
        "gpuAckSignals": ack, "gpuReadyWaits": ready,
        "gpuCreditWaits": int(s.get("creditStreamWaits", "-1")),
        "crossNodePutCalls": int(s.get("crossNodePutCalls", "0")),
    }


def make_command(args):
    cfg = TOPOLOGIES[args.topology]
    n, ppn, nodes = cfg["ranks"], cfg["ppn"], cfg["nodes"]
    c = [args.mpirun]
    if args.hostfile:
        c += ["--hostfile", args.hostfile]
    if args.mpi_options:
        c += shlex.split(args.mpi_options)
    c += ["--map-by", f"ppr:{ppn}:node", "--bind-to", "none", "-np", str(n)]
    if args.bind_script:
        c.append(args.bind_script)
    c += [args.exe, "--cuda", f"--numgpus={args.numgpus}", f"--size={args.size}",
          "--nvshmem-p52", f"--nvshmem-p52-case={args.case}",
          f"--nvshmem-p52-epochs={args.epochs}",
          f"--nvshmem-p52-cycles={args.cycles}",
          f"--nvshmem-p52-skew-ms={args.skew_ms}"]
    if nodes == 1:
        c.append("--nvshmem-p52-max-nodes=1")
    else:
        c += ["--nvshmem-p52-min-nodes=2", "--nvshmem-p52-max-nodes=2"]
        if args.case == "all" or args.case in CROSSING[args.topology]:
            c.append("--nvshmem-p52-cross-node")
    if not args.show_peers:
        c.append("--no-nvshmem-p52-peers")
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topology", choices=TOPOLOGIES, required=True)
    ap.add_argument("--case", choices=["all", *MIN_PES], default="all")
    ap.add_argument("--exe", required=True)
    ap.add_argument("--mpirun", default="mpirun")
    ap.add_argument("--bind-script", default="", help="Optional proven per-rank CUDA_VISIBLE_DEVICES wrapper")
    ap.add_argument("--numgpus", type=int, required=True, help="Visible CUDA GPUs per rank AFTER its wrapper")
    ap.add_argument("--hostfile", default="")
    ap.add_argument("--mpi-options", default="", help="Additional mpirun arguments, e.g. --mca pml ucx")
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--skew-ms", type=int, default=0)
    ap.add_argument("--show-peers", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--log-dir", default="./p52-test-logs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    n = TOPOLOGIES[args.topology]["ranks"]
    if args.case != "all" and MIN_PES[args.case] > n:
        ap.error(f"{args.case} requires at least {MIN_PES[args.case]} ranks, not {n}")
    if args.size is None:
        args.size = 100 if args.topology == "inter16" and args.case in ("all", "matrix") else 8
    if args.numgpus < 1 or args.epochs < 4 or args.cycles < 2 or args.size < 1:
        ap.error("numgpus and size must be positive; P5.2 requires epochs>=4, cycles>=2")
    if args.skew_ms < 0 or args.skew_ms > 3000:
        ap.error("P5.2 skew must be 0..3000 milliseconds")
    if args.timeout < 10:
        ap.error("timeout must be >=10 seconds")
    cmd = make_command(args)
    print("LAUNCH: " + shlex.join(cmd), flush=True)
    if args.dry_run:
        print("DRY-RUN: no MPI/GPU process started")
        return
    if not Path(args.exe).is_file():
        die("executable missing: " + args.exe)
    if not os.access(args.exe, os.X_OK):
        die("executable not executable: " + args.exe)
    if args.bind_script and (not Path(args.bind_script).is_file() or
                             not os.access(args.bind_script, os.X_OK)):
        die("GPU binding script missing or not executable: " + args.bind_script)
    if not shutil.which(args.mpirun):
        die("MPI launcher unavailable: " + args.mpirun)
    logdir = Path(args.log_dir)
    logdir.mkdir(parents=True, exist_ok=True)
    variant = f"skew{args.skew_ms}" if args.skew_ms else "normal"
    stem = f"p52_{args.topology}_{args.case}_{variant}_n{args.size}_e{args.epochs}_c{args.cycles}"
    logpath = logdir / (stem + ".log")
    resultpath = logdir / (stem + ".json")
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    began = time.monotonic()
    with logpath.open("w") as log:
        # Start as a process group so an MPI hang is killed as a whole.
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            ret = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            die(f"MPI run timed out after {args.timeout}s", logpath)
    duration = round(time.monotonic() - began, 3)
    data = {"startedUtc": started, "elapsedSeconds": duration,
            "command": cmd, "log": str(logpath), "mpiExitCode": ret,
            "nvshmemEnv": {k: os.environ.get(k, "<unset>") for k in
                          ("NVSHMEM_REMOTE_TRANSPORT", "NVSHMEM_IB_ENABLE_IBGDA", "NVSHMEM_SYMMETRIC_SIZE")}}
    if ret:
        resultpath.write_text(json.dumps({**data, "status": "FAIL"}, indent=2))
        die(f"MPI run exited with status {ret}", logpath)
    try:
        verified = verify_log(logpath.read_text(errors="replace"), args.topology,
                              args.case, args.size, args.epochs, args.cycles)
    except (ValueError, KeyError) as exc:
        resultpath.write_text(json.dumps({**data, "status": "FAIL", "reason": str(exc)}, indent=2))
        die(str(exc), logpath)
    resultpath.write_text(json.dumps({**data, **verified}, indent=2))
    print(f"PASS: {args.topology}/{args.case} cases={verified['casesPassed']} "
          f"puts={verified['gpuPutCalls']} elems={verified['gpuPutElements']} "
          f"elapsed={duration}s | {logpath}")


if __name__ == "__main__":
    main()
