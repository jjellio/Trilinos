#!/usr/bin/env bash
# P1: two-rank, per-case direct SpMV versus classic fallback verification.
# Uses a fresh mpirun job for each Behavior environment configuration.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_vmm_p1_tests.sh --exe /path/to/TpetraCore_VmmRegression.exe [options]
  --exe PATH        Built Tpetra VmmRegression executable (required)
  --gpu-bind PATH   Rank-to-GPU wrapper (default: ~/cronus/gpu_bind.sh)
  --mpirun PATH     MPI launcher (default: mpirun)
  --np N            MPI ranks (P1's exact fixtures require 2)
  --outdir DIR      Logs + results.tsv (default: ./vmm-p1-results)
  --timeout SEC     Per-launch timeout in seconds (default: 120)
  --quick           Classic, VMM quiet, VMM with statistics (default)
  --full            All six VMM/NativeMV/Stats environment combinations
  -h, --help        Show usage

Example command for each scenario:
  TPETRA_EXPERIMENTAL_VMM=1 TPETRA_EXPERIMENTAL_VMM_NATIVE_MV=1 \
  TPETRA_EXPERIMENTAL_VMM_STATISTICS=1 TPETRA_EXPERIMENTAL_VMM_IPC=posix \
  mpirun --bind-to none -np 2 ~/cronus/gpu_bind.sh <exe> \
    --group=VMM_SpMV --globally-reduce-test-result

A stats-enabled run checks eight actual apply-path events *on each rank*
(S01, S02, S03, S04, four S06 nonzero-alpha calls) and two alpha-zero bypass
calls per rank.  Stats-disabled runs assert correctness and silent diagnostics.
USAGE
}

exe=''
gpu_bind="${HOME}/cronus/gpu_bind.sh"
mpi_exec=mpirun
np=2
outdir=./vmm-p1-results
timeout_sec=120
mode=quick

while (($#)); do
  case "$1" in
    --exe|--gpu-bind|--mpirun|--np|--outdir|--timeout)
      opt=$1
      if (($# < 2)); then echo "Missing argument for $opt" >&2; exit 2; fi
      val=$2
      case "$opt" in
        --exe) exe=$val ;;
        --gpu-bind) gpu_bind=$val ;;
        --mpirun) mpi_exec=$val ;;
        --np) np=$val ;;
        --outdir) outdir=$val ;;
        --timeout) timeout_sec=$val ;;
      esac
      shift 2
      ;;
    --quick) mode=quick; shift ;;
    --full) mode=full; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$exe" || ! -x "$exe" ]]; then
  echo "--exe must point to an executable: $exe" >&2; exit 2
fi
if [[ ! -x "$gpu_bind" ]]; then
  echo "GPU binding wrapper must be executable: $gpu_bind" >&2; exit 2
fi
if ! [[ "$np" =~ ^[0-9]+$ ]] || ((np != 2)); then
  echo "P1 is restricted to exactly --np 2" >&2; exit 2
fi
if ! [[ "$timeout_sec" =~ ^[0-9]+$ ]] || ((timeout_sec < 1)); then
  echo "--timeout must be a positive integer" >&2; exit 2
fi
command -v "$mpi_exec" >/dev/null 2>&1 || { echo "Missing MPI launcher" >&2; exit 2; }
command -v timeout >/dev/null 2>&1 || { echo "Missing GNU timeout" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "P1 requires Python 3 for log validation" >&2; exit 2; }

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
checker="$script_dir/verify_vmm_p1_log.py"
[[ -f "$checker" ]] || { echo "Missing P1 validator: $checker" >&2; exit 2; }
mkdir -p "$outdir"
outdir=$(cd "$outdir" && pwd)
results="$outdir/results.tsv"
printf 'scenario\tstatus\trc\tlog\n' > "$results"

run_mpi() {
  local log=$1
  shift
  timeout --signal=TERM --kill-after=5 "${timeout_sec}s" \
    "$mpi_exec" --bind-to none -np "$np" "$gpu_bind" "$exe" "$@" \
    >>"$log" 2>&1
}

preflight_log="$outdir/gpu-preflight.log"
{
  printf '# P1 GPU preflight: MPI ranks=%s\n' "$np"
  printf '# launcher=%s gpu_bind=%s executable=%s\n' "$mpi_exec" "$gpu_bind" "$exe"
} >"$preflight_log"
echo 'GPU placement preflight ...'
if run_mpi "$preflight_log" --vmm-gpu-preflight-only && \
   grep -q '\[VMM_TEST\] event=gpuPreflight result=PASS ranks=2' "$preflight_log"; then
  printf 'gpu-preflight\tPASS\t0\t%s\n' "$preflight_log" >> "$results"
else
  printf 'gpu-preflight\tFAIL\t1\t%s\n' "$preflight_log" >> "$results"
  echo "GPU preflight FAILED; see $preflight_log" >&2
  tail -n 35 "$preflight_log" >&2
  exit 1
fi

# scenario | global VMM | native MV | stats | expected direct apply path
cases=('classic|0|0|0|classic' 'vmm_nostats|1|1|0|vmm'
       'vmm_stats|1|1|1|vmm')
if [[ $mode == full ]]; then
  cases=('classic|0|0|0|classic' 'classic_stats|0|0|1|classic'
         'matrix_only|1|0|1|classic'
         'native_requested_global_off|0|1|1|classic'
         'vmm_nostats|1|1|0|vmm' 'vmm_stats|1|1|1|vmm')
fi

failures=0
printf '\n%-30s %-7s %s\n' SCENARIO STATUS LOG
for entry in "${cases[@]}"; do
  IFS='|' read -r name enabled native stats expected <<< "$entry"
  log="$outdir/${name}.log"
  {
    printf '# scenario=%s ranks=%s expected_path=%s\n' "$name" "$np" "$expected"
    printf '# TPETRA_EXPERIMENTAL_VMM=%s\n' "$enabled"
    printf '# TPETRA_EXPERIMENTAL_VMM_NATIVE_MV=%s\n' "$native"
    printf '# TPETRA_EXPERIMENTAL_VMM_STATISTICS=%s\n' "$stats"
    printf '# TPETRA_EXPERIMENTAL_VMM_IPC=posix\n'
    printf '# TPETRA_SYNCHRONIZE_APPLY=0 TPETRA_DEBUG=0\n'
  } >"$log"

  rc=0
  if TPETRA_EXPERIMENTAL_VMM="$enabled" \
     TPETRA_EXPERIMENTAL_VMM_NATIVE_MV="$native" \
     TPETRA_EXPERIMENTAL_VMM_STATISTICS="$stats" \
     TPETRA_EXPERIMENTAL_VMM_IPC=posix \
     TPETRA_SYNCHRONIZE_APPLY=0 \
     TPETRA_DEBUG=0 \
     run_mpi "$log" --group=VMM_SpMV --globally-reduce-test-result; then
    rc=0
  else
    rc=$?
  fi

  if ((rc == 0)) && python3 "$checker" --log "$log" \
         --expected "$expected" --stats "$stats" --np "$np" >>"$log" 2>&1; then
    status=PASS
  else
    status=FAIL
    if ((rc == 0)); then rc=1; fi
    failures=$((failures + 1))
    printf '\n--- %s: log tail ---\n' "$name" >&2
    tail -n 40 "$log" >&2
  fi
  printf '%-30s %-7s %s\n' "$name" "$status" "$log"
  printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$rc" "$log" >> "$results"
done

printf '\nResults: %s\n' "$results"
if ((failures > 0)); then
  echo "P1 sweep FAILED: $failures scenario(s)" >&2; exit 1
fi
echo "P1 sweep PASSED: ${#cases[@]} scenarios"
