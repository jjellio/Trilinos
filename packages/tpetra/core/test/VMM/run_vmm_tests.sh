#!/usr/bin/env bash
# P0: deterministic two-rank MPI/GPU smoke runner for the in-tree Tpetra VMM
# test executable.  Scenarios are separate jobs because Behavior caches env.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_vmm_tests.sh --exe /path/to/Tpetra_VmmRegression.exe [options]

  --exe PATH        Built TriBITS VmmRegression executable (required)
  --gpu-bind PATH   Rank-to-GPU launcher (default: ~/cronus/gpu_bind.sh)
  --mpirun PATH     MPI launcher (default: mpirun)
  --np N            Number of MPI ranks (P0 requires exactly 2)
  --outdir DIR      Logs + results.tsv (default: ./vmm-p0-logs)
  --timeout SEC     Per-launch timeout (default: 120)
  --quick           Preflight, classical and full VMM+stats (default)
  --full            Include no-stats and partial-enablement scenarios
  -h, --help        Show this help

The command launched is:
  mpirun --bind-to none -np 2 <gpu_bind.sh> <executable> <Teuchos args>

This P0 runner verifies GPU placement, numerical test success, and expected
VMM allocation statistics where enabled.  Kernel-path assertions are P1.
USAGE
}

exe=''
gpu_bind="${HOME}/cronus/gpu_bind.sh"
mpi_exec='mpirun'
np=2
outdir='./vmm-p0-logs'
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
  echo "Expected --exe to point to an executable file: $exe" >&2
  exit 2
fi
if [[ ! -x "$gpu_bind" ]]; then
  echo "GPU binding wrapper missing or not executable: $gpu_bind" >&2
  exit 2
fi
if ! [[ "$np" =~ ^[0-9]+$ ]] || ((np != 2)); then
  echo "P0 only supports --np 2 (received '$np')" >&2
  exit 2
fi
if ! [[ "$timeout_sec" =~ ^[0-9]+$ ]] || ((timeout_sec < 1)); then
  echo "--timeout must be a positive integer" >&2
  exit 2
fi
if ! command -v "$mpi_exec" >/dev/null 2>&1; then
  echo "MPI launcher not found: $mpi_exec" >&2
  exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required" >&2
  exit 2
fi

mkdir -p "$outdir"
outdir=$(cd "$outdir" && pwd)
results="$outdir/results.tsv"
printf 'scenario\tstatus\trc\tlog\n' > "$results"

# timeout is external to MPI: a deadlocked job cannot hang the sweep forever.
# The MPI job receives identical VMM environment variables on every rank.
run_mpi() {
  local log=$1
  shift
  local rc=0
  if timeout --signal=TERM --kill-after=5 "${timeout_sec}s" \
      "$mpi_exec" --bind-to none -np "$np" "$gpu_bind" "$exe" "$@" \
      >>"$log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  return "$rc"
}

preflight_log="$outdir/gpu-preflight.log"
{
  printf '# P0 GPU preflight: MPI ranks=%s\n' "$np"
  printf '# launcher=%s bind=%s exe=%s\n' "$mpi_exec" "$gpu_bind" "$exe"
} > "$preflight_log"
echo 'GPU placement preflight ...'
if run_mpi "$preflight_log" --vmm-gpu-preflight-only && \
   grep -q '\[VMM_TEST\] event=gpuPreflight result=PASS ranks=2' "$preflight_log"; then
  printf 'gpu-preflight\tPASS\t0\t%s\n' "$preflight_log" >> "$results"
else
  rc=$?
  # If the command ran successfully but omitted the expected marker, do not
  # mistake it for a successful preflight.
  if ((rc == 0)); then rc=1; fi
  printf 'gpu-preflight\tFAIL\t%s\t%s\n' "$rc" "$preflight_log" >> "$results"
  echo "GPU preflight FAILED; see $preflight_log" >&2
  tail -n 35 "$preflight_log" >&2
  exit 1
fi

declare -a cases=(
  'classic|0|0|0'
  'vmm_stats|1|1|1'
)
if [[ "$mode" == full ]]; then
  cases=(
    'classic|0|0|0'
    'classic_stats|0|0|1'
    'matrix_only|1|0|1'
    'native_requested_global_off|0|1|1'
    'vmm_nostats|1|1|0'
    'vmm_stats|1|1|1'
  )
fi

failed=0
printf '\n%-30s %-7s %s\n' 'SCENARIO' 'STATUS' 'LOG'
for case_line in "${cases[@]}"; do
  IFS='|' read -r name enable native stats <<< "$case_line"
  log="$outdir/${name}.log"
  {
    printf '# scenario=%s ranks=%s\n' "$name" "$np"
    printf '# TPETRA_EXPERIMENTAL_VMM=%s\n' "$enable"
    printf '# TPETRA_EXPERIMENTAL_VMM_NATIVE_MV=%s\n' "$native"
    printf '# TPETRA_EXPERIMENTAL_VMM_STATISTICS=%s\n' "$stats"
    printf '# TPETRA_EXPERIMENTAL_VMM_IPC=posix\n'
    printf '# TPETRA_SYNCHRONIZE_APPLY=0 TPETRA_DEBUG=0\n'
  } > "$log"

  rc=0
  if TPETRA_EXPERIMENTAL_VMM="$enable" \
     TPETRA_EXPERIMENTAL_VMM_NATIVE_MV="$native" \
     TPETRA_EXPERIMENTAL_VMM_STATISTICS="$stats" \
     TPETRA_EXPERIMENTAL_VMM_IPC=posix \
     TPETRA_SYNCHRONIZE_APPLY=0 \
     TPETRA_DEBUG=0 \
     run_mpi "$log" --group=VMM_Smoke --globally-reduce-test-result; then
    rc=0
  else
    rc=$?
  fi

  # The Teuchos main prints this only after the selected unit tests complete.
  # Also check that it actually ran a test rather than an empty filter match.
  if ((rc == 0)) && \
     grep -q 'End Result: TEST PASSED' "$log" && \
     grep -Eq 'Summary:.*run = [1-9]' "$log"; then
    if [[ "$name" == vmm_stats ]] && \
       ! grep -Eq '\[TPETRA_MV\].*event=allocate.*storage=vmm' "$log"; then
      status=FAIL
      rc=1
      printf '# ERROR: VMM+stats did not report a native VMM MV allocation\n' >> "$log"
    else
      status=PASS
    fi
  else
    status=FAIL
    if ((rc == 0)); then rc=1; fi
  fi
  if [[ "$status" == FAIL ]]; then
    failed=$((failed + 1))
    printf '\n--- %s: last 35 lines ---\n' "$name" >&2
    tail -n 35 "$log" >&2
  fi
  printf '%-30s %-7s %s\n' "$name" "$status" "$log"
  printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$rc" "$log" >> "$results"
done

printf '\nResults: %s\n' "$results"
if ((failed)); then
  printf 'P0 sweep FAILED: %s scenario(s) failed\n' "$failed" >&2
  exit 1
fi
printf 'P0 sweep PASSED: %s scenarios\n' "${#cases[@]}"
