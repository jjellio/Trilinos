#!/usr/bin/env bash
# P3: GPU-bound two-rank Map eligibility and Map identity regression sweep.
# Fresh mpirun for every case because Tpetra::Details::Behavior caches flags.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_vmm_p3_tests.sh --exe /path/to/TpetraCore_VmmRegression.exe [options]
  --exe PATH        Built in-tree Tpetra VMM regression executable (required)
  --gpu-bind PATH   GPU binding wrapper (default ~/cronus/gpu_bind.sh)
  --mpirun PATH     MPI launcher (default mpirun)
  --np N            MPI ranks (P3 requires 2)
  --outdir DIR      Per-scenario logs and results.tsv (default ./vmm-p3-results)
  --timeout SEC     Per-MPI job timeout (default 180)
  --mode MODE       baseline (default), generalized (future acceptance oracle)
  --quick           Classic, full VMM without stats, full VMM+stats (default)
  --full            All six VMM/native/stats combinations
  -h, --help        Show this help

Launch: mpirun --bind-to none -np 2 <gpu_bind.sh> <exe> --group=VMM_Map
CTest: ctest -j1 -R VMM_P3_NP2 -V
Generalized mode is expected to FAIL until the Map allocation/index refactor.
USAGE
}

exe=''
gpu_bind="${HOME}/cronus/gpu_bind.sh"
mpi_exec=mpirun
np=2
outdir='./vmm-p3-results'
timeout_sec=180
mode=quick
map_mode=baseline

while (($#)); do
  case "$1" in
    --exe|--gpu-bind|--mpirun|--np|--outdir|--timeout|--mode)
      option=$1
      if (($# < 2)); then echo "Missing argument for $option" >&2; exit 2; fi
      value=$2
      case "$option" in
        --exe) exe=$value ;;
        --gpu-bind) gpu_bind=$value ;;
        --mpirun) mpi_exec=$value ;;
        --np) np=$value ;;
        --outdir) outdir=$value ;;
        --timeout) timeout_sec=$value ;;
        --mode) map_mode=$value ;;
      esac
      shift 2
      ;;
    --quick) mode=quick; shift ;;
    --full) mode=full; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$exe" && -x "$exe" ]] || { echo "Invalid --exe: $exe" >&2; exit 2; }
[[ -x "$gpu_bind" ]] || { echo "GPU binder missing/not executable: $gpu_bind" >&2; exit 2; }
[[ "$np" =~ ^[0-9]+$ ]] && ((np == 2)) || { echo 'P3 requires --np 2' >&2; exit 2; }
[[ "$timeout_sec" =~ ^[0-9]+$ ]] && ((timeout_sec > 0)) || { echo 'Invalid --timeout' >&2; exit 2; }
[[ "$map_mode" == baseline || "$map_mode" == generalized ]] || { echo 'Invalid --mode' >&2; exit 2; }
command -v "$mpi_exec" >/dev/null 2>&1 || { echo "MPI launcher not found: $mpi_exec" >&2; exit 2; }
command -v timeout >/dev/null 2>&1 || { echo 'GNU timeout is required' >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo 'Python3 is required' >&2; exit 2; }

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
validator="$script_dir/verify_vmm_p3_log.py"
[[ -f "$validator" ]] || { echo "Missing validator: $validator" >&2; exit 2; }
mkdir -p "$outdir"
outdir=$(cd "$outdir" && pwd)
results="$outdir/results.tsv"
printf 'scenario\tmode\tstatus\trc\tlog\n' > "$results"

run_mpi() {
  local log=$1
  shift
  timeout --signal=TERM --kill-after=5 "${timeout_sec}s" \
    "$mpi_exec" --bind-to none -np "$np" "$gpu_bind" "$exe" "$@" \
    >> "$log" 2>&1
}

preflight="$outdir/gpu-preflight.log"
{
  printf '# P3 GPU preflight mode=%s ranks=%s\n' "$map_mode" "$np"
  printf '# launcher=%s bind=%s exe=%s\n' "$mpi_exec" "$gpu_bind" "$exe"
} > "$preflight"
echo 'P3 GPU placement preflight ...'
if run_mpi "$preflight" --vmm-gpu-preflight-only && \
   grep -q '\[VMM_TEST\] event=gpuPreflight result=PASS ranks=2' "$preflight"; then
  printf 'preflight\t%s\tPASS\t0\t%s\n' "$map_mode" "$preflight" >> "$results"
else
  printf 'preflight\t%s\tFAIL\t1\t%s\n' "$map_mode" "$preflight" >> "$results"
  echo "P3 GPU preflight failed; see $preflight" >&2
  tail -n 30 "$preflight" >&2
  exit 1
fi

cases=('classic|0|0|0' 'vmm_nostats|1|1|0' 'vmm_stats|1|1|1')
if [[ "$mode" == full ]]; then
  cases=('classic|0|0|0' 'classic_stats|0|0|1'
         'matrix_only|1|0|1' 'native_requested_global_off|0|1|1'
         'vmm_nostats|1|1|0' 'vmm_stats|1|1|1')
fi

failures=0
completed=0
printf '\n%-31s %-13s %-7s %s\n' SCENARIO MODE STATUS LOG
for entry in "${cases[@]}"; do
  IFS='|' read -r name enabled native stats <<< "$entry"
  log="$outdir/${name}_map.log"
  {
    printf '# P3 scenario=%s mode=%s ranks=%s\n' "$name" "$map_mode" "$np"
    printf '# TPETRA_EXPERIMENTAL_VMM=%s\n' "$enabled"
    printf '# TPETRA_EXPERIMENTAL_VMM_NATIVE_MV=%s\n' "$native"
    printf '# TPETRA_EXPERIMENTAL_VMM_STATISTICS=%s\n' "$stats"
    printf '# TPETRA_VMM_TEST_MAP_MODE=%s\n' "$map_mode"
    printf '# TPETRA_EXPERIMENTAL_VMM_IPC=posix\n'
    printf '# TPETRA_SYNCHRONIZE_APPLY=0 TPETRA_DEBUG=0\n'
  } > "$log"
  rc=0
  if TPETRA_EXPERIMENTAL_VMM="$enabled" \
     TPETRA_EXPERIMENTAL_VMM_NATIVE_MV="$native" \
     TPETRA_EXPERIMENTAL_VMM_STATISTICS="$stats" \
     TPETRA_VMM_TEST_MAP_MODE="$map_mode" \
     TPETRA_EXPERIMENTAL_VMM_IPC=posix \
     TPETRA_SYNCHRONIZE_APPLY=0 \
     TPETRA_DEBUG=0 \
     run_mpi "$log" --group=VMM_Map --globally-reduce-test-result; then
    rc=0
  else
    rc=$?
  fi
  if ((rc == 0)) && python3 "$validator" --log "$log" \
      --vmm "$enabled" --native "$native" --stats "$stats" \
      --mode "$map_mode" --np "$np" >> "$log" 2>&1; then
    status=PASS
  else
    status=FAIL
    if ((rc == 0)); then rc=1; fi
    failures=$((failures + 1))
    printf '\n--- P3 %s: last 60 log lines ---\n' "$name" >&2
    tail -n 60 "$log" >&2
  fi
  completed=$((completed + 1))
  printf '%-31s %-13s %-7s %s\n' "$name" "$map_mode" "$status" "$log"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$map_mode" "$status" "$rc" "$log" >> "$results"
done

printf '\nResults: %s\n' "$results"
if ((failures)); then
  echo "P3 sweep FAILED: $failures/$completed launches" >&2
  exit 1
fi
echo "P3 sweep PASSED: $completed launches"
