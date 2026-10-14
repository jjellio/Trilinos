#!/usr/bin/env bash
# P2: independent two-rank MultiVector and classical fallback test sweeps.
# Each scenario/group is a fresh MPI process (Behavior caches environment).
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_vmm_p2_tests.sh --exe /path/to/TpetraCore_VmmRegression.exe [options]
  --exe PATH        In-tree Tpetra regression executable (required)
  --gpu-bind PATH   Rank-to-GPU binding wrapper (default: ~/cronus/gpu_bind.sh)
  --mpirun PATH     MPI launcher (default: mpirun)
  --np N            MPI ranks (P2 requires 2)
  --outdir DIR      Per-case logs and results.tsv (default: ./vmm-p2-results)
  --timeout SEC     MPI job timeout (default: 180)
  --quick           Classic, full VMM quiet, full VMM+stats (default)
  --full            All six environment configurations, both test groups
  --suite KIND      both (default), mv, or fallback
  -h, --help        Show usage

MPI launch: mpirun --bind-to none -np 2 <gpu_bind.sh> <exe> <Teuchos args>
CTest alternative (using configured MPI_EXEC_* flags):
  ctest -j1 -R VMM_P2_NP2 -V
USAGE
}

exe=''
gpu_bind="${HOME}/cronus/gpu_bind.sh"
mpi_exec=mpirun
np=2
outdir=./vmm-p2-results
timeout_sec=180
mode=quick
suite=both

while (($#)); do
  case "$1" in
    --exe|--gpu-bind|--mpirun|--np|--outdir|--timeout|--suite)
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
        --suite) suite=$value ;;
      esac
      shift 2
      ;;
    --quick) mode=quick; shift ;;
    --full) mode=full; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$exe" || ! -x "$exe" ]]; then
  echo "--exe must name an executable: $exe" >&2; exit 2
fi
if [[ ! -x "$gpu_bind" ]]; then
  echo "--gpu-bind wrapper is not executable: $gpu_bind" >&2; exit 2
fi
if ! [[ "$np" =~ ^[0-9]+$ ]] || ((np != 2)); then
  echo "P2 supports exactly --np 2" >&2; exit 2
fi
if ! [[ "$timeout_sec" =~ ^[0-9]+$ ]] || ((timeout_sec < 1)); then
  echo "--timeout must be a positive integer" >&2; exit 2
fi
if [[ "$suite" != both && "$suite" != mv && "$suite" != fallback ]]; then
  echo "--suite must be one of both, mv, or fallback" >&2; exit 2
fi
command -v "$mpi_exec" >/dev/null 2>&1 || { echo "MPI launcher missing" >&2; exit 2; }
command -v timeout >/dev/null 2>&1 || { echo "GNU timeout missing" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "Python 3 missing" >&2; exit 2; }

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
validator="$script_dir/verify_vmm_p2_log.py"
[[ -f "$validator" ]] || { echo "Missing P2 validator: $validator" >&2; exit 2; }
mkdir -p "$outdir"
outdir=$(cd "$outdir" && pwd)
results="$outdir/results.tsv"
printf 'scenario\tsuite\tstatus\trc\tlog\n' > "$results"

run_mpi() {
  local logfile=$1
  shift
  timeout --signal=TERM --kill-after=5 "${timeout_sec}s" \
    "$mpi_exec" --bind-to none -np "$np" "$gpu_bind" "$exe" "$@" \
    >> "$logfile" 2>&1
}

preflight="$outdir/gpu-preflight.log"
{
  printf '# P2 GPU preflight ranks=%s\n' "$np"
  printf '# launcher=%s bind=%s exe=%s\n' "$mpi_exec" "$gpu_bind" "$exe"
} > "$preflight"
echo 'P2 GPU placement preflight ...'
if run_mpi "$preflight" --vmm-gpu-preflight-only && \
   grep -q '\[VMM_TEST\] event=gpuPreflight result=PASS ranks=2' "$preflight"; then
  printf 'preflight\tgpu\tPASS\t0\t%s\n' "$preflight" >> "$results"
else
  printf 'preflight\tgpu\tFAIL\t1\t%s\n' "$preflight" >> "$results"
  echo "P2 preflight failed; see $preflight" >&2
  tail -n 30 "$preflight" >&2
  exit 1
fi

cases=('classic|0|0|0' 'vmm_nostats|1|1|0' 'vmm_stats|1|1|1')
if [[ "$mode" == full ]]; then
  cases=('classic|0|0|0' 'classic_stats|0|0|1'
         'matrix_only|1|0|1' 'native_requested_global_off|0|1|1'
         'vmm_nostats|1|1|0' 'vmm_stats|1|1|1')
fi
suites=(mv fallback)
if [[ "$suite" != both ]]; then suites=("$suite"); fi

failures=0
completed=0
printf '\n%-29s %-10s %-7s %s\n' SCENARIO GROUP STATUS LOG
for entry in "${cases[@]}"; do
  IFS='|' read -r name enabled native stats <<< "$entry"
  for group in "${suites[@]}"; do
    if [[ "$group" == mv ]]; then
      teuchos_group=VMM_MultiVector
    else
      teuchos_group=VMM_Fallback
    fi
    log="$outdir/${name}_${group}.log"
    {
      printf '# scenario=%s suite=%s ranks=%s\n' "$name" "$group" "$np"
      printf '# TPETRA_EXPERIMENTAL_VMM=%s\n' "$enabled"
      printf '# TPETRA_EXPERIMENTAL_VMM_NATIVE_MV=%s\n' "$native"
      printf '# TPETRA_EXPERIMENTAL_VMM_STATISTICS=%s\n' "$stats"
      printf '# TPETRA_EXPERIMENTAL_VMM_IPC=posix\n'
      printf '# TPETRA_SYNCHRONIZE_APPLY=0 TPETRA_DEBUG=0\n'
    } > "$log"

    rc=0
    if TPETRA_EXPERIMENTAL_VMM="$enabled" \
       TPETRA_EXPERIMENTAL_VMM_NATIVE_MV="$native" \
       TPETRA_EXPERIMENTAL_VMM_STATISTICS="$stats" \
       TPETRA_EXPERIMENTAL_VMM_IPC=posix \
       TPETRA_SYNCHRONIZE_APPLY=0 \
       TPETRA_DEBUG=0 \
       run_mpi "$log" --group="$teuchos_group" --globally-reduce-test-result; then
      rc=0
    else
      rc=$?
    fi

    if ((rc == 0)) && python3 "$validator" --log "$log" --suite "$group" \
        --vmm "$enabled" --native "$native" --stats "$stats" --np "$np" \
        >> "$log" 2>&1; then
      status=PASS
    else
      status=FAIL
      if ((rc == 0)); then rc=1; fi
      failures=$((failures + 1))
      printf '\n--- %s/%s: log tail ---\n' "$name" "$group" >&2
      tail -n 50 "$log" >&2
    fi
    completed=$((completed + 1))
    printf '%-29s %-10s %-7s %s\n' "$name" "$group" "$status" "$log"
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$group" "$status" "$rc" "$log" >> "$results"
  done
done

printf '\nResults: %s\n' "$results"
if ((failures > 0)); then
  echo "P2 sweep FAILED: $failures/$completed launches" >&2
  exit 1
fi
echo "P2 sweep PASSED: $completed launches"
