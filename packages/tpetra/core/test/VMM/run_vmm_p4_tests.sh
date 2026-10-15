#!/usr/bin/env bash
# P4: boundary, debug-only negative Map, optional synchronous timers/Fabric.
# Separate MPI jobs are mandatory because Tpetra::Behavior caches env values.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_vmm_p4_tests.sh --exe /path/to/TpetraCore_VmmRegression.exe [options]
  --exe PATH        In-tree regression executable (required)
  --gpu-bind PATH   GPU binding wrapper (default ~/cronus/gpu_bind.sh)
  --mpirun PATH     MPI launcher (default mpirun)
  --np N            MPI ranks; P4 requires exactly 2
  --outdir DIR      Logs and results.tsv (default ./vmm-p4-results)
  --timeout SEC     Wall-time limit per MPI job (default 180)
  --mode MODE       matrix_generalized (default M5.2), allocation_only (M5.1),
                    baseline (M4.1), or generalized (future M5.4 acceptance)
  --quick           Boundary classic, boundary VMM stats, debug classic,
                    VMM timer, and Fabric capability probe (default)
  --full            3 boundary, 2 debug jobs (nine tests each), 2 timer,
                    1 Fabric probe
  --fabric          After capability probe, optionally run Fabric VMM timer
  -h, --help        Help

Normal test jobs use TPETRA_SYNCHRONIZE_APPLY=0. Timer diagnostic jobs
explicitly set it to 1; no production apply() changes are made.
USAGE
}

exe=''
gpu_bind="${HOME}/cronus/gpu_bind.sh"
mpi_exec=mpirun
np=2
outdir=./vmm-p4-results
timeout_sec=180
mode=quick
map_mode=matrix_generalized
fabric_requested=0
while (($#)); do
  case "$1" in
    --exe|--gpu-bind|--mpirun|--np|--outdir|--timeout|--mode)
      if (($# < 2)); then echo "Missing argument for $1" >&2; exit 2; fi
      option=$1
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
      shift 2 ;;
    --quick) mode=quick; shift ;;
    --full) mode=full; shift ;;
    --fabric) fabric_requested=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$exe" && -x "$exe" ]] || { echo "Invalid --exe: $exe" >&2; exit 2; }
[[ -x "$gpu_bind" ]] || { echo "GPU binder missing or not executable: $gpu_bind" >&2; exit 2; }
[[ "$np" =~ ^[0-9]+$ ]] && ((np == 2)) || { echo 'P4 requires --np 2' >&2; exit 2; }
[[ "$timeout_sec" =~ ^[0-9]+$ ]] && ((timeout_sec > 0)) || { echo 'Invalid --timeout' >&2; exit 2; }
[[ "$map_mode" == baseline || "$map_mode" == allocation_only || "$map_mode" == matrix_generalized || "$map_mode" == generalized ]] || { echo 'Invalid --mode' >&2; exit 2; }
# A pre-hotfix P3 variable is not recognized by Tpetra::Behavior.
if [[ ${TPETRA_VMM_TEST_MAP_MODE+x} ]]; then
  echo "Obsolete TPETRA_VMM_TEST_MAP_MODE is set; run: unset TPETRA_VMM_TEST_MAP_MODE" >&2
  exit 2
fi
command -v "$mpi_exec" >/dev/null 2>&1 || { echo "Missing MPI launcher: $mpi_exec" >&2; exit 2; }
command -v timeout >/dev/null 2>&1 || { echo 'GNU timeout is required' >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo 'Python3 is required' >&2; exit 2; }

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
validator="$script_dir/verify_vmm_p4_log.py"
[[ -f "$validator" ]] || { echo "Missing P4 validator: $validator" >&2; exit 2; }
mkdir -p "$outdir"
outdir=$(cd "$outdir" && pwd)
results="$outdir/results.tsv"
printf 'scenario\tsuite\tstatus\trc\tlog\n' > "$results"

run_mpi() {
  local log=$1
  shift
  timeout --signal=TERM --kill-after=5 "${timeout_sec}s" \
      "$mpi_exec" --bind-to none -np "$np" "$gpu_bind" "$exe" "$@" \
      >> "$log" 2>&1
}
preflight="$outdir/gpu-preflight.log"
{
  printf '# P4 preflight ranks=%s\n' "$np"
  printf '# launcher=%s binder=%s exe=%s\n' "$mpi_exec" "$gpu_bind" "$exe"
} > "$preflight"
echo 'P4 GPU placement preflight ...'
if run_mpi "$preflight" --vmm-gpu-preflight-only &&
   grep -q '\[VMM_TEST\] event=gpuPreflight result=PASS ranks=2' "$preflight"; then
  printf 'preflight\tgpu\tPASS\t0\t%s\n' "$preflight" >> "$results"
else
  printf 'preflight\tgpu\tFAIL\t1\t%s\n' "$preflight" >> "$results"
  echo "P4 GPU preflight FAILED: $preflight" >&2
  tail -n 40 "$preflight" >&2
  exit 1
fi

# name | test suite | enabled VMM | native MV | statistics | debug | sync | IPC
cases=(
  'boundary_classic|boundary|0|0|0|0|0|posix'
  'boundary_vmm_stats|boundary|1|1|1|0|0|posix'
  'debug_classic|debug|0|0|1|1|0|posix'
  'timer_vmm|timer|1|1|1|0|1|posix'
  'fabric_probe|probe|0|0|0|0|0|posix'
)
if [[ "$mode" == full ]]; then
  cases=(
    'boundary_classic|boundary|0|0|0|0|0|posix'
    'boundary_vmm_nostats|boundary|1|1|0|0|0|posix'
    'boundary_vmm_stats|boundary|1|1|1|0|0|posix'
    'debug_classic|debug|0|0|1|1|0|posix'
    'debug_vmm|debug|1|1|1|1|0|posix'
    'timer_classic|timer|0|0|1|0|1|posix'
    'timer_vmm|timer|1|1|1|0|1|posix'
    'fabric_probe|probe|0|0|0|0|0|posix'
  )
fi

failures=0
completed=0
fabric_supported=0
printf '\n%-26s %-11s %-8s %s\n' SCENARIO SUITE STATUS LOG
for entry in "${cases[@]}"; do
  IFS='|' read -r name suite enabled native stats debug sync ipc <<< "$entry"
  log="$outdir/${name}.log"
  {
    printf '# P4 scenario=%s suite=%s mode=%s ranks=%s\n' "$name" "$suite" "$map_mode" "$np"
    printf '# TPETRA_EXPERIMENTAL_VMM=%s\n' "$enabled"
    printf '# TPETRA_EXPERIMENTAL_VMM_NATIVE_MV=%s\n' "$native"
    printf '# TPETRA_EXPERIMENTAL_VMM_STATISTICS=%s\n' "$stats"
    printf '# TPETRA_EXPERIMENTAL_VMM_IPC=%s\n' "$ipc"
    printf '# TPETRA_DEBUG=%s TPETRA_SYNCHRONIZE_APPLY=%s\n' "$debug" "$sync"
    printf '# VMM_TEST_MAP_MODE=%s\n' "$map_mode"
  } > "$log"

  rc=0
  group=''
  case "$suite" in
    boundary) group=VMM_Boundary ;;
    debug) group=VMM_Debug ;;
    timer) group=VMM_Timer ;;
    probe) group=VMM_FabricProbe ;;
  esac
  if TPETRA_EXPERIMENTAL_VMM="$enabled" \
     TPETRA_EXPERIMENTAL_VMM_NATIVE_MV="$native" \
     TPETRA_EXPERIMENTAL_VMM_STATISTICS="$stats" \
     TPETRA_EXPERIMENTAL_VMM_IPC="$ipc" \
     TPETRA_DEBUG="$debug" \
     TPETRA_SYNCHRONIZE_APPLY="$sync" \
     VMM_TEST_MAP_MODE="$map_mode" \
     run_mpi "$log" "--group=$group" --globally-reduce-test-result; then
    rc=0
  else
    rc=$?
  fi
  if ((rc == 0)) && python3 "$validator" --log "$log" --suite "$suite" \
       --vmm "$enabled" --native "$native" --stats "$stats" \
       --mode "$map_mode" --np "$np" >> "$log" 2>&1; then
    status=PASS
  else
    status=FAIL
    if ((rc == 0)); then rc=1; fi
    failures=$((failures + 1))
    printf '\n--- P4 %s: last 55 lines ---\n' "$name" >&2
    tail -n 55 "$log" >&2
  fi
  completed=$((completed + 1))
  printf '%-26s %-11s %-8s %s\n' "$name" "$suite" "$status" "$log"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$suite" "$status" "$rc" "$log" >> "$results"
  if [[ "$name" == fabric_probe && "$status" == PASS ]]; then
    if [[ $(grep -Ec '\[VMM_P4\] event=fabricProbe rank=[01] supported=1 reason=ok' "$log") -eq 2 ]]; then
      fabric_supported=1
    fi
  fi
done

if ((fabric_requested)); then
  log="$outdir/fabric_timer.log"
  if ((fabric_supported == 0)); then
    printf '# P4 optional Fabric IPC not supported by both GPUs\n' > "$log"
    printf '%-26s %-11s %-8s %s\n' fabric_timer timer SKIP "$log"
    printf 'fabric_timer\ttimer\tSKIP\t0\t%s\n' "$log" >> "$results"
  else
    # Every optional Fabric run is a separate MPI launch. Individual-GPU
    # allocation support does not guarantee peer import permission, so an
    # actual failure is a FAIL (not an automatic SKIP).
    printf '# Fabric requested; individual allocation preflight PASS\n' > "$log"
    rc=0
    if TPETRA_EXPERIMENTAL_VMM=1 \
       TPETRA_EXPERIMENTAL_VMM_NATIVE_MV=1 \
       TPETRA_EXPERIMENTAL_VMM_STATISTICS=1 \
       TPETRA_EXPERIMENTAL_VMM_IPC=fabric \
       TPETRA_DEBUG=0 TPETRA_SYNCHRONIZE_APPLY=1 \
       VMM_TEST_MAP_MODE="$map_mode" \
       run_mpi "$log" --group=VMM_Timer --globally-reduce-test-result; then
      rc=0
    else
      rc=$?
    fi
    if ((rc == 0)) && python3 "$validator" --log "$log" --suite timer \
         --vmm 1 --native 1 --stats 1 --mode "$map_mode" --np "$np" \
         >> "$log" 2>&1; then
      status=PASS
    else
      status=FAIL
      if ((rc == 0)); then rc=1; fi
      failures=$((failures + 1))
      tail -n 55 "$log" >&2
    fi
    printf '%-26s %-11s %-8s %s\n' fabric_timer timer "$status" "$log"
    printf 'fabric_timer\ttimer\t%s\t%s\t%s\n' "$status" "$rc" "$log" >> "$results"
  fi
fi

printf '\nResults: %s\n' "$results"
if ((failures)); then
  printf 'P4 sweep FAILED: %s/%s mandatory jobs\n' "$failures" "$completed" >&2
  exit 1
fi
printf 'P4 sweep PASSED: %s mandatory jobs' "$completed"
if ((fabric_requested)) && ((fabric_supported == 0)); then
  printf '; optional Fabric SKIPPED (capability probe)'
fi
printf '\n'
