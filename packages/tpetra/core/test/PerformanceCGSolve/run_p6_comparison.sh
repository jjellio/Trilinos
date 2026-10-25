#!/usr/bin/env bash
# P6 direct end-to-end comparison plus point-to-point instrumented phases.
# Intended to run INSIDE the corresponding GPU-node allocation. Does not
# schedule nodes or assume OpenMPI allocates them automatically.
set -euo pipefail
TOPOLOGY="${1:-inter16}"   # smoke1, intra2, intra4, inter2, inter16
SIZE="${2:-100}"         # miniFE size, e.g. 32/64/100
MODE="${3:-both}"        # both|direct|phases
EXE="${P6_EXE:-./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe}"
BIND="${P6_BIND:-$HOME/cronus/gpu_bind.sh}"
OUT_DIR="${P6_OUT_DIR:-./p6-results}"
MPI="${P6_MPIRUN:-mpirun}"
mkdir -p "$OUT_DIR"
case "$TOPOLOGY" in
  smoke1)  PROCS=1; PPN=1; NODE_MIN=1; NODE_MAX=1; REQUIRE_CROSS=0 ;;
  intra2)  PROCS=2; PPN=2; NODE_MIN=1; NODE_MAX=1; REQUIRE_CROSS=0 ;;
  intra4)  PROCS=4; PPN=4; NODE_MIN=1; NODE_MAX=1; REQUIRE_CROSS=0 ;;
  inter2)  PROCS=2; PPN=1; NODE_MIN=2; NODE_MAX=2; REQUIRE_CROSS=1 ;;
  inter8)  PROCS=8; PPN=4; NODE_MIN=2; NODE_MAX=2; REQUIRE_CROSS=1 ;;
  inter16) PROCS=16; PPN=8; NODE_MIN=2; NODE_MAX=2; REQUIRE_CROSS=1 ;;
  *) echo "Unknown topology: $TOPOLOGY" >&2; exit 2 ;;
esac
if [[ ! -x "$EXE" ]]; then echo "Executable not found: $EXE" >&2; exit 2; fi
if [[ ! -x "$BIND" ]]; then echo "Binding wrapper not executable: $BIND" >&2; exit 2; fi
if [[ "$MODE" != both && "$MODE" != direct && "$MODE" != phases ]]; then
  echo 'Mode must be both, direct, or phases' >&2; exit 2
fi
RUN_ID="${TOPOLOGY}_size${SIZE}_${MODE}_${P6_LABEL:-run}"
PREFIX="$OUT_DIR/$RUN_ID"
CROSS=()
if (( REQUIRE_CROSS )); then CROSS=(--nvshmem-p6-cross-node); fi
COMMAND=("$MPI" --map-by "ppr:${PPN}:node" --bind-to none -np "$PROCS"
         "$BIND" "$EXE" --cuda --numgpus=1 "--size=$SIZE"
         --nvshmem-p6 "--nvshmem-p6-case=${P6_CASE:-matrix}"
         "--nvshmem-p6-put-variant=${P6_PUT_VARIANT:-thread}"
         "--nvshmem-p6-mode=$MODE"
         "--nvshmem-p6-warmup=${P6_WARMUP:-20}"
         "--nvshmem-p6-reps=${P6_REPS:-100}"
         "--nvshmem-p6-phase-reps=${P6_PHASE_REPS:-100}"
         "--nvshmem-p6-min-nodes=$NODE_MIN" "--nvshmem-p6-max-nodes=$NODE_MAX"
         "--nvshmem-p6-out=$PREFIX" "${CROSS[@]}")
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'NVSHMEM_REMOTE_TRANSPORT=%s\n' "${NVSHMEM_REMOTE_TRANSPORT:-<unset>}"
  printf 'NVSHMEM_IB_ENABLE_IBGDA=%s\n' "${NVSHMEM_IB_ENABLE_IBGDA:-<unset>}"
  printf 'NVSHMEM_IBGDA_NIC_HANDLER=%s\n' "${NVSHMEM_IBGDA_NIC_HANDLER:-<unset>}"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
  printf 'command='; printf '%q ' "${COMMAND[@]}"; printf '\n'
} >"${PREFIX}.env.txt"
# tee preserves complete MPI stdout and startup transport warning evidence.
"${COMMAND[@]}" 2>&1 | tee "${PREFIX}.log"
