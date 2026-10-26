#!/usr/bin/env bash
# Controlled P6.2 sweep. Run from an allocation already containing the nodes.
# One MPI job per (chunk size, thread/warp/block) point; the two CUDA kernels
# execute only inside that job. Historical P6.1 drivers are in the pre-P7 Git tag.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPOLOGY="${1:-inter8}"  # smoke1, intra2, intra4, inter2, inter8, inter16
SIZE="${2:-100}"
MODE="${3:-direct}"  # direct|diagnostics|both
EXE="${P62_EXE:-./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe}"
BIND="${P62_BIND:-$HOME/hops/gpu_bind.sh}"
MPI="${P62_MPIRUN:-mpirun}"
OUT_DIR="${P62_OUT_DIR:-./p62-results}"
PASSES="${P62_PASSES:-1}"
CHUNKS="${P62_CHUNKS:-1 2 4 8 16 32 64 128 256 512 1024 2048}"
VARIANTS="${P62_VARIANTS:-thread warp block}"
case "$TOPOLOGY" in
  smoke1) PROCS=1; PPN=1; NODE_MIN=1; NODE_MAX=1; REQUIRE_CROSS=0 ;;
  intra2) PROCS=2; PPN=2; NODE_MIN=1; NODE_MAX=1; REQUIRE_CROSS=0 ;;
  intra4) PROCS=4; PPN=4; NODE_MIN=1; NODE_MAX=1; REQUIRE_CROSS=0 ;;
  inter2) PROCS=2; PPN=1; NODE_MIN=2; NODE_MAX=2; REQUIRE_CROSS=1 ;;
  inter4) PROCS=4; PPN=1; NODE_MIN=2; NODE_MAX=2; REQUIRE_CROSS=1 ;;
  #inter8) PROCS=8; PPN=4; NODE_MIN=2; NODE_MAX=2; REQUIRE_CROSS=1 ;;
  #inter16) PROCS=16; PPN=8; NODE_MIN=2; NODE_MAX=2; REQUIRE_CROSS=1 ;;
  *) echo "Unknown topology: $TOPOLOGY" >&2; exit 2 ;;
esac
[[ -x "$EXE" ]] || { echo "Executable is missing or not executable: $EXE" >&2; exit 2; }
[[ -x "$BIND" ]] || { echo "GPU binding wrapper is not executable: $BIND" >&2; exit 2; }
[[ "$MODE" == direct || "$MODE" == diagnostics || "$MODE" == both ]] || {
  echo "Mode must be direct, diagnostics or both" >&2; exit 2;
}
[[ "$PASSES" =~ ^[1-9][0-9]*$ ]] || { echo 'P62_PASSES must be positive' >&2; exit 2; }
read -r -a chunks <<<"$CHUNKS"
read -r -a variants <<<"$VARIANTS"
((${#chunks[@]} > 0 && ${#variants[@]} > 0)) || { echo 'Empty chunk/variant sweep' >&2; exit 2; }
for c in "${chunks[@]}"; do
  [[ "$c" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid chunk: $c" >&2; exit 2; }
done
for v in "${variants[@]}"; do
  [[ "$v" == thread || "$v" == warp || "$v" == block ]] || {
    echo "Unknown put variant: $v" >&2; exit 2;
  }
done
mkdir -p "$OUT_DIR"
CROSS=()
if (( REQUIRE_CROSS )); then CROSS=(--nvshmem-p62-cross-node); fi
for ((pass=1; pass<=PASSES; pass++)); do
  order=("${variants[@]}")
  if ((pass % 2 == 0)); then
    reversed=()
    for ((i=${#order[@]}-1; i>=0; --i)); do reversed+=("${order[$i]}"); done
    order=("${reversed[@]}")
  fi
  for variant in "${order[@]}"; do
    for chunk in "${chunks[@]}"; do
      label="${P62_LABEL:-p62}_${TOPOLOGY}_s${SIZE}_${MODE}_pass${pass}_${variant}_k${chunk}"
      prefix="$OUT_DIR/$label"
      command=("$MPI" --map-by "ppr:${PPN}:node" --bind-to none -np "$PROCS"
               "$BIND" "$EXE" --cuda --numgpus=1 "--size=$SIZE"
               --nvshmem-p62 "--nvshmem-p62-case=${P62_CASE:-matrix}"
               "--nvshmem-p62-mode=$MODE"
               "--nvshmem-p62-put-variant=$variant"
               "--nvshmem-p62-chunk=$chunk"
               "--nvshmem-p62-receiver-blocks=${P62_RECEIVER_BLOCKS:-4}"
               "--nvshmem-p62-warmup=${P62_WARMUP:-8}"
               "--nvshmem-p62-reps=${P62_REPS:-50}"
               "--nvshmem-p62-diagnostic-reps=${P62_DIAGNOSTIC_REPS:-50}"
               "--nvshmem-p62-verify-every=${P62_VERIFY_EVERY:-0}"
               "--nvshmem-p62-min-nodes=$NODE_MIN"
               "--nvshmem-p62-max-nodes=$NODE_MAX"
               "--nvshmem-p62-out=$prefix" "${CROSS[@]}")
      {
        printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
        printf 'hostname=%s\n' "$(hostname)"
        printf 'NVSHMEM_REMOTE_TRANSPORT=%s\n' "${NVSHMEM_REMOTE_TRANSPORT:-<unset>}"
        printf 'NVSHMEM_IB_ENABLE_IBGDA=%s\n' "${NVSHMEM_IB_ENABLE_IBGDA:-<unset>}"
        printf 'NVSHMEM_IBGDA_NIC_HANDLER=%s\n' "${NVSHMEM_IBGDA_NIC_HANDLER:-<unset>}"
        printf 'command='; printf '%q ' "${command[@]}"; printf '\n'
      } >"${prefix}.env.txt"
      printf 'P6.2 pass=%s/%s variant=%s chunk=%s topology=%s size=%s\n' \
        "$pass" "$PASSES" "$variant" "$chunk" "$TOPOLOGY" "$SIZE"
      # A GPU wait that hangs should be interrupted with a job timeout by
      # the launcher/site scheduler. This script intentionally adds no MPI
      # progress collectives between P6.2 descriptor transfers.
      "${command[@]}" 2>&1 | tee "${prefix}.log"
    done
  done
done
