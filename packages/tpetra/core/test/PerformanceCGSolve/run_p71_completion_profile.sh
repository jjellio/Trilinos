#!/usr/bin/env bash
# P7.1 repeatable instrumentation of the FROZEN P6.2 PUT_SIGNAL algorithm.
# Each MPI job includes an uninstrumented Tpetra/P6.2 comparison THEN a
# separate instrumented diagnostics phase; no NBI, quiet, fence or ACK.
set -Eeuo pipefail
export LC_ALL=C
if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then
  cat <<'HELP'
Usage: run_p71_completion_profile.sh [inter2|inter8|intra2|intra4] [size=100]
Run in a two-node Slurm allocation, FROM THE TRILINOS BUILD DIRECTORY.

Environment controls:
 P71_EXE=./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe
 P71_BIND=$HOME/cronus/gpu_bind.sh
 P71_MPI=mpirun
 P71_OUT_DIR=./p71-profiles-YYYYMMDD_HHMMSS
 P71_PASSES=3
 P71_VARIANTS='warp block'
 P71_CHUNKS='512 1024 2048'
 P71_TRANSPORTS='ibrc ibgda-cpu'    # ibgda-cpu explicitly selects CPU handler
 P71_CASE=matrix
 P71_WARMUP=5 P71_REPS=30 P71_PROFILE_REPS=12
 P71_SAMPLE_CAP=64 P71_RECEIVER_BLOCKS=4 P71_VERIFY_EVERY=1
 P71_TIMEOUT_SECONDS=300            # abort GPU hangs; requires GNU timeout
 P71_DRY_RUN=1                      # generate plan without MPI or GPU
 P71_FAIL_FAST=1                    # fail on first bad job
 P71_VERBOSE_TRANSPORT=0            # set 1 to inspect NVSHMEM handler (more logs)
 P71_MPI_EXTRA='...'                # additional mpirun args (whitespace split)

Run inter2 then inter8 separately; keep ibrc and ibgda-cpu results separate.
Use nsys outside this script for GPU overlap and NIC transport tracing.
HELP
  exit 0
fi
TOPOLOGY="${1:-inter8}"; SIZE="${2:-100}"
[[ $# -le 2 && "$SIZE" =~ ^[1-9][0-9]*$ ]] || { echo 'Expected topology and positive size' >&2; exit 2; }
case "$TOPOLOGY" in
  inter2) NP=2; PPN=1; MIN_NODES=2; MAX_NODES=2; CROSS=1 ;;
  inter8) NP=8; PPN=4; MIN_NODES=2; MAX_NODES=2; CROSS=1 ;;
  intra2) NP=2; PPN=2; MIN_NODES=1; MAX_NODES=1; CROSS=0 ;;
  intra4) NP=4; PPN=4; MIN_NODES=1; MAX_NODES=1; CROSS=0 ;;
  *) echo "Unsupported topology $TOPOLOGY" >&2; exit 2 ;;
esac
EXE="${P71_EXE:-./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe}"
BIND="${P71_BIND:-$HOME/cronus/gpu_bind.sh}"
MPI="${P71_MPI:-mpirun}"
PASSES="${P71_PASSES:-3}"
VARIANTS="${P71_VARIANTS:-warp block}"
CHUNKS="${P71_CHUNKS:-512 1024 2048}"
TRANSPORTS="${P71_TRANSPORTS:-ibrc ibgda-cpu}"
CASE="${P71_CASE:-matrix}"
[[ "$CASE" == matrix ]] || {
  echo 'run_p71_completion_profile.sh summarizes the matrix case only; use the executable directly for --case=all correctness' >&2; exit 2;
}
WARMUP="${P71_WARMUP:-5}"
REPS="${P71_REPS:-30}"
PROFILE_REPS="${P71_PROFILE_REPS:-12}"
SAMPLE_CAP="${P71_SAMPLE_CAP:-64}"
RECEIVER_BLOCKS="${P71_RECEIVER_BLOCKS:-4}"
VERIFY_EVERY="${P71_VERIFY_EVERY:-1}"
TIMEOUT_SECONDS="${P71_TIMEOUT_SECONDS:-300}"
DRY_RUN="${P71_DRY_RUN:-0}"
FAIL_FAST="${P71_FAIL_FAST:-1}"
VERBOSE_TRANSPORT="${P71_VERBOSE_TRANSPORT:-0}"
OUT_DIR="${P71_OUT_DIR:-./p71-profiles-$(date +%Y%m%d_%H%M%S)}"
read -r -a variant_list <<< "$VARIANTS"
read -r -a chunk_list <<< "$CHUNKS"
read -r -a transport_list <<< "$TRANSPORTS"
read -r -a extra_mpi <<< "${P71_MPI_EXTRA:-}"
(( ${#variant_list[@]} && ${#chunk_list[@]} && ${#transport_list[@]} )) || { echo 'Empty experiment space' >&2; exit 2; }
for v in "${variant_list[@]}"; do
  [[ "$v" == thread || "$v" == warp || "$v" == block ]] || { echo "Invalid variant $v" >&2; exit 2; }
done
for t in "${transport_list[@]}"; do
  [[ "$t" == ibrc || "$t" == ibgda-cpu ]] || { echo "Invalid transport $t" >&2; exit 2; }
done
for k in "${chunk_list[@]}"; do
  [[ "$k" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid chunk $k" >&2; exit 2; }
done
for var in PASSES WARMUP REPS PROFILE_REPS SAMPLE_CAP RECEIVER_BLOCKS TIMEOUT_SECONDS; do
  [[ "${!var}" =~ ^[0-9]+$ ]] || { echo "$var must be an integer" >&2; exit 2; }
done
(( PASSES >= 1 && WARMUP >= 2 && REPS >= 5 && PROFILE_REPS >= 2 && SAMPLE_CAP >= 1 && SAMPLE_CAP <= 4096 && RECEIVER_BLOCKS >= 1 && RECEIVER_BLOCKS <= 16 )) || {
  echo 'Invalid iteration/sample/receiver counts' >&2; exit 2;
}
for var in DRY_RUN FAIL_FAST VERBOSE_TRANSPORT VERIFY_EVERY; do
  [[ "${!var}" == 0 || "${!var}" == 1 ]] || { echo "$var must be 0 or 1" >&2; exit 2; }
done
if (( ! DRY_RUN )); then
  [[ -x "$EXE" && -x "$BIND" ]] || { echo "Executable/binder missing: $EXE / $BIND" >&2; exit 2; }
  command -v "$MPI" >/dev/null 2>&1 || { echo "MPI launcher missing: $MPI" >&2; exit 2; }
  if (( TIMEOUT_SECONDS > 0 )); then command -v timeout >/dev/null || exit 2; fi
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd -P)"
MANIFEST="$OUT_DIR/manifest.tsv"
[[ ! -e "$MANIFEST" ]] || { echo "Output directory already has manifest.tsv: $OUT_DIR" >&2; exit 2; }
printf 'run_id\tpass\tvariant\tchunk\ttransport\ttopology\tstatus\tlog\toutput_prefix\n' > "$MANIFEST"
{
  printf 'date=%s\nhost=%s\n' "$(date --iso-8601=seconds)" "$(hostname)"
  printf 'topology=%s\nsize=%s\nnp=%s\nppn=%s\n' "$TOPOLOGY" "$SIZE" "$NP" "$PPN"
  printf 'variants=%s\nchunks=%s\ntransports=%s\n' "$VARIANTS" "$CHUNKS" "$TRANSPORTS"
  printf 'reps=%s\nprofile_reps=%s\nmax_samples=%s\n' "$REPS" "$PROFILE_REPS" "$SAMPLE_CAP"
  printf 'receiver_blocks=%s\ncase=%s\nexe=%s\nbinder=%s\n' "$RECEIVER_BLOCKS" "$CASE" "$EXE" "$BIND"
} > "$OUT_DIR/config.txt"
if (( ! DRY_RUN )); then
  sha256sum "$EXE" "$BIND" > "$OUT_DIR/file_hashes.sha256"
  "$MPI" "${extra_mpi[@]}" --map-by "ppr:${PPN}:node" --bind-to none -np "$NP" hostname > "$OUT_DIR/hostnames.txt"
  actual_nodes="$(sort -u "$OUT_DIR/hostnames.txt" | wc -l)"
  [[ "$actual_nodes" -ge "$MIN_NODES" && "$actual_nodes" -le "$MAX_NODES" ]] || {
    echo "Expected $MIN_NODES-$MAX_NODES nodes, got $actual_nodes" >&2; exit 2;
  }
fi
failures=0; ordinal=0
for ((pass=1; pass<=PASSES; ++pass)); do
  tasks=()
  for transport in "${transport_list[@]}"; do
    for variant in "${variant_list[@]}"; do
      for chunk in "${chunk_list[@]}"; do
        tasks+=("$variant:$chunk:$transport")
      done
    done
  done
  if (( pass % 2 == 0 )); then
    reverse=()
    for ((i=${#tasks[@]}-1; i>=0; --i)); do reverse+=("${tasks[i]}"); done
    tasks=("${reverse[@]}")
  fi
  for task in "${tasks[@]}"; do
    IFS=: read -r variant chunk transport <<< "$task"
    ordinal=$((ordinal + 1)); printf -v rid '%04d' "$ordinal"
    stem="p71_${TOPOLOGY}_s${SIZE}_pass${pass}_${transport}_${variant}_k${chunk}"
    prefix="$OUT_DIR/$stem"
    log="$prefix.log"
    if [[ "$transport" == 'ibgda-cpu' ]]; then
      enable_ibgda=1; nic_handler=cpu
    else
      enable_ibgda=0; nic_handler=''
    fi
    command=("$MPI" "${extra_mpi[@]}" --map-by "ppr:${PPN}:node" --bind-to none
             -np "$NP" -x NVSHMEM_REMOTE_TRANSPORT -x NVSHMEM_IB_ENABLE_IBGDA
             -x NVSHMEM_DEBUG)
    if [[ -n "$nic_handler" ]]; then command+=(-x NVSHMEM_IBGDA_NIC_HANDLER); fi
    command+=("$BIND" "$EXE" --cuda --numgpus=1 "--size=$SIZE" --nvshmem-p62
             --nvshmem-p62-mode=p71 "--nvshmem-p62-case=$CASE"
             "--nvshmem-p62-put-variant=$variant" "--nvshmem-p62-chunk=$chunk"
             "--nvshmem-p62-receiver-blocks=$RECEIVER_BLOCKS"
             "--nvshmem-p62-warmup=$WARMUP" "--nvshmem-p62-reps=$REPS"
             "--nvshmem-p62-p71-reps=$PROFILE_REPS"
             "--nvshmem-p62-p71-max-samples=$SAMPLE_CAP"
             "--nvshmem-p62-verify-every=$VERIFY_EVERY"
             "--nvshmem-p62-min-nodes=$MIN_NODES" "--nvshmem-p62-max-nodes=$MAX_NODES"
             "--nvshmem-p62-out=$prefix")
    if (( CROSS )); then command+=(--nvshmem-p62-cross-node); fi
    {
      printf 'timestamp=%s\ntransport=%s\n' "$(date --iso-8601=seconds)" "$transport"
      printf 'NVSHMEM_REMOTE_TRANSPORT=ibrc\nNVSHMEM_IB_ENABLE_IBGDA=%s\n' "$enable_ibgda"
      printf 'NVSHMEM_IBGDA_NIC_HANDLER=%s\n' "${nic_handler:-<unset>}"
      printf 'NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH=<unset>\nNVSHMEM_IBGDA_NUM_RC_PER_PE=<unset>\n'
      printf 'command='; printf '%q ' "${command[@]}"; printf '\n'
    } > "$prefix.env.txt"
    echo "P7.1 [$rid] $transport $variant k=$chunk pass=$pass/$PASSES"
    if (( DRY_RUN )); then
      printf '  DRY RUN: '; printf '%q ' "${command[@]}"; printf '\n'
      status=DRY_RUN
    else
      if (
        export NVSHMEM_REMOTE_TRANSPORT=ibrc NVSHMEM_IB_ENABLE_IBGDA="$enable_ibgda"
        export NVSHMEM_DEBUG=$([[ "$VERBOSE_TRANSPORT" == 1 ]] && echo INFO || echo WARN)
        unset NVSHMEM_IBGDA_NIC_HANDLER NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH NVSHMEM_IBGDA_NUM_RC_PER_PE
        if [[ -n "$nic_handler" ]]; then export NVSHMEM_IBGDA_NIC_HANDLER="$nic_handler"; fi
        if (( TIMEOUT_SECONDS )); then
          timeout --signal=TERM --kill-after=20 "${TIMEOUT_SECONDS}s" "${command[@]}"
        else
          "${command[@]}"
        fi
      ) > "$log" 2>&1; then status=0; else status=$?; fi
      if [[ "$status" == 0 ]] && ! grep -q '\[NVSHMEM P6.2\] SUMMARY status=PASS' "$log"; then status=98; fi
      if [[ "$status" == 0 ]] && ! grep -q '\[NVSHMEM P7.1 PROFILE\].*status=PASS' "$log"; then status=97; fi
      if [[ "$status" != 0 ]]; then
        failures=$((failures + 1))
        echo "P7.1 failure code=$status: $log" >&2
      fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$rid" "$pass" "$variant" "$chunk" "$transport" "$TOPOLOGY" "$status" "$log" "$prefix" >> "$MANIFEST"
    if [[ "$status" != 0 && "$status" != DRY_RUN ]] && (( FAIL_FAST )); then exit 1; fi
  done
done
printf 'P7.1: planned=%d failures=%d output=%s\n' "$ordinal" "$failures" "$OUT_DIR"
if (( ! DRY_RUN )); then
  python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/analyze_p71_profiles.py" "$OUT_DIR"
fi
(( failures == 0 ))
