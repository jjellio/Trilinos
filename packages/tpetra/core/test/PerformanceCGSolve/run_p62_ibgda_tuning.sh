#!/usr/bin/env bash
# P6.2 IBGDA tuning: paired/bracketed batch and RC queue-pair sweeps.
# Run from the Trilinos build directory inside an existing 2-node allocation.
# This script changes only launch-time NVSHMEM tuning; P6.2 source is unchanged.
set -Eeuo pipefail
export LC_ALL=C
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Usage: run_p62_ibgda_tuning.sh [inter8|inter2|inter16|intra4|intra2] [size=100]
Run from the Trilinos build directory with an existing MPI allocation.

Main environment options:
  P62_SWEEP_POINTS='warp:512 block:1024 block:2048'  # variant:chunk pairs
  P62_TUNING_STAGES='batch rc'                 # optional third stage: combo
  P62_BATCH_VALUES='1 2 4 8 16 32'
  P62_RC_VALUES='1 2 3 4 5 6'
  P62_PASSES=3                                 # independent MPI launches per candidate
  P62_INCLUDE_OFF_CONTROL=1                    # one interleaved IBGDA-off control per block
  P62_COMBO_BATCHES='8 16' P62_COMBO_RCS='2 3'
  P62_SEED=20261025                            # deterministic candidate shuffle
  P62_DRY_RUN=1                                # plan only: no MPI, GPU or executable needed
  P62_OUT_DIR=./p62-ibgda-tuning-YYYYMMDD_HHMMSS
  P62_EXE=./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe
  P62_BIND=$HOME/hops/gpu_bind.sh               # set to your working GPU binder
  P62_MPIRUN=mpirun P62_MPI_EXTRA='...'
  P62_TIMEOUT_SECONDS=300                      # per MPI launch; 0 to disable
  P62_FAIL_FAST=1                              # stop at first failed launch
  P62_NIC_HANDLER=''                           # unset (auto) or explicitly cpu
  P62_REPS=50 P62_WARMUP=8 P62_VERIFY_EVERY=0
  P62_INFO_FIRST=1                             # NVSHMEM_INFO=1 on first IBGDA reference
  P62_POST_ANALYZE=1                           # run Python analyzer after sweep

The 'default' candidate means an UNSET variable, not a numerical approximation.
For the batch stage, RC is always unset. For the RC stage, batch is always unset.
The optional combo stage varies both parameters, after single-factor stages.
Every work unit gets an IBGDA-on reference before and after the candidates.

Default runs: 3 workloads x 3 passes x 2 stages x (6 candidates + 2 references
+ 1 IBGDA-off control) = 162 independent MPI launches.
USAGE
  exit 0
fi

TOPOLOGY="${1:-inter8}"
SIZE="${2:-100}"
[[ $# -le 2 ]] || { echo 'Expected at most topology and size arguments' >&2; exit 2; }
case "$TOPOLOGY" in
  intra2) PROCS=2; PPN=2; MIN_NODES=1; MAX_NODES=1; CROSS=0 ;;
  intra4) PROCS=4; PPN=4; MIN_NODES=1; MAX_NODES=1; CROSS=0 ;;
  inter2) PROCS=2; PPN=1; MIN_NODES=2; MAX_NODES=2; CROSS=1 ;;
  inter8) PROCS=8; PPN=4; MIN_NODES=2; MAX_NODES=2; CROSS=1 ;;
#  inter16) PROCS=16; PPN=8; MIN_NODES=2; MAX_NODES=2; CROSS=1 ;;
  *) echo "Unsupported topology: $TOPOLOGY" >&2; exit 2 ;;
esac
[[ "$SIZE" =~ ^[1-9][0-9]*$ ]] || { echo 'size must be positive' >&2; exit 2; }

EXE="${P62_EXE:-./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe}"
BIND="${P62_BIND:-$HOME/hops/gpu_bind.sh}"
MPI="${P62_MPIRUN:-mpirun}"
MPI_EXTRA="${P62_MPI_EXTRA:-}"
POINTS="${P62_SWEEP_POINTS:-warp:512 block:1024 block:2048}"
STAGES="${P62_TUNING_STAGES:-batch rc}"
BATCH_VALUES="${P62_BATCH_VALUES:-1 2 4 8 16 32}"
RC_VALUES="${P62_RC_VALUES:-1 2 3 4 5 6}"
COMBO_BATCHES="${P62_COMBO_BATCHES:-8 16}"
COMBO_RCS="${P62_COMBO_RCS:-2 3}"
PASSES="${P62_PASSES:-3}"
SEED="${P62_SEED:-20261025}"
OFF_CONTROL="${P62_INCLUDE_OFF_CONTROL:-1}"
DRY_RUN="${P62_DRY_RUN:-0}"
TIMEOUT_SECONDS="${P62_TIMEOUT_SECONDS:-300}"
FAIL_FAST="${P62_FAIL_FAST:-1}"
NIC_HANDLER="${P62_NIC_HANDLER:-}"
INFO_FIRST="${P62_INFO_FIRST:-1}"
POST_ANALYZE="${P62_POST_ANALYZE:-1}"
REPS="${P62_REPS:-50}"
WARMUP="${P62_WARMUP:-8}"
VERIFY_EVERY="${P62_VERIFY_EVERY:-0}"
OUT_DIR="${P62_OUT_DIR:-./p62-ibgda-tuning-$(date +%Y%m%d_%H%M%S)}"
CASE="${P62_CASE:-matrix}"
RECEIVER_BLOCKS="${P62_RECEIVER_BLOCKS:-4}"

for setting in PASSES SEED TIMEOUT_SECONDS REPS WARMUP VERIFY_EVERY RECEIVER_BLOCKS; do
  value="${!setting}"
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "$setting must be a nonnegative integer" >&2; exit 2; }
done
(( PASSES > 0 && REPS > 0 && RECEIVER_BLOCKS > 0 )) || { echo 'PASSES, REPS, RECEIVER_BLOCKS must be positive' >&2; exit 2; }
for setting in OFF_CONTROL DRY_RUN FAIL_FAST INFO_FIRST POST_ANALYZE; do
  [[ "${!setting}" == 0 || "${!setting}" == 1 ]] || { echo "$setting must be 0 or 1" >&2; exit 2; }
done
[[ -z "$NIC_HANDLER" || "$NIC_HANDLER" == auto || "$NIC_HANDLER" == cpu ]] || {
  echo 'P62_NIC_HANDLER may only be unset, auto or cpu (gpu is unavailable on this cluster)' >&2; exit 2;
}
read -r -a POINT_LIST <<<"$POINTS"
read -r -a STAGE_LIST <<<"$STAGES"
read -r -a BATCH_LIST <<<"$BATCH_VALUES"
read -r -a RC_LIST <<<"$RC_VALUES"
read -r -a COMBO_BATCH_LIST <<<"$COMBO_BATCHES"
read -r -a COMBO_RC_LIST <<<"$COMBO_RCS"
read -r -a EXTRA_MPI_ARGS <<<"$MPI_EXTRA"
(( ${#POINT_LIST[@]} > 0 && ${#STAGE_LIST[@]} > 0 )) || { echo 'Empty workload or stage list' >&2; exit 2; }
for workload in "${POINT_LIST[@]}"; do
  [[ "$workload" =~ ^(thread|warp|block):([1-9][0-9]*)$ ]] || {
    echo "Invalid P62_SWEEP_POINTS value: $workload" >&2; exit 2;
  }
done
for stage in "${STAGE_LIST[@]}"; do
  [[ "$stage" == batch || "$stage" == rc || "$stage" == combo ]] || {
    echo "Unknown stage $stage" >&2; exit 2;
  }
done
for arrname in BATCH_LIST RC_LIST COMBO_BATCH_LIST COMBO_RC_LIST; do
  declare -n values="$arrname"
  (( ${#values[@]} > 0 )) || { echo "Empty $arrname" >&2; exit 2; }
  for value in "${values[@]}"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid $arrname: $value" >&2; exit 2; }
  done
  unset -n values
done
if (( ! DRY_RUN )); then
  [[ -x "$EXE" ]] || { echo "Executable is not executable: $EXE" >&2; exit 2; }
  [[ -x "$BIND" ]] || { echo "GPU binding script is not executable: $BIND" >&2; exit 2; }
  command -v "$MPI" >/dev/null 2>&1 || { echo "Cannot find MPI launcher: $MPI" >&2; exit 2; }
  if (( TIMEOUT_SECONDS > 0 )); then
    command -v timeout >/dev/null || { echo 'GNU timeout is required' >&2; exit 2; }
  fi
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd -P)"
MANIFEST="$OUT_DIR/manifest.tsv"
[[ ! -e "$MANIFEST" ]] || { echo "Manifest already exists: $MANIFEST (use a fresh output dir)" >&2; exit 2; }
printf 'run_id\tstage\tpass\tvariant\tchunk\trole\tibgda\tbatch\trc\texit_code\tlog\n' > "$MANIFEST"
cat > "$OUT_DIR/config.env.txt" <<EOF
created=$(date --iso-8601=seconds)
host=$(hostname)
topology=$TOPOLOGY
np=$PROCS
ppn=$PPN
size=$SIZE
p62_case=$CASE
sweep_points=$POINTS
stages=$STAGES
batch_values=$BATCH_VALUES
rc_values=$RC_VALUES
combo_batches=$COMBO_BATCHES
combo_rcs=$COMBO_RCS
passes=$PASSES
seed=$SEED
ibgda_nic_handler_override=${NIC_HANDLER:-<unset>}
ibgda_off_control=$OFF_CONTROL
reps=$REPS
warmup=$WARMUP
verify_every=$VERIFY_EVERY
receiver_blocks=$RECEIVER_BLOCKS
executable=$EXE
binder=$BIND
mpi=$MPI
mpi_extra=$MPI_EXTRA
out_dir=$OUT_DIR
dry_run=$DRY_RUN
EOF
if (( ! DRY_RUN )); then
  { sha256sum "$EXE" "$BIND" || true; } > "$OUT_DIR/file_hashes.sha256"
  if command -v git >/dev/null 2>&1; then
    git rev-parse --verify HEAD > "$OUT_DIR/git_revision.txt" 2>/dev/null || true
  fi
  "$MPI" "${EXTRA_MPI_ARGS[@]}" --map-by "ppr:${PPN}:node" --bind-to none -np "$PROCS" /bin/hostname \
    > "$OUT_DIR/launch_hostnames.txt" 2>&1 || {
      echo 'MPI topology preflight failed; inspect launch_hostnames.txt' >&2; exit 2;
    }
fi

# Shuffle deterministically within each workload/pass, using Bash's local PRNG.
# The seed depends on stage/pass/work unit; references always bracket the permutation.
shuffle_candidates() {
  local stage="$1" pass="$2" variant="$3" chunk="$4"; shift 4
  local stage_code=0 variant_code=0 i j swap
  case "$stage" in batch) stage_code=17;; rc) stage_code=37;; combo) stage_code=53;; esac
  case "$variant" in thread) variant_code=1;; warp) variant_code=2;; block) variant_code=3;; esac
  RANDOM=$(( (SEED + stage_code * 691 + pass * 3509 + variant_code * 10007 + chunk * 13) % 32767 + 1 ))
  local -a arr=("$@")
  for ((i=${#arr[@]}-1; i>0; --i)); do
    j=$((RANDOM % (i + 1)))
    swap="${arr[i]}"; arr[i]="${arr[j]}"; arr[j]="$swap"
  done
  printf '%s\n' "${arr[@]}"
}

run_number=0
failures=0
info_done=0
run_one() {
  local stage="$1" pass="$2" variant="$3" chunk="$4" role="$5" batch="$6" rc="$7" ibgda="$8"
  run_number=$((run_number + 1))
  local run_id label prefix log env_file exit_code status
  printf -v run_id '%04d' "$run_number"
  label="run${run_id}_${stage}_p${pass}_${variant}_k${chunk}_${role}_b${batch}_rc${rc}_gda${ibgda}"
  prefix="$OUT_DIR/$label"
  log="${prefix}.log"
  env_file="${prefix}.env.txt"
  local -a cross_arg=() mpi_exports=() command=() timeout_args=()
  if (( CROSS )); then cross_arg+=(--nvshmem-p62-cross-node); fi
  mpi_exports=(-x NVSHMEM_REMOTE_TRANSPORT -x NVSHMEM_IB_ENABLE_IBGDA)
  if [[ "$batch" != default ]]; then mpi_exports+=(-x NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH); fi
  if [[ "$rc" != default ]]; then mpi_exports+=(-x NVSHMEM_IBGDA_NUM_RC_PER_PE); fi
  if [[ -n "$NIC_HANDLER" ]]; then mpi_exports+=(-x NVSHMEM_IBGDA_NIC_HANDLER); fi
  if (( INFO_FIRST && ! info_done && ibgda )); then
    mpi_exports+=(-x NVSHMEM_INFO)
    info_done=1
    local nvshmem_info=1
  else
    local nvshmem_info=0
  fi
  command=("$MPI" "${EXTRA_MPI_ARGS[@]}" --map-by "ppr:${PPN}:node" --bind-to none
           "${mpi_exports[@]}" -np "$PROCS" "$BIND" "$EXE"
           --cuda --numgpus=1 "--size=$SIZE" --nvshmem-p62
           "--nvshmem-p62-case=$CASE" --nvshmem-p62-mode=direct
           "--nvshmem-p62-put-variant=$variant" "--nvshmem-p62-chunk=$chunk"
           "--nvshmem-p62-receiver-blocks=$RECEIVER_BLOCKS"
           "--nvshmem-p62-warmup=$WARMUP" "--nvshmem-p62-reps=$REPS"
           "--nvshmem-p62-verify-every=$VERIFY_EVERY"
           "--nvshmem-p62-min-nodes=$MIN_NODES" "--nvshmem-p62-max-nodes=$MAX_NODES"
           "--nvshmem-p62-out=$prefix" "${cross_arg[@]}")
  {
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
    printf 'stage=%s\npass=%s\nvariant=%s\nchunk=%s\nrole=%s\n' "$stage" "$pass" "$variant" "$chunk" "$role"
    printf 'NVSHMEM_REMOTE_TRANSPORT=ibrc\nNVSHMEM_IB_ENABLE_IBGDA=%s\n' "$ibgda"
    printf 'NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH=%s\n' "${batch/default/<unset>}"
    printf 'NVSHMEM_IBGDA_NUM_RC_PER_PE=%s\n' "${rc/default/<unset>}"
    printf 'NVSHMEM_IBGDA_NIC_HANDLER=%s\n' "${NIC_HANDLER:-<unset>}"
    printf 'NVSHMEM_INFO=%s\n' "$nvshmem_info"
    printf 'command='; printf '%q ' "${command[@]}"; printf '\n'
  } > "$env_file"
  printf '[%04d] stage=%-5s pass=%s mode=%-6s chunk=%-5s %-16s IBGDA=%s batch=%-7s rc=%-7s\n' \
    "$run_number" "$stage" "$pass" "$variant" "$chunk" "$role" "$ibgda" "$batch" "$rc"
  if (( DRY_RUN )); then
    printf 'DRY RUN: '; printf '%q ' "${command[@]}"; printf '\n'
    exit_code=DRY_RUN
  else
    if (
      export NVSHMEM_REMOTE_TRANSPORT=ibrc
      export NVSHMEM_IB_ENABLE_IBGDA="$ibgda"
      unset NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH NVSHMEM_IBGDA_NUM_RC_PER_PE
      unset NVSHMEM_IBGDA_NIC_HANDLER NVSHMEM_INFO
      # RC QP placement and NIC handling must not drift across runs.
      unset NVSHMEM_IBGDA_RC_MAP_BY NVSHMEM_IBGDA_DCI_MAP_BY
      if [[ "$batch" != default ]]; then export NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH="$batch"; fi
      if [[ "$rc" != default ]]; then export NVSHMEM_IBGDA_NUM_RC_PER_PE="$rc"; fi
      if [[ -n "$NIC_HANDLER" ]]; then export NVSHMEM_IBGDA_NIC_HANDLER="$NIC_HANDLER"; fi
      if (( nvshmem_info )); then export NVSHMEM_INFO=1; fi
      if (( TIMEOUT_SECONDS > 0 )); then
        timeout --signal=TERM --kill-after=20s "${TIMEOUT_SECONDS}s" "${command[@]}"
      else
        "${command[@]}"
      fi
    ) > "$log" 2>&1; then
      exit_code=0
    else
      exit_code=$?
      failures=$((failures + 1))
    fi
    # Do not treat MPI exit=0 as correctness success: require the P6.2 PASS marker.
    if [[ "$exit_code" == 0 ]] && ! grep -q '\[NVSHMEM P6\.2\] SUMMARY status=PASS' "$log"; then
      exit_code=98
      failures=$((failures + 1))
    fi
    grep -E '\[NVSHMEM P6\.2 (DIRECT|BARRIER|SPEEDUP|CASE)\]|\[NVSHMEM P6\.2\] SUMMARY' "$log" || true
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_id" "$stage" "$pass" "$variant" "$chunk" "$role" "$ibgda" "$batch" "$rc" "$exit_code" "$log" >> "$MANIFEST"
  if [[ "$exit_code" != DRY_RUN && "$exit_code" != 0 ]]; then
    echo "FAILED: $label status=$exit_code (inspect $log)" >&2
    if (( FAIL_FAST )); then exit 1; fi
  fi
}

for stage in "${STAGE_LIST[@]}"; do
  candidate_tokens=()
  case "$stage" in
    batch) for b in "${BATCH_LIST[@]}"; do candidate_tokens+=("$b:default"); done ;;
    rc) for rc in "${RC_LIST[@]}"; do candidate_tokens+=("default:$rc"); done ;;
    combo) for b in "${COMBO_BATCH_LIST[@]}"; do
             for rc in "${COMBO_RC_LIST[@]}"; do candidate_tokens+=("$b:$rc"); done
           done ;;
  esac
  for ((pass=1; pass<=PASSES; pass++)); do
    # Reverse workload order on even passes, to reduce chronological bias.
    work_order=("${POINT_LIST[@]}")
    if (( pass % 2 == 0 )); then
      reversed=()
      for ((i=${#work_order[@]}-1; i>=0; --i)); do reversed+=("${work_order[i]}"); done
      work_order=("${reversed[@]}")
    fi
    for workload in "${work_order[@]}"; do
      variant="${workload%%:*}"
      chunk="${workload#*:}"
      run_one "$stage" "$pass" "$variant" "$chunk" reference_before default default 1
      # The IBGDA-off control is randomized *inside* the reference bracket too.
      block_candidates=("${candidate_tokens[@]}")
      if (( OFF_CONTROL )); then block_candidates+=("OFF"); fi
      mapfile -t ordered < <(shuffle_candidates "$stage" "$pass" "$variant" "$chunk" "${block_candidates[@]}")
      for token in "${ordered[@]}"; do
        if [[ "$token" == OFF ]]; then
          run_one "$stage" "$pass" "$variant" "$chunk" off_control default default 0
        else
          b="${token%%:*}"; rc="${token#*:}"
          run_one "$stage" "$pass" "$variant" "$chunk" candidate "$b" "$rc" 1
        fi
      done
      run_one "$stage" "$pass" "$variant" "$chunk" reference_after default default 1
    done
  done
done

printf '\nCompleted %s planned MPI launches, failures=%s; output: %s\n' "$run_number" "$failures" "$OUT_DIR"
if (( ! DRY_RUN && POST_ANALYZE )); then
  if [[ -f "$HERE/analyze_p62_ibgda_tuning.py" ]]; then
    python3 "$HERE/analyze_p62_ibgda_tuning.py" "$OUT_DIR"
  else
    echo 'Companion analyzer not found beside this script; raw logs and manifest are preserved.' >&2
  fi
fi
if (( failures > 0 )); then exit 1; fi
