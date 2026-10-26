#!/usr/bin/env bash
# P7.1b matrix-free Tpetra Import benchmark with controlled host-level degrees.
# Run from the Trilinos BUILD directory, within the preallocated physical nodes.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Usage: run_import_topology_sweep.sh [inter2|inter3|inter4|inter8|inter12|inter16|intra4] [per-peer|fixed-total]
Run from a Trilinos build directory, inside a matching MPI/Slurm allocation.

P71B_EXE: ./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_ImportTopologyBenchmark.exe
P71B_BIND: $HOME/cronus/gpu_bind.sh  (override with working GPU binder)
P71B_OUT_DIR: ./topology-results-TIMESTAMP
P71B_DRY_RUN=1 plans without MPI/GPU
P71B_DEGREES='1 2 4'  # number of OFF-node peers (invalid topology degrees rejected)
P71B_INTRA_DEGREE=0   # for intra4 use P71B_DEGREES='0' and P71B_INTRA_DEGREE=1..3
P71B_PEER_BYTES='1024 8192 65536' # per-peer mode (multiples of 8)
P71B_TOTAL_BYTES='65536'           # fixed-total mode (per PE)
P71B_OWNED=65536       # doubles per source Map on every PE
P71B_SOURCE_PATTERNS='contiguous'   # or 'contiguous strided permuted'
P71B_SOURCE_STRIDE=17  # coprime with P71B_OWNED if using strided
P71B_TARGET_LAYOUT=remote-only | prefix | interleave | shuffle
P71B_LOCAL_VALUES=0
P71B_VARIANTS='block'  # 'warp block', etc.
P71B_CHUNKS='1024'     # doubles per NVSHMEM PUT descriptor
P71B_TRANSPORTS='ibrc ibgda-cpu'
P71B_MODE=direct|p71|diagnostics|both
P71B_PASSES=3 P71B_WARMUP=8 P71B_REPS=50 P71B_VERIFY_EVERY=0
P71B_P71_REPS=12 P71B_P71_SAMPLES=64
P71B_TIMEOUT_SECONDS=300 P71B_FAIL_FAST=1
P71B_MPI_EXTRA='...'  # extra mpirun options (simple whitespace splitting)

Each independent MPI launch produces rank CSVs and a log; rank0 also writes
an NxN traffic matrix CSV, rank topology CSV and matched Tpetra/P6.2 timings.
Source remains normal Tpetra CUDA allocation; NVSHMEM uses symmetric scratch.
HELP
  exit 0
fi
TOPOLOGY="${1:-inter2}"
VOLUME="${2:-per-peer}"
[[ "$VOLUME" == per-peer || "$VOLUME" == fixed-total ]] || {
  echo 'Volume must be per-peer or fixed-total' >&2; exit 2;
}
case "$TOPOLOGY" in
  inter2) NP=2; PPN=1; NODES=2;;
  inter3) NP=3; PPN=1; NODES=3;;
  inter4) NP=4; PPN=1; NODES=4;;
  inter8) NP=8; PPN=4; NODES=2;;
  inter12) NP=12; PPN=4; NODES=3;;
  inter16) NP=16; PPN=4; NODES=4;;
  intra4) NP=4; PPN=4; NODES=1;;
  *) echo "Unsupported topology $TOPOLOGY" >&2; exit 2;;
esac
EXE="${P71B_EXE:-./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_ImportTopologyBenchmark.exe}"
BIND="${P71B_BIND:-$HOME/cronus/gpu_bind.sh}"
MPIRUN="${P71B_MPIRUN:-mpirun}"
OUT="${P71B_OUT_DIR:-./topology-results-$(date +%Y%m%d_%H%M%S)}"
DRY="${P71B_DRY_RUN:-0}"
PASSES="${P71B_PASSES:-3}"
OWNED="${P71B_OWNED:-65536}"
INTRA="${P71B_INTRA_DEGREE:-0}"
if [[ -n "${P71B_DEGREES:-}" ]]; then
  DEGREES="$P71B_DEGREES"
else
  case "$TOPOLOGY" in
    inter2) DEGREES='1';;
    inter3) DEGREES='1 2';;
    inter4) DEGREES='1 2 3';;
    inter8) DEGREES='1 2 4';;
    inter12) DEGREES='1 4 8';;
    inter16) DEGREES='1 4 8 12';;
    intra4) DEGREES='0'; INTRA="${P71B_INTRA_DEGREE:-1}";;
  esac
fi
PEER_BYTES="${P71B_PEER_BYTES:-1024 8192 65536}"
TOTAL_BYTES="${P71B_TOTAL_BYTES:-65536}"
PATTERNS="${P71B_SOURCE_PATTERNS:-contiguous}"
STRIDE="${P71B_SOURCE_STRIDE:-17}"
LAYOUT="${P71B_TARGET_LAYOUT:-remote-only}"
LOCAL_VALUES="${P71B_LOCAL_VALUES:-0}"
VARIANTS="${P71B_VARIANTS:-block}"
CHUNKS="${P71B_CHUNKS:-1024}"
TRANSPORTS="${P71B_TRANSPORTS:-ibrc ibgda-cpu}"
MODE="${P71B_MODE:-direct}"
REPS="${P71B_REPS:-50}"
WARMUP="${P71B_WARMUP:-8}"
VERIFY="${P71B_VERIFY_EVERY:-0}"
PROFILE_REPS="${P71B_P71_REPS:-12}"
PROFILE_SAMPLES="${P71B_P71_SAMPLES:-64}"
TIMEOUT_SECONDS="${P71B_TIMEOUT_SECONDS:-300}"
FAIL_FAST="${P71B_FAIL_FAST:-1}"
MPI_EXTRA="${P71B_MPI_EXTRA:-}"
RECEIVER_BLOCKS="${P71B_RECEIVER_BLOCKS:-4}"
SEED="${P71B_SEED:-12345}"
for key in DRY PASSES OWNED INTRA STRIDE LOCAL_VALUES REPS WARMUP VERIFY PROFILE_REPS PROFILE_SAMPLES TIMEOUT_SECONDS FAIL_FAST RECEIVER_BLOCKS SEED; do
  [[ "${!key}" =~ ^[0-9]+$ ]] || { echo "$key must be nonnegative integer" >&2; exit 2; }
done
(( PASSES >= 1 && OWNED >= 1 && REPS >= 5 && WARMUP >= 2 )) || {
  echo 'PASSES/OWNED/REPS/WARMUP outside supported minimums' >&2; exit 2;
}
[[ "$DRY" == 0 || "$DRY" == 1 ]] && [[ "$FAIL_FAST" == 0 || "$FAIL_FAST" == 1 ]] || exit 2
[[ "$MODE" == direct || "$MODE" == p71 || "$MODE" == diagnostics || "$MODE" == both ]] || exit 2
[[ "$LAYOUT" == remote-only || "$LAYOUT" == prefix || "$LAYOUT" == interleave || "$LAYOUT" == shuffle ]] || exit 2
read -r -a degree_list <<< "$DEGREES"
read -r -a peer_bytes_list <<< "$PEER_BYTES"
read -r -a total_bytes_list <<< "$TOTAL_BYTES"
read -r -a pattern_list <<< "$PATTERNS"
read -r -a variant_list <<< "$VARIANTS"
read -r -a chunk_list <<< "$CHUNKS"
read -r -a transport_list <<< "$TRANSPORTS"
read -r -a extra_mpi_args <<< "$MPI_EXTRA"
(( ${#degree_list[@]} && ${#variant_list[@]} && ${#pattern_list[@]} &&
   ${#chunk_list[@]} && ${#transport_list[@]} )) || { echo 'Empty sweep dimension' >&2; exit 2; }
if [[ "$VOLUME" == per-peer ]]; then
  volumes=("${peer_bytes_list[@]}")
else
  volumes=("${total_bytes_list[@]}")
fi
(( ${#volumes[@]} )) || { echo 'Empty volume sweep' >&2; exit 2; }
for value in "${volumes[@]}"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] && (( value % 8 == 0 )) || {
    echo "Bytes must be positive multiples of 8: $value" >&2; exit 2;
  }
done
for value in "${degree_list[@]}"; do
  [[ "$value" =~ ^[0-9]+$ ]] && (( value <= (NODES-1)*PPN )) || {
    echo "Off-node degree $value unavailable for $TOPOLOGY (max $(((NODES-1)*PPN)))" >&2; exit 2;
  }
done
(( INTRA <= PPN - 1 )) || { echo "Intra degree cannot exceed $((PPN-1))" >&2; exit 2; }
for value in "${variant_list[@]}"; do
  [[ "$value" == thread || "$value" == warp || "$value" == block ]] || exit 2
done
for value in "${pattern_list[@]}"; do
  [[ "$value" == contiguous || "$value" == strided || "$value" == permuted ]] || exit 2
done
for value in "${transport_list[@]}"; do
  [[ "$value" == ibrc || "$value" == ibgda-cpu ]] || exit 2
done
for value in "${chunk_list[@]}"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || exit 2
done
if (( ! DRY )); then
  [[ -x "$EXE" && -x "$BIND" ]] || { echo "Executable or GPU wrapper not found: $EXE $BIND" >&2; exit 2; }
  command -v "$MPIRUN" >/dev/null || exit 2
  if (( TIMEOUT_SECONDS > 0 )); then command -v timeout >/dev/null || exit 2; fi
fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd -P)"
MANIFEST="$OUT/manifest.tsv"
[[ ! -e "$MANIFEST" ]] || { echo "Existing manifest $MANIFEST; choose a new output directory" >&2; exit 2; }
printf 'run_id\tpass\ttopology\tvolume\tdegree\tintra_degree\tpayload_bytes\tsource\ttarget\tvariant\tchunk\ttransport\tmode\tstatus\tlog\toutput_prefix\n' > "$MANIFEST"
{
  printf 'created=%s\nnp=%s\nppn=%s\nnodes=%s\n' "$(date --iso-8601=seconds)" "$NP" "$PPN" "$NODES"
  printf 'volume=%s\nowned=%s\nsource_patterns=%s\ntarget_layout=%s\n' "$VOLUME" "$OWNED" "$PATTERNS" "$LAYOUT"
  printf 'degree_list=%s\nintra_degree=%s\nbytes=%s\n' "$DEGREES" "$INTRA" "${volumes[*]}"
  printf 'variants=%s\nchunks=%s\ntransports=%s\npasses=%s\n' "$VARIANTS" "$CHUNKS" "$TRANSPORTS" "$PASSES"
  printf 'matrix_constructed=NO\nsource_storage=ordinary_Tpetra_CudaSpace\n'
} > "$OUT/config.txt"
if (( ! DRY )); then
  sha256sum "$EXE" "$BIND" > "$OUT/build_files.sha256" || true
  "$MPIRUN" "${extra_mpi_args[@]}" --map-by "ppr:${PPN}:node" --bind-to none -np "$NP" hostname > "$OUT/hostnames.txt"
  actualnodes=$(sort -u "$OUT/hostnames.txt" | wc -l)
  [[ "$actualnodes" -eq "$NODES" ]] || { echo "Expected $NODES hosts, got $actualnodes" >&2; exit 2; }
fi
run=0; failures=0
for ((pass=1; pass<=PASSES; pass++)); do
  for degree in "${degree_list[@]}"; do
    (( degree + INTRA > 0 || LOCAL_VALUES > 0 )) || {
      echo 'Zero total degree requires positive localValues' >&2; exit 2;
    }
    for bytes in "${volumes[@]}"; do
      values=$((bytes / 8))
      # Fixed total modes partition values over incoming layers; require one value per layer.
      if [[ "$VOLUME" == fixed-total ]] && (( values < degree + INTRA || degree + INTRA == 0 )); then
        echo 'Fixed-total bytes insufficient or zero remote peer degree' >&2; exit 2;
      fi
      count_per_edge=$values
      if [[ "$VOLUME" == fixed-total ]]; then
        count_per_edge=$(( (values + degree + INTRA - 1) / (degree + INTRA) ))
      fi
      if (( count_per_edge > OWNED )); then
        echo "Require owned >= maximum values per peer ($OWNED < $count_per_edge)" >&2; exit 2;
      fi
      for source in "${pattern_list[@]}"; do
        for variant in "${variant_list[@]}"; do
          for chunk in "${chunk_list[@]}"; do
            # Reverse transport order each pass (not a latency estimate by itself).
            transports=("${transport_list[@]}")
            if (( pass % 2 == 0 )); then
              reversed=()
              for ((i=${#transports[@]}-1;i>=0;--i)); do reversed+=("${transports[i]}"); done
              transports=("${reversed[@]}")
            fi
            for transport in "${transports[@]}"; do
              run=$((run+1)); printf -v id '%04d' "$run"
              label="topo_${id}_${TOPOLOGY}_${VOLUME}_d${degree}_b${bytes}_${source}_${variant}_k${chunk}_${transport}_p${pass}"
              prefix="$OUT/$label"
              log="$prefix.log"
              envtxt="$prefix.env.txt"
              cross=()
              if (( degree > 0 )); then cross=(--require-cross-node); fi
              cmd=("$MPIRUN" "${extra_mpi_args[@]}" --map-by "ppr:${PPN}:node" --bind-to none
                   -x NVSHMEM_REMOTE_TRANSPORT -x NVSHMEM_IB_ENABLE_IBGDA
                   -np "$NP" "$BIND" "$EXE"
                   --numgpus=1 --owned="$OWNED" --expected-ppn="$PPN"
                   --offnode-degree="$degree" --intranode-degree="$INTRA"
                   --volume="$VOLUME" --values-per-peer="$values"
                   --remote-values-total="$values" --local-values="$LOCAL_VALUES"
                   --source-pattern="$source" --source-stride="$STRIDE"
                   --target-layout="$LAYOUT" --seed="$SEED"
                   --put-variant="$variant" --chunk="$chunk"
                   --receiver-blocks="$RECEIVER_BLOCKS" --mode="$MODE"
                   --warmup="$WARMUP" --reps="$REPS"
                   --p71-reps="$PROFILE_REPS" --p71-max-samples="$PROFILE_SAMPLES"
                   --verify-every="$VERIFY" --out="$prefix"
                   --min-nodes="$NODES" --max-nodes="$NODES" "${cross[@]}")
              # Add explicit MPI export of the handler ONLY on CPU-assisted
              # IBGDA jobs. Do not leak NIC-handler settings to IBRC jobs.
              if [[ "$transport" == ibgda-cpu ]]; then
                handler_arg=(-x NVSHMEM_IBGDA_NIC_HANDLER)
                cmd=("$MPIRUN" "${extra_mpi_args[@]}" --map-by "ppr:${PPN}:node" --bind-to none
                     -x NVSHMEM_REMOTE_TRANSPORT -x NVSHMEM_IB_ENABLE_IBGDA
                     "${handler_arg[@]}" -np "$NP" "$BIND" "$EXE"
                     --numgpus=1 --owned="$OWNED" --expected-ppn="$PPN"
                     --offnode-degree="$degree" --intranode-degree="$INTRA"
                     --volume="$VOLUME" --values-per-peer="$values"
                     --remote-values-total="$values" --local-values="$LOCAL_VALUES"
                     --source-pattern="$source" --source-stride="$STRIDE"
                     --target-layout="$LAYOUT" --seed="$SEED"
                     --put-variant="$variant" --chunk="$chunk"
                     --receiver-blocks="$RECEIVER_BLOCKS" --mode="$MODE"
                     --warmup="$WARMUP" --reps="$REPS"
                     --p71-reps="$PROFILE_REPS" --p71-max-samples="$PROFILE_SAMPLES"
                     --verify-every="$VERIFY" --out="$prefix"
                     --min-nodes="$NODES" --max-nodes="$NODES" "${cross[@]}")
              fi
              {
                printf 'NVSHMEM_REMOTE_TRANSPORT=ibrc\n'
                if [[ "$transport" == ibgda-cpu ]]; then
                  printf 'NVSHMEM_IB_ENABLE_IBGDA=1\nNVSHMEM_IBGDA_NIC_HANDLER=cpu\n'
                else
                  printf 'NVSHMEM_IB_ENABLE_IBGDA=0\nNVSHMEM_IBGDA_NIC_HANDLER=<unset>\n'
                fi
                printf 'command='; printf '%q ' "${cmd[@]}"; printf '\n'
              } > "$envtxt"
              printf '[%04d] %s %s degree=%s bytes=%s source=%s variant=%s chunk=%s transport=%s\n' \
                 "$run" "$TOPOLOGY" "$VOLUME" "$degree" "$bytes" "$source" "$variant" "$chunk" "$transport"
              status=DRY_RUN
              if (( DRY )); then
                printf 'DRY RUN: '; printf '%q ' "${cmd[@]}"; printf '\n'
              else
                status=0
                if (
                  export NVSHMEM_REMOTE_TRANSPORT=ibrc
                  unset NVSHMEM_IBGDA_NIC_HANDLER
                  if [[ "$transport" == ibgda-cpu ]]; then
                    export NVSHMEM_IB_ENABLE_IBGDA=1
                    export NVSHMEM_IBGDA_NIC_HANDLER=cpu
                  else
                    export NVSHMEM_IB_ENABLE_IBGDA=0
                  fi
                  if (( TIMEOUT_SECONDS > 0 )); then
                    timeout --signal=TERM --kill-after=20s "${TIMEOUT_SECONDS}s" "${cmd[@]}"
                  else
                    "${cmd[@]}"
                  fi
                ) > "$log" 2>&1; then
                  # Both the independent Importer check and P6.2 benchmarkCase
                  # must pass. The matrix-free executable never prints the
                  # standalone NvshmemP62::run '[NVSHMEM P6.2] SUMMARY' marker.
                  if ! grep -qF '[NVSHMEM TOPOLOGY IMPORTER] status=PASS' "$log" ||
                     ! grep -qF '[NVSHMEM P6.2 CASE] case=topology status=PASS' "$log" ||
                     ! grep -qF '[NVSHMEM TOPOLOGY] SUMMARY status=PASS' "$log"; then
                    status=98
                  fi
                else
                  status=$?
                fi
                if [[ "$status" != 0 ]]; then
                  failures=$((failures+1))
                  echo "FAILED run $id status=$status; inspect $log" >&2
                else
                  grep -E '\[NVSHMEM (P6\.2 DIRECT|P6\.2 P71_FULL|TOPOLOGY\] SUMMARY)' "$log" || true
                fi
              fi
              printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$id" "$pass" "$TOPOLOGY" "$VOLUME" "$degree" "$INTRA" "$bytes" "$source" "$LAYOUT" \
                "$variant" "$chunk" "$transport" "$MODE" "$status" "$log" "$prefix" >> "$MANIFEST"
              if [[ "$status" != 0 && "$status" != DRY_RUN ]] && (( FAIL_FAST )); then exit 1; fi
            done
          done
        done
      done
    done
  done
done
printf '[P7.1b SWEEP] planned=%s failures=%s directory=%s\n' "$run" "$failures" "$OUT"
(( failures == 0 )) || exit 1
if (( ! DRY )) && [[ "$MODE" == direct || "$MODE" == p71 ]]; then
  python3 "$HERE/analyze_import_topology.py" "$OUT"
  if [[ "$MODE" == p71 ]]; then
    # Separate name namespace so P7.1 and topology summary tables do not
    # clobber each other; group by degree and bytes, not merely topology.
    python3 "$HERE/analyze_p71_profiles.py" "$OUT" --output-stem p71
  fi
else
  echo 'No direct comparison analyzer for a dry run or diagnostics-only run.'
fi
