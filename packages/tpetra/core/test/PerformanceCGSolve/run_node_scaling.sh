#!/usr/bin/env bash
# Matrix-free Tpetra vs NVSHMEM P6.2 node scaling: N physical nodes, PPN=1/2/4.
# Uses the existing TpetraCore_ImportTopologyBenchmark.exe (P7.1b), unmodified.
# Scales 1..8 physical nodes, including odd node counts and intra-node controls.
# Run INSIDE a sufficiently large Slurm allocation, from Trilinos BUILD directory.
set -Eeuo pipefail
export LC_ALL=C

if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then
  cat <<'HELP'
Usage: run_node_scaling.sh
Run from the Trilinos build directory inside a Slurm allocation of up to 8
physical nodes with 4 H100 GPUs per node. It selects node SUBSETS with an
Open MPI hostfile and verifies the actual per-host process count before runs.
No power-of-two or even-node constraint: node counts 1,2,3,...,8 are valid.

Defaults (environment overrides):
  SCALE_NODES='1 2 ... N'       auto, where N=min(8, number of allocated nodes)
  SCALE_MAX_NODES=8             cap for auto nodes
  SCALE_HOSTS='hostA hostB ...'  manual host list, overrides Slurm
  SCALE_HOSTFILE=/path/hosts    one host per line (optional alternative)
  SCALE_PPN1=1 SCALE_PPN4=1     include pure inter-node (1 or 4 GPUs/node)
  SCALE_INTRA=1                one-node PPN=2 and PPN=4 controls
  SCALE_MIXED_NODES='2 4 8'    on these nodes, also use off-node+3 intra peers
  SCALE_INTRA_ONLY_NODES='2 4 8'  intra-node-only PPN=4 controls on multi-node jobs
  SCALE_FIXED_TOTAL_BYTES=65536  imported bytes per PE, fixed across degrees
  SCALE_VOLUME=fixed-total     or per-peer (different scaling question)
  SCALE_PER_PEER_BYTES=8192    used if SCALE_VOLUME=per-peer
  SCALE_OWNED=65536            owned source doubles per PE
  SCALE_SOURCE=contiguous      or strided / permuted
  SCALE_TARGET=remote-only     or prefix / interleave / shuffle
  SCALE_LOCAL_VALUES=0         remote-only requires zero local values
  SCALE_SOURCE_STRIDE=17       coprime with SCALE_OWNED if strided
  SCALE_VARIANT=block SCALE_CHUNK=1024 SCALE_RECEIVER_BLOCKS=4
  SCALE_TRANSPORTS='ibrc'      optionally 'ibrc ibgda-cpu'
  SCALE_PASSES=3 SCALE_WARMUP=8 SCALE_REPS=50 SCALE_VERIFY_EVERY=0
  SCALE_TIMEOUT=600            seconds per MPI job, 0 disables watchdog
  SCALE_MPI=mpirun             Open MPI compatible with --hostfile / ppr
  SCALE_MPI_EXTRA=''           extra MPI options, whitespace separated
  SCALE_EXE=./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_ImportTopologyBenchmark.exe
  SCALE_BIND=$HOME/hops/gpu_bind.sh   rank-to-GPU binding wrapper
  SCALE_OUT=./node-scaling-TIMESTAMP
  SCALE_ROTATE_HOSTS=1         rotate which allocated nodes are selected per pass
  SCALE_VERIFY_PLACEMENT=1     execute MPI hostname preflight per (pass,N,PPN)
  SCALE_VERIFY_EVERY=0         for a correctness-first run set 1
  SCALE_DRY_RUN=1              no MPI or GPUs; prints planned commands
  SCALE_RESUME=1               skip already successful cases in same output dir
  SCALE_FAIL_FAST=1            stop on failed job; 0 proceeds and reports failures
  SCALE_PLOT=1                 analyze CSV and produce SVG plots at end

Primary series: 2..8 nodes, PPN=1 or PPN=4, ALL off-node peers, 64 KiB
per PE. Supplementary: 1-node local controls, intra-only PPN=4 controls,
and mixed inter/intra traffic at 2,4,8 nodes. Each job compares MPI
and NVSHMEM under the SAME Tpetra Maps. Report both NVSHMEM Import-only
and NVSHMEM Import+required MPI_Barrier; only the latter is a realistic
comparison against MPI Import (which does not require this barrier).

Examples:
  SCALE_DRY_RUN=1 ./run_node_scaling.sh
  SCALE_PASSES=3 ./run_node_scaling.sh
  SCALE_NODES='2 3 4 5 6 7 8' SCALE_RESUME=1 SCALE_OUT=./run-A ./run_node_scaling.sh
  SCALE_HOSTS='hops44 hops45 hops60 hops61 ...' ./run_node_scaling.sh
HELP
  exit 0
fi
[[ $# -eq 0 ]] || { echo 'No positional arguments supported; run with --help' >&2; exit 2; }

this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
EXE="${SCALE_EXE:-./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_ImportTopologyBenchmark.exe}"
BIND="${SCALE_BIND:-$HOME/hops/gpu_bind.sh}"
MPI="${SCALE_MPI:-mpirun}"
OUT="${SCALE_OUT:-./node-scaling-$(date +%Y%m%d_%H%M%S)}"
DRY="${SCALE_DRY_RUN:-0}"
RESUME="${SCALE_RESUME:-0}"
FAIL_FAST="${SCALE_FAIL_FAST:-1}"
PLOT="${SCALE_PLOT:-1}"
PASSES="${SCALE_PASSES:-3}"
WARMUP="${SCALE_WARMUP:-8}"
REPS="${SCALE_REPS:-50}"
VERIFY_EVERY="${SCALE_VERIFY_EVERY:-0}"
TIMEOUT="${SCALE_TIMEOUT:-600}"
PPN1="${SCALE_PPN1:-1}"
PPN4="${SCALE_PPN4:-1}"
INTRA="${SCALE_INTRA:-1}"
ROTATE="${SCALE_ROTATE_HOSTS:-1}"
CHECK_PLACEMENT="${SCALE_VERIFY_PLACEMENT:-1}"
OWNED="${SCALE_OWNED:-65536}"
SOURCE="${SCALE_SOURCE:-contiguous}"
STRIDE="${SCALE_SOURCE_STRIDE:-17}"
TARGET="${SCALE_TARGET:-remote-only}"
LOCAL_VALUES="${SCALE_LOCAL_VALUES:-0}"
VOLUME="${SCALE_VOLUME:-fixed-total}"
FIXED_BYTES="${SCALE_FIXED_TOTAL_BYTES:-65536}"
PER_PEER_BYTES="${SCALE_PER_PEER_BYTES:-8192}"
VARIANT="${SCALE_VARIANT:-block}"
CHUNK="${SCALE_CHUNK:-1024}"
RECEIVER_BLOCKS="${SCALE_RECEIVER_BLOCKS:-4}"
SEED="${SCALE_SEED:-12345}"
TRANSPORTS="${SCALE_TRANSPORTS:-ibrc}"
MAX_NODES="${SCALE_MAX_NODES:-8}"
MIXED_NODES="${SCALE_MIXED_NODES:-2 4 8}"
INTRA_ONLY_NODES="${SCALE_INTRA_ONLY_NODES:-2 4 8}"
MPI_EXTRA="${SCALE_MPI_EXTRA:-}"

for name in DRY RESUME FAIL_FAST PLOT PPN1 PPN4 INTRA ROTATE CHECK_PLACEMENT VERIFY_EVERY; do
  [[ "${!name}" == 0 || "${!name}" == 1 ]] || { echo "$name must be 0 or 1" >&2; exit 2; }
done
for name in PASSES WARMUP REPS TIMEOUT OWNED STRIDE LOCAL_VALUES FIXED_BYTES PER_PEER_BYTES CHUNK RECEIVER_BLOCKS SEED MAX_NODES; do
  [[ "${!name}" =~ ^[0-9]+$ ]] || { echo "$name must be an unsigned integer" >&2; exit 2; }
done
(( PASSES >= 1 && WARMUP >= 2 && REPS >= 5 && OWNED >= 1 && CHUNK >= 1 &&
   RECEIVER_BLOCKS >= 1 && RECEIVER_BLOCKS <= 16 && MAX_NODES >= 1 && MAX_NODES <= 128 )) || {
  echo 'Invalid repetitions, owned count, chunk, receiver blocks, or max nodes' >&2; exit 2;
}
[[ "$VOLUME" == 'fixed-total' || "$VOLUME" == 'per-peer' ]] || { echo 'SCALE_VOLUME must be fixed-total|per-peer' >&2; exit 2; }
[[ "$SOURCE" == contiguous || "$SOURCE" == strided || "$SOURCE" == permuted ]] || { echo 'Bad source pattern' >&2; exit 2; }
[[ "$TARGET" == remote-only || "$TARGET" == prefix || "$TARGET" == interleave || "$TARGET" == shuffle ]] || { echo 'Bad target layout' >&2; exit 2; }
[[ "$VARIANT" == thread || "$VARIANT" == warp || "$VARIANT" == block ]] || { echo 'Bad PUT variant' >&2; exit 2; }
if [[ "$TARGET" == remote-only && "$LOCAL_VALUES" != 0 ]]; then
  echo 'remote-only target layout requires SCALE_LOCAL_VALUES=0' >&2; exit 2
fi
for bytes in "$FIXED_BYTES" "$PER_PEER_BYTES"; do
  (( bytes > 0 && bytes % 8 == 0 )) || { echo 'Byte counts must be positive multiples of 8' >&2; exit 2; }
done
read -r -a transport_list <<< "$TRANSPORTS"
read -r -a mpi_extra_arr <<< "$MPI_EXTRA"
for t in "${transport_list[@]}"; do
  [[ "$t" == 'ibrc' || "$t" == 'ibgda-cpu' ]] || { echo "Bad transport '$t'" >&2; exit 2; }
done
(( ${#transport_list[@]} > 0 )) || { echo 'Empty SCALE_TRANSPORTS' >&2; exit 2; }

# Inputs are *physical hosts*, never rank IDs or Slurm allocation slot counts.
physical_hosts=()
if [[ -n "${SCALE_HOSTS:-}" ]]; then
  read -r -a physical_hosts <<< "$SCALE_HOSTS"
elif [[ -n "${SCALE_HOSTFILE:-}" ]]; then
  [[ -f "$SCALE_HOSTFILE" ]] || { echo "Cannot read SCALE_HOSTFILE: $SCALE_HOSTFILE" >&2; exit 2; }
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    read -r first _ <<< "$line"
    [[ -z "$first" ]] || physical_hosts+=("$first")
  done < "$SCALE_HOSTFILE"
elif [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  mapfile -t physical_hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
elif (( DRY )); then
  for ((i=1;i<=MAX_NODES;i++)); do printf -v host 'dryrun-node%02d' "$i"; physical_hosts+=("$host"); done
  echo 'DRY RUN: synthetic hostnames in use; supply SCALE_HOSTS or Slurm allocation for a real run.' >&2
else
  echo 'No physical host allocation found. Set SCALE_HOSTS or SCALE_HOSTFILE, or use an active Slurm allocation.' >&2
  exit 2
fi
(( ${#physical_hosts[@]} > 0 )) || { echo 'Empty physical host allocation' >&2; exit 2; }
declare -A host_seen=()
for host in "${physical_hosts[@]}"; do
  [[ "$host" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid hostname '$host'" >&2; exit 2; }
  short="${host%%.*}"
  [[ -z "${host_seen[$short]:-}" ]] || { echo "Duplicate physical hostname: $host" >&2; exit 2; }
  host_seen[$short]=1
done
available=${#physical_hosts[@]}

if [[ -n "${SCALE_NODES:-}" ]]; then
  read -r -a node_counts <<< "$SCALE_NODES"
else
  upper=$(( available < MAX_NODES ? available : MAX_NODES ))
  node_counts=()
  for ((n=1;n<=upper;n++)); do node_counts+=("$n"); done
fi
(( ${#node_counts[@]} > 0 )) || { echo 'No requested node counts' >&2; exit 2; }
declare -A node_count_seen=()
for n in "${node_counts[@]}"; do
  [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid node count '$n'" >&2; exit 2; }
  (( n <= available )) || { echo "Requested $n physical nodes but only $available allocated" >&2; exit 2; }
  [[ -z "${node_count_seen[$n]:-}" ]] || { echo "Repeated node count $n" >&2; exit 2; }
  node_count_seen[$n]=1
done
read -r -a mixed_counts <<< "$MIXED_NODES"
read -r -a intra_only_counts <<< "$INTRA_ONLY_NODES"
for n in "${mixed_counts[@]}" "${intra_only_counts[@]}"; do
  [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid supplementary node count '$n'" >&2; exit 2; }
done
contains() { local needle="$1"; shift; local x; for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done; return 1; }

if (( ! DRY )); then
  [[ -x "$EXE" ]] || { echo "ImportTopology executable not executable: $EXE" >&2; exit 2; }
  [[ -x "$BIND" ]] || { echo "GPU binding wrapper not executable: $BIND" >&2; exit 2; }
  command -v "$MPI" >/dev/null || { echo "MPI launcher not available: $MPI" >&2; exit 2; }
  if (( TIMEOUT > 0 )); then command -v timeout >/dev/null || { echo 'GNU timeout is required' >&2; exit 2; }; fi
fi

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd -P)"
manifest="$OUT/manifest.tsv"
if [[ -e "$manifest" && "$RESUME" != 1 ]]; then
  echo "Output directory already contains a manifest: $OUT (set SCALE_RESUME=1 to retry unfinished jobs)" >&2
  exit 2
fi
if [[ ! -e "$manifest" ]]; then
  printf 'case_id\tpass\tnodes\tppn\tpe_count\tscenario\toff_degree\tintra_degree\tvolume\tbytes_setting\tremote_bytes_per_pe\tcross_bytes_per_pe\ttransport\tsource\ttarget\tvariant\tchunk\tstatus\tlog\toutput_prefix\tselected_hosts\n' > "$manifest"
fi
# Never silently blend measurements from different source revisions,
# communication volumes, GPU placement or benchmark settings on --resume.
# We intentionally include the user-visible selected host list and executable
# hash: changing them requires a fresh output directory.
exe_hash='DRY_RUN'
binder_hash='DRY_RUN'
if (( ! DRY )); then
  exe_hash="$(sha256sum "$EXE" | awk '{print $1}')"
  binder_hash="$(sha256sum "$BIND" | awk '{print $1}')"
fi
config_fingerprint="$({
  printf '%s\n' "${physical_hosts[*]}" "${node_counts[*]}" "$PASSES" "$PPN1" "$PPN4" "$INTRA" "$ROTATE"
  printf '%s\n' "$MIXED_NODES" "$INTRA_ONLY_NODES" "$VOLUME" "$FIXED_BYTES" "$PER_PEER_BYTES"
  printf '%s\n' "$OWNED" "$SOURCE" "$STRIDE" "$TARGET" "$LOCAL_VALUES" "$VARIANT" "$CHUNK" "$RECEIVER_BLOCKS" "$SEED"
  printf '%s\n' "$TRANSPORTS" "$WARMUP" "$REPS" "$VERIFY_EVERY" "$MPI" "$MPI_EXTRA"
  printf '%s\n' "$EXE" "$BIND" "$exe_hash" "$binder_hash" "$DRY"
} | sha256sum | awk '{print $1}')"
if [[ -e "$OUT/config_fingerprint.sha256" ]]; then
  saved="$(cat "$OUT/config_fingerprint.sha256")"
  if [[ "$saved" != "$config_fingerprint" ]]; then
    echo "Output directory contains DIFFERENT benchmark configuration: $OUT" >&2
    echo 'Refusing to mix data; choose a fresh SCALE_OUT directory.' >&2
    exit 2
  fi
else
  printf '%s\n' "$config_fingerprint" > "$OUT/config_fingerprint.sha256"
fi
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'available_nodes=%s\n' "$available"
  printf 'physical_hostnames=%s\n' "${physical_hosts[*]}"
  printf 'node_counts=%s\n' "${node_counts[*]}"
  printf 'volume=%s\nfixed_total_bytes=%s\nper_peer_bytes=%s\n' "$VOLUME" "$FIXED_BYTES" "$PER_PEER_BYTES"
  printf 'owned=%s\nsource=%s\ntarget=%s\nvariant=%s\nchunk=%s\n' "$OWNED" "$SOURCE" "$TARGET" "$VARIANT" "$CHUNK"
  printf 'passes=%s\nwarmup=%s\nrepetitions=%s\nverify_every=%s\n' "$PASSES" "$WARMUP" "$REPS" "$VERIFY_EVERY"
  printf 'transports=%s\nrotated_hosts=%s\n' "$TRANSPORTS" "$ROTATE"
  printf 'executable=%s\nbinder=%s\n' "$EXE" "$BIND"
  printf 'sweep_type=physical_node_scaling\nmatrix_constructed=NO\nsource_storage=ordinary_Tpetra_CudaSpace\n'
} > "$OUT/config.txt"
if (( ! DRY )); then
  sha256sum "$EXE" "$BIND" > "$OUT/sha256_files.txt" || true
  ("$MPI" --version 2>&1 | head -3) > "$OUT/mpi_version.txt" || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$OUT/local_gpu_info.txt" 2>/dev/null || true
  fi
fi

# Create/validate hostfile once per pass/node count/PPN, not inside each timed run.
make_hostfile() {
  local pass="$1" nodes="$2" ppn="$3"; local file="$OUT/hosts_p${pass}_n${nodes}_ppn${ppn}.txt"
  local start=0
  if (( ROTATE )); then start=$(( (pass - 1) % available )); fi
  local i host
  : > "$file"
  for ((i=0;i<nodes;i++)); do
    host="${physical_hosts[$(((start + i) % available))]}"
    printf '%s slots=%s max_slots=%s\n' "$host" "$ppn" "$ppn" >> "$file"
  done
  printf '%s\n' "$file"
}

run_checked() {
  if (( TIMEOUT > 0 )); then
    timeout --signal=TERM --kill-after=20s "${TIMEOUT}s" "$@"
  else
    "$@"
  fi
}

check_hostfile() {
  local file="$1" nodes="$2" ppn="$3" actual verified
  actual="$OUT/placement_$(basename "$file")"
  verified="$actual.verified"
  (( DRY || ! CHECK_PLACEMENT )) && return 0
  [[ -e "$verified" ]] && return 0
  local -a cmd=("$MPI" "${mpi_extra_arr[@]}" --hostfile "$file" --map-by "ppr:${ppn}:node" --bind-to none -np "$((nodes*ppn))" /bin/hostname)
  run_checked "${cmd[@]}" > "$actual" 2> "$actual.stderr" || {
    echo "MPI host placement preflight failed: $actual (stderr: $actual.stderr)" >&2; return 1;
  }
  declare -A expected=() counted=()
  local host line short
  while IFS= read -r line; do
    read -r host _ <<< "$line"
    short="${host%%.*}"
    expected[$short]="$ppn"
  done < "$file"
  while IFS= read -r line; do
    short="${line%%.*}"
    [[ -n "${expected[$short]:-}" ]] || {
      echo "Unexpected host $short in MPI placement output $actual" >&2; return 1;
    }
    counted[$short]="$(( ${counted[$short]:-0} + 1 ))"
  done < "$actual"
  for short in "${!expected[@]}"; do
    if [[ "${counted[$short]:-0}" != "${expected[$short]}" ]]; then
      echo "Wrong GPU PE placement on $short: got ${counted[$short]:-0}, expected ${expected[$short]}" >&2
      return 1
    fi
  done
  printf 'nodes=%s ppn=%s hosts=%s\n' "$nodes" "$ppn"     "$(awk '{printf "%s,", $1}' "$file")" > "$verified"
  echo "Placement PASS: ${nodes} physical nodes × ${ppn} PEs/node, hostfile $(basename "$file")"
}

# Deterministic scenario identities for analysis and resume.
# off1/off4: ALL possible off-node neighbors. mixed4: same plus 3 intra.
# intra4multi: no off-node transfers even with multiple physical nodes.
scenario_spec() {
  local n="$1" name="$2"; local ppn off intra route
  case "$name" in
    intra2)     ppn=2; off=0; intra=1; route=intra ;;
    intra4)     ppn=4; off=0; intra=3; route=intra ;;
    off1)       ppn=1; off=$((n-1)); intra=0; route=off ;;
    off4)       ppn=4; off=$(((n-1)*4)); intra=0; route=off ;;
    mixed4)     ppn=4; off=$(((n-1)*4)); intra=3; route=mixed ;;
    intra4multi) ppn=4; off=0; intra=3; route=intra ;;
    *) echo "Unknown scenario $name" >&2; return 2 ;;
  esac
  printf '%s %s %s %s\n' "$ppn" "$off" "$intra" "$route"
}

planned=0
executed=0
skipped=0
failed=0
for ((pass=1;pass<=PASSES;pass++)); do
  ordered_nodes=("${node_counts[@]}")
  if (( pass % 2 == 0 )); then
    reordered=()
    for ((i=${#ordered_nodes[@]}-1;i>=0;i--)); do reordered+=("${ordered_nodes[i]}"); done
    ordered_nodes=("${reordered[@]}")
  fi
  for nodes in "${ordered_nodes[@]}"; do
    scenarios=()
    if (( nodes == 1 )); then
      if (( INTRA )); then scenarios+=(intra2 intra4); fi
    else
      if (( PPN1 )); then scenarios+=(off1); fi
      if (( PPN4 )); then scenarios+=(off4); fi
      if (( PPN4 )) && contains "$nodes" "${mixed_counts[@]}"; then scenarios+=(mixed4); fi
      if (( INTRA && PPN4 )) && contains "$nodes" "${intra_only_counts[@]}"; then scenarios+=(intra4multi); fi
    fi
    for scenario in "${scenarios[@]}"; do
      read -r ppn off intra route < <(scenario_spec "$nodes" "$scenario")
      degree=$((off+intra))
      [[ "$degree" -gt 0 ]] || { echo 'Selected empty Import graph' >&2; exit 2; }
      if [[ "$VOLUME" == fixed-total ]]; then
        bytes_setting="$FIXED_BYTES"
        remote_bytes="$FIXED_BYTES"
        values_per_peer=1 # unused in fixed-total mode
        total_values=$((FIXED_BYTES/8))
      else
        bytes_setting="$PER_PEER_BYTES"
        remote_bytes=$((PER_PEER_BYTES*degree))
        values_per_peer=$((PER_PEER_BYTES/8))
        total_values=1 # unused in per-peer mode
      fi
      if (( remote_bytes % 8 != 0 || remote_bytes <= 0 )); then
        echo "Unrepresentable communication volume $remote_bytes" >&2; exit 2
      fi
      if (( (remote_bytes/8 + degree - 1)/degree > OWNED )); then
        echo "Requested more than $OWNED owned values per peer: $scenario, $nodes nodes" >&2; exit 2
      fi
      if [[ "$VOLUME" == fixed-total ]] && (( total_values < degree )); then
        echo 'Fixed-total bytes too small for the selected degree' >&2; exit 2
      fi
      if [[ "$VOLUME" == fixed-total ]]; then
        cross_bytes=$(( (remote_bytes * off) / degree )) # display estimate only!
        # Exact split follows the quotient/remainder assignment of graph layers.
        q=$((total_values/degree)); rem=$((total_values%degree))
        cross_doubles=$((q*off + (off<rem ? off : rem)))
        cross_bytes=$((cross_doubles*8))
      else
        cross_bytes=$((PER_PEER_BYTES*off))
      fi
      hfile="$(make_hostfile "$pass" "$nodes" "$ppn")"
      selected_hosts="$(awk '{printf "%s%s", (NR==1?"":","), $1}' "$hfile")"
      # If an MPI job does not place ranks on the intended subset, the benchmark
      # will have a different node count and must never silently enter the plot.
      check_hostfile "$hfile" "$nodes" "$ppn"
      order_t=("${transport_list[@]}")
      if (( pass % 2 == 0 )); then
        reverse_t=()
        for ((i=${#order_t[@]}-1;i>=0;i--)); do reverse_t+=("${order_t[i]}"); done
        order_t=("${reverse_t[@]}")
      fi
      for transport in "${order_t[@]}"; do
        planned=$((planned+1))
        case_id="p${pass}_n${nodes}_ppn${ppn}_${scenario}_${VOLUME}_${transport}"
        prefix="$OUT/$case_id"
        log="$prefix.log"
        if (( RESUME )) && [[ -s "$log" && -s "${prefix}_topology_p62_samples.csv" && -s "${prefix}_traffic.csv" ]] &&
            grep -qF '[NVSHMEM TOPOLOGY IMPORTER] status=PASS' "$log" &&
            grep -qF '[NVSHMEM P6.2 CASE] case=topology status=PASS' "$log" &&
            grep -qF '[NVSHMEM TOPOLOGY] SUMMARY status=PASS' "$log"; then
          skipped=$((skipped+1))
          printf 'SKIP previously PASS: %s\n' "$case_id"
          continue
        fi
        export_handler=()
        if [[ "$transport" == ibgda-cpu ]]; then export_handler=(-x NVSHMEM_IBGDA_NIC_HANDLER); fi
        cmd=("$MPI" "${mpi_extra_arr[@]}" --hostfile "$hfile"
             --map-by "ppr:${ppn}:node" --bind-to none
             -x NVSHMEM_REMOTE_TRANSPORT -x NVSHMEM_IB_ENABLE_IBGDA
             "${export_handler[@]}" -np "$((nodes*ppn))"
             "$BIND" "$EXE" --numgpus=1 --owned="$OWNED" --expected-ppn="$ppn"
             --offnode-degree="$off" --intranode-degree="$intra"
             --volume="$VOLUME" --values-per-peer="$values_per_peer"
             --remote-values-total="$total_values" --local-values="$LOCAL_VALUES"
             --source-pattern="$SOURCE" --source-stride="$STRIDE"
             --target-layout="$TARGET" --seed="$SEED"
             --put-variant="$VARIANT" --chunk="$CHUNK" --receiver-blocks="$RECEIVER_BLOCKS"
             --mode=direct --warmup="$WARMUP" --reps="$REPS"
             --verify-every="$VERIFY_EVERY" --out="$prefix"
             --min-nodes="$nodes" --max-nodes="$nodes")
        if (( off > 0 )); then cmd+=(--require-cross-node); fi
        {
          printf 'date=%s\ncase_id=%s\nselected_hosts=%s\n' "$(date --iso-8601=seconds)" "$case_id" "$selected_hosts"
          printf 'nodes=%s\nppn=%s\nscenario=%s\noff_degree=%s\nintra_degree=%s\n' "$nodes" "$ppn" "$scenario" "$off" "$intra"
          printf 'remote_bytes_per_PE=%s\ncross_node_bytes_per_PE=%s\n' "$remote_bytes" "$cross_bytes"
          printf 'NVSHMEM_REMOTE_TRANSPORT=ibrc\n'
          if [[ "$transport" == ibrc ]]; then
            printf 'NVSHMEM_IB_ENABLE_IBGDA=0\nNVSHMEM_IBGDA_NIC_HANDLER=<unset>\n'
          else
            printf 'NVSHMEM_IB_ENABLE_IBGDA=1\nNVSHMEM_IBGDA_NIC_HANDLER=cpu\n'
          fi
          printf 'NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH=<unset>\nNVSHMEM_IBGDA_NUM_RC_PER_PE=<unset>\n'
          printf 'command='; printf '%q ' "${cmd[@]}"; printf '\n'
        } > "${prefix}.env.txt"
        printf '[%03d] N=%s ppn=%s scenario=%-11s peers=%s+%s bytes/PE=%s %s pass=%s\n' \
            "$planned" "$nodes" "$ppn" "$scenario" "$off" "$intra" "$remote_bytes" "$transport" "$pass"
        status=DRY_RUN
        if (( DRY )); then
          printf '  command: '; printf '%q ' "${cmd[@]}"; printf '\n'
        else
          status=0
          if (
            export NVSHMEM_REMOTE_TRANSPORT=ibrc
            unset NVSHMEM_IBGDA_NIC_HANDLER NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH NVSHMEM_IBGDA_NUM_RC_PER_PE
            if [[ "$transport" == ibgda-cpu ]]; then
              export NVSHMEM_IB_ENABLE_IBGDA=1 NVSHMEM_IBGDA_NIC_HANDLER=cpu
            else
              export NVSHMEM_IB_ENABLE_IBGDA=0
            fi
            run_checked "${cmd[@]}"
          ) > "$log" 2>&1; then
            for marker in '[NVSHMEM TOPOLOGY IMPORTER] status=PASS' \
                          '[NVSHMEM P6.2 CASE] case=topology status=PASS' \
                          '[NVSHMEM TOPOLOGY] SUMMARY status=PASS'; do
              if ! grep -qF "$marker" "$log"; then status=98; break; fi
            done
            if [[ "$status" == 0 && ( ! -s "${prefix}_topology_p62_samples.csv" || ! -s "${prefix}_traffic.csv" ) ]]; then
              status=97
            fi
          else
            status=$?
          fi
          executed=$((executed+1))
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$case_id" "$pass" "$nodes" "$ppn" "$((nodes*ppn))" "$scenario" "$off" "$intra" \
          "$VOLUME" "$bytes_setting" "$remote_bytes" "$cross_bytes" "$transport" "$SOURCE" \
          "$TARGET" "$VARIANT" "$CHUNK" "$status" "$log" "$prefix" "$selected_hosts" >> "$manifest"
        if [[ "$status" != 0 && "$status" != DRY_RUN ]]; then
          failed=$((failed+1))
          echo "FAILED $case_id status=$status (see $log)" >&2
          if (( FAIL_FAST )); then
            echo "Use SCALE_RESUME=1 SCALE_OUT=$OUT to resume, then run the analyzer." >&2
            exit 1
          fi
        fi
      done
    done
  done
done
printf '[NODE SCALING] planned=%d executed=%d skipped=%d failures=%d output=%s\n' \
  "$planned" "$executed" "$skipped" "$failed" "$OUT"
if (( ! DRY && PLOT )); then
  python3 "$this_dir/analyze_node_scaling.py" "$OUT"
fi
(( failed == 0 ))
