# Physical-node scalability benchmark — Tpetra/MPI versus NVSHMEM P6.2

The scripts in this bundle drive the **existing matrix-free**
`TpetraCore_ImportTopologyBenchmark.exe`. They do not modify Trilinos,
NVSHMEM, the Importer, or CUDA kernels. **Use them from your Trilinos build
directory**, inside an allocation with the physical nodes you want to use.

This is a node-scaling study, **not another MiniFE decomposition study**.
The default benchmark fixes remote traffic at **65,536 bytes (8,192 doubles)
per PE** while increasing physical node count and hence possible communication
partners. For each experiment both stock Tpetra `doImport()` and existing
NVSHMEM P6.2 import use the *same* synthetic source/target Maps.

## Experiment design

With eight allocated physical nodes, each having four H100 GPUs, the default
scripts make **22 distinct configurations × 3 independent passes = 66 MPI
launches** (IBRC only). It uses **every node count from 1 through 8**, including
3, 5, and 7; there is no even-/power-of-two requirement.

| Profile | Node counts | GPU PEs per node | Source peers per PE | Traffic type |
|---|---|---:|---|---|
| `intra2` | 1 | 2 | 1 intra | Single-node GPU-to-GPU baseline |
| `intra4` | 1 | 4 | 3 intra | Full H100 NVLink/P2P node |
| `off1` | 2–8 | 1 | `nodes-1` off | **All traffic must use the inter-node network** |
| `off4` | 2–8 | 4 | `4×(nodes-1)` off | All four GPUs per node, all off-node peers |
| `mixed4` | 2, 4, 8 | 4 | `4×(nodes-1)` off + 3 intra | Same volume shared across on/off-node peers |
| `intra4multi` | 2, 4, 8 | 4 | 3 intra only | Same-node communication, but an increasing global barrier population |

For eight nodes, `off4` means **32 PEs, 28 distinct off-node partners per PE**.
The total incoming traffic **per PE remains 64 KiB**, so this is a controlled
*communication fragmentation / node-count scaling* study. It is not a
fixed-per-peer traffic study: an increasing degree divides the 64 KiB among
more peers. The total global volume increases with the total PE count.

The optional `SCALE_VOLUME=per-peer SCALE_PER_PEER_BYTES=8192` changes the
experiment to a fixed 8 KiB *per peer* study. Run that as a **separate output
directory**, never mix the two volume definitions in the same scaling figure.

Defaults match the successful prior 16-PE topology run: ordinary IBRC
(`NVSHMEM_IB_ENABLE_IBGDA=0`), source pattern `contiguous`, target
`remote-only`, `block` PUT collective, and `--chunk=1024` doubles. You can
select `SCALE_TRANSPORTS='ibrc ibgda-cpu'` to compare the CPU-handler IBGDA
path explicitly. Given the earlier 4-node results, **IBRC-only** is the
recommended primary scalability figure.

## How the driver controls physical nodes

1. In an existing Slurm allocation, the script queries
   `scontrol show hostnames "$SLURM_JOB_NODELIST"` to discover **actual physical
   hostnames**, not merely rank counts. Alternatively set
   `SCALE_HOSTS='hops11 hops12 ...'`, or set `SCALE_HOSTFILE` to a plain file
   containing one allocated hostname per line.
2. For node count `n`, it writes an Open MPI **hostfile containing exactly n
   selected nodes** with `slots=PPN max_slots=PPN` per node.
3. Launches use `mpirun --hostfile FILE --map-by ppr:PPN:node --bind-to none
   -np n*PPN` and the existing GPU-binding wrapper. **Every selected
   (pass,n,PPN) placement is verified** by a separate MPI hostname preflight;
   wrong process counts fail before timing.
4. The selected subset rotates each pass when `SCALE_ROTATE_HOSTS=1` (default),
   while reversing the node count traversal on even passes. These controls
   reduce dependence on one physical node subset and chronological drift.
   Each run records the exact hosts, topology CSV, executable hashes, command
   line, environment, and raw rank timing samples.

The cluster's GPU-binding wrapper must place ranks 0–3 on **distinct GPUs**
when PPN=4. `--numgpus=1` assumes the wrapper restricts each rank's CUDA
visibility to its assigned device, consistent with your previous tests.
The hostname preflight verifies **node and rank placement**, not GPU UUIDs;
check your wrapper once before the performance sweep.

Open MPI provides the `--hostfile` and `--map-by ppr:N:node` controls, and
restricts a hostfile to the named subset of an existing resource-manager
allocation. This is why the script does not simply ask for `-np` on an
8-node allocation and assume the MPI launcher will pick the desired hosts.

## Running the full study

From your **Trilinos build directory** inside a Slurm allocation of 8 nodes:

```bash
# Put run_node_scaling.sh and analyze_node_scaling.py together.
chmod +x /path/to/run_node_scaling.sh

SCALE_EXE=./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_ImportTopologyBenchmark.exe \
SCALE_BIND="$HOME/hops/gpu_bind.sh" \
SCALE_NODES='1 2 3 4 5 6 7 8' \
SCALE_PASSES=3 \
SCALE_OUT="./nvshmem-node-scaling-${SLURM_JOB_ID:-manual}" \
  /path/to/run_node_scaling.sh
```

No need to recompile Trilinos for this sweep; it invokes the already built
P7.1b executable. The sweep automatically invokes the companion analyzer and
produces SVG figures. Allow enough allocation wall time for 66 independent
MPI startups and normal warmup/measurement runs.

**First: a short correctness smoke**, exercising local and multi-node paths:

```bash
SCALE_NODES='1 2 3' \
SCALE_PASSES=1 SCALE_REPS=5 SCALE_WARMUP=2 SCALE_VERIFY_EVERY=1 \
SCALE_OUT="./node-scaling-smoke-$(date +%Y%m%d_%H%M%S)" \
  /path/to/run_node_scaling.sh
```

**Optional fixed-per-peer traffic scaling** (separate study, fixed bytes per
peer rather than fixed total remote bytes):

```bash
SCALE_VOLUME=per-peer SCALE_PER_PEER_BYTES=8192 \
SCALE_NODES='2 3 4 5 6 7 8' SCALE_PASSES=3 \
SCALE_OUT="./node-scaling-per-peer-$(date +%Y%m%d_%H%M%S)" \
  /path/to/run_node_scaling.sh
```

**For a quick plan-only dry run, without Slurm or GPUs**:

```bash
SCALE_DRY_RUN=1 SCALE_MAX_NODES=8 SCALE_PASSES=1 \
SCALE_OUT=./node-scaling-dryrun /path/to/run_node_scaling.sh
```

**To resume a partial allocation run**, provide identical configuration,
script inputs and executable/binder hashes, the same `SCALE_OUT`, and:

```bash
SCALE_RESUME=1 SCALE_OUT=./nvshmem-node-scaling-YOUR_RUN \
  /path/to/run_node_scaling.sh
```

The script checks a configuration fingerprint before resuming, skips only
cases with all three required PASS markers and raw timing/traffic files,
and appends a failed/retried result only when required. The analyzer uses
the **latest manifest record per case**.

Run the Python analyzer independently when needed:

```bash
python3 /path/to/analyze_node_scaling.py ./nvshmem-node-scaling-YOUR_RUN
```

## Figure/data outputs

All outputs appear in `SCALE_OUT`. **No matplotlib/pandas installation is
required**; figure generation uses the Python standard library and writes
editable, publication-quality SVG vector plots.

- `scaling_runs.csv`: independently validated launch-level results with exact
  physical node count/PPN, peer degrees and bytes/PE.
- `scaling_summary.csv`: aggregation across independent MPI launches,
  including observed minimum/maximum for error bars.
- `scaling_report.md`: formatted summary and metrics/caveats.
- `speedup_vs_nodes_ibrc.svg`: **primary figure**, x = physical node count,
  y = `MPI_Import / (P6.2_Import + required_MPI_Barrier)`. Curves for PPN=1
  and PPN=4, with one-node intra-GPU reference points.
- `latency_vs_nodes_ppn1_ibrc.svg`, `latency_vs_nodes_ppn4_ibrc.svg`: MPI,
  P6.2 with barrier, and P6.2 Import alone (dashed; diagnostic only).
- `local_and_mixed_controls_ibrc.svg`: isolates intra-only, all-off-node,
  and mixed communication behavior at the same total remote volume.
- `*.log`, `*.env.txt`, `*_traffic.csv`, `*_topology_p62_samples.csv`,
  `hosts_*.txt`, `placement_*.verified`: complete reproducibility evidence.

The **barrier-inclusive** P6.2 metric is computed at the PE/iteration level:

`max_rank[ time_P6.2(r,epoch) + time_MPI_barrier(r,epoch) ]`

and then reduced to a median over iterations. This is **not** the sum of
independent maxima or medians. Stock MPI's `doImport()` does not require a
world barrier between Imports, so its reference time **does not include**
an extra barrier. The P6.2 Import-only number is still reported to show
its communication potential; do not use that number for the realistic
speedup claim.

Every benchmark call verifies the Importer and outputs according to the
existing P7.1b correctness settings. With `SCALE_VERIFY_EVERY=0`, it still
checks its configured final/warmup correctness; use `=1` for the short
acceptance suite. The script fails if any of the Importer, P6.2 case, or
topology PASS markers is missing, and the analyzer independently validates
every byte count and each rank/epoch timing record.

## Validation done while creating these scripts

- Dry-run command generation for all node counts 1–8, including 3/5/7, and
  the default 22 configurations. Reversals/host rotation included.
- Host-side end-to-end smoke of MPI-CLI mock runner, exact host subset
  selection and PPN checking, CSV generation, analysis, SVG generation,
  successful run resume and strict failure handling. **Mock timings are
  synthetic fixtures, not real performance data.**
- The analyzer also successfully read **all 24 authentic four-node runs**
  from the provided `topology-results-20261026_005249.tar.gz`, reproducing
  the prior result of approximately **1.99× barrier-inclusive speedup** for
  12 off-node peers using IBRC. These authentic numbers validate the parser
  but are **not** evidence of 8-node scaling.

The scripts have not been executed on a live Slurm allocation, OpenMPI/H100
fabric or the actual 8-node system. Check your MPI/Slurm site policy,
GPU binder, and topology placement preflight before trusting timings.
