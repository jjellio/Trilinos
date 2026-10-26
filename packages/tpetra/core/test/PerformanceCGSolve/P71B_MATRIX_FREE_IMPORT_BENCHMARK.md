# P7.1b — Controlled, matrix-free Tpetra Import communication benchmark

**Code status:** Implemented against the P7.1 clean source archive `nvshmem-pack-P7.1-full-source.tar.xz`. Portable C++17 graph tests, parser fixtures, launcher dry-runs, and existing host regressions pass. **Native Trilinos/NVCC and multi-GPU/NVSHMEM runtime correctness/performance must be run on the cluster.** P6.2 kernels and P7.1 instrumented variants have **not been changed**.

## Why this replaces MiniFE as the primary performance study

MiniFE `--size=100` yields a specific, topology-dependent Importer with a few dominant send/receive relationships and a small rank-0 fanout. MPI's naturally coarse packets are favored, while most of the earlier 8-PE communication was intra-node. Those are legitimate cases but insufficient for assessing the algorithm's scalability. **A CrsMatrix is not required to construct an Importer.** We now build one directly using synthetic balanced source/target `Tpetra::Map` objects and use the existing reference `NvshmemP62::benchmarkCase` unchanged.

The source domain map owns the same `N` GIDs per PE:

```
GID(ownerRank, sourceLID) = ownerRank * N + sourceLID
sourceLID = 0 .. N-1
```

The target map requests exactly the GIDs selected by the communication graph, optionally interleaved with a specified number of local values. It has no hidden rank-0 request, no sparse matrix, and no assumed Cartesian geometry. The actual Tpetra Importer is built from these Maps, then its owner-local LIDs, exporting ranks, receiving ranks and per-peer counts are checked against the requested graph **before** NVSHMEM setup or timing. Unknown GIDs and duplicate target GIDs are rejected.

### Topology construction

All ranks gather their true physical hostnames (`MPI_Get_processor_name`), group MPI ranks by hostname and assign a stable rank-local slot within each node. Uniform PPN is required; nonidentity MPI-rank↔NVSHMEM-PE mappings remain supported by the existing P0 and P5.1 infrastructure. Neither contiguous rank numbering per node nor sorted rank-to-node placement is assumed.

Every communication layer is a **bijective permutation** of ranks. A cross-node layer sends from `(node,slot)` to `((node+nodeDelta)%nodes, (slot+slotDelta)%PPN)`; an intra-node layer changes only the local slot, with nonzero `slotDelta`. Layers do not repeat directed source/target relationships. Combining `k_off` cross-node layers and `k_intra` same-node layers produces:

- Exactly `k_off` off-node **send peers and receive peers** per rank;
- Exactly `k_intra` intra-node send and receive peers per rank;
- No self transfers, no unexpected rank-0 fanout, and exactly equal send and receive payload totals per rank;
- Works for *odd* node counts too. Two or three ranks may use directed rings without requiring an undirected degree-1 pairing.

Bounds: `0 <= k_off <= (nodes-1)*PPN`, `0 <= k_intra <= PPN-1` and at least one selected peer (or positive local copies) is required. Uniform PPN is rejected early if actual placement differs.

**Two independent volume modes:**

- `--volume=per-peer --values-per-peer=M`: every directed peer transfer has exactly `M` doubles, and total remote volume per rank is `degree*M`.
- `--volume=fixed-total --remote-values-total=T`: total remote volume is exactly `T` doubles per rank for *every* degree; distribute `T` over the graph's permutation layers using quotient/remainder. The first `T % degree` layers get one extra double. Thus a 64 KiB total distributed over 12 partners has some 683-double and some 682-double packets, while **each rank still sends/receives exactly 8,192 doubles**. Counts are deterministic and balanced by layer, even for directed three-node rings.

`--owned=N` must be at least as large as each selected per-peer source count, and total/source/target Map lengths must fit the instantiated Tpetra `LO`/`GO` types.

### Source packing and target layouts

| Option | Behavior | Purpose |
|---|---|---|
| `--source-pattern=contiguous` | A peer-specific contiguous interval of source LIDs, no wrap | Separate transport cost from irregular gather |
| `--source-pattern=strided --source-stride=17` | Modulo-`N` stride, with step coprime to `N` | Test deterministic strided gathers, no duplicate local LIDs |
| `--source-pattern=permuted` | Reproducible hashed affine permutation of source LIDs | Test irregular gathers without changing communication volume |
| `--target-layout=remote-only --local-values=0` | Only remotely sourced column-vector values | Isolate communication/unpack, no local-copy overhead |
| `--target-layout=prefix --local-values=L` | Include `L` local GIDs followed by remote GIDs | Test local-copy region |
| `--target-layout=interleave --local-values=L` | Spread locally owned requests among remote GIDs | Test noncontiguous scatter plus local permutation |
| `--target-layout=shuffle --local-values=L` | Deterministically shuffle the entire target list | Stress arbitrary destination LID mapping |

The chosen memory model is intentionally the current **P6.2 Model A**: ordinary Tpetra CUDA-space source `Vector`, symmetric send scratch, symmetric inbox, and ordinary Tpetra target `Vector`. A fully symmetric source/vector allocation would change staging assumptions and should be a distinct later experiment, **not mixed into this topology benchmark**. GID↔LID semantics remain those defined by the original Maps; storage base addresses are an independent issue.

## Build

`CMakeLists.txt` introduces `ImportTopologyBenchmark` in the existing
`PERFORMANCE_CGSOLVE_NVSHMEM=ON` CUDA/double configuration, separate from
`Performance-CGSolve`. It does **not** add any topology options to `cg_solve_file.cpp`.

Build Trilinos as before, including relocatable CUDA device code, and rebuild the
new target (your build tool may call the TriBITS target
`TpetraCore_ImportTopologyBenchmark`). The expected executable is:

```
./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_ImportTopologyBenchmark.exe
```

Set `PERFORMANCE_CGSOLVE_IMPORT_TOPOLOGY_CTEST=ON` to register an opt-in one-PE **local-copy-only smoke**; it cannot validate InfiniBand. The older optional P6.2 and P7.1 CTests and original CG solver remain unchanged.

## First runs: pure inter-node, controlled peer degree

Run from the **build directory** under an allocation containing the requested distinct physical nodes. A rank-local binding wrapper may expose only one GPU per rank (hence `--numgpus=1`); otherwise check device mappings.

**2 nodes, PPN=1, one peer and an 8 KiB packet:**

```bash
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=0
mpirun --map-by ppr:1:node --bind-to none -np 2 \
  "$HOME/cronus/gpu_bind.sh" \
  ./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_ImportTopologyBenchmark.exe \
  --numgpus=1 --expected-ppn=1 --owned=65536 \
  --offnode-degree=1 --intranode-degree=0 \
  --volume=per-peer --values-per-peer=1024 \
  --source-pattern=contiguous --target-layout=remote-only \
  --put-variant=block --chunk=1024 --mode=direct \
  --warmup=5 --reps=30 --verify-every=1 \
  --min-nodes=2 --max-nodes=2 --require-cross-node \
  --out=p71b-inter2-block1024
```

**4 nodes, PPN=1, degree 3, 64 KiB TOTAL per PE:**

```bash
mpirun --map-by ppr:1:node --bind-to none -np 4 \
  "$HOME/cronus/gpu_bind.sh" \
  ./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_ImportTopologyBenchmark.exe \
  --numgpus=1 --expected-ppn=1 --owned=65536 \
  --offnode-degree=3 --volume=fixed-total --remote-values-total=8192 \
  --source-pattern=contiguous --target-layout=remote-only \
  --put-variant=block --chunk=1024 --mode=direct \
  --warmup=5 --reps=30 --verify-every=1 \
  --min-nodes=4 --max-nodes=4 --require-cross-node \
  --out=p71b-inter4-degree3
```

**4 nodes, PPN=4, degree 12, 64 KiB total per PE:**

```bash
mpirun --map-by ppr:4:node --bind-to none -np 16 \
  "$HOME/cronus/gpu_bind.sh" \
  ./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_ImportTopologyBenchmark.exe \
  --numgpus=1 --expected-ppn=4 --owned=65536 \
  --offnode-degree=12 --volume=fixed-total --remote-values-total=8192 \
  --source-pattern=strided --source-stride=17 --target-layout=remote-only \
  --put-variant=block --chunk=1024 --mode=p71 \
  --warmup=5 --reps=30 --p71-reps=12 --p71-max-samples=64 \
  --verify-every=1 --min-nodes=4 --max-nodes=4 --require-cross-node \
  --out=p71b-inter16-d12
```

The P7.1 profile mode performs an **unmodified P6.2 completed-Import comparison** followed by **separately instrumented clone kernels**; it is not a new completion protocol. A scheduler timeout remains important because the original bounded GPU polling scheme is experimental.

## Repeatable sweep and topology-aware analysis

Place the scripts under `packages/tpetra/core/test/PerformanceCGSolve/` and run from the **build directory**:

```bash
P71B_BIND="$HOME/cronus/gpu_bind.sh" \
P71B_PASSES=3 \
  /path/to/Trilinos/packages/tpetra/core/test/PerformanceCGSolve/run_import_topology_sweep.sh \
  inter2 per-peer

P71B_BIND="$HOME/cronus/gpu_bind.sh" \
P71B_DEGREES='1 2 4 8 12' \
P71B_TOTAL_BYTES='65536' \
  /path/to/Trilinos/packages/tpetra/core/test/PerformanceCGSolve/run_import_topology_sweep.sh \
  inter16 fixed-total
```

Other topologies: `inter3` (3 nodes × 1), `inter4` (4 × 1), `inter8` (2 × 4), `inter12` (3 × 4), and `intra4` (1 × 4). Use `P71B_DRY_RUN=1` to validate the complete command matrix without MPI/GPU. The default runner executes IBRC (`NVSHMEM_IB_ENABLE_IBGDA=0`) and IBGDA **with CPU handler** (`NVSHMEM_IB_ENABLE_IBGDA=1`, `NVSHMEM_IBGDA_NIC_HANDLER=cpu`), distinct from unsupported GPU doorbell handling on the previously observed Cronus setup. It never passes symmetric VMM memory to MPI.

Output: one `*_traffic.csv` with all `N²` sender/receiver pairs including zeros; `*_rank_topology.csv` with each rank's actual hostname and slot; the ordinary `*_topology_p62_samples.csv` with per-rank direct/barrier timings; all-rank GPU probes in `*_topology_p71_rankR_probes.csv` when enabled; `manifest.tsv`, command/environment files, and logs. No file I/O or plan construction is in the timed Import operation.

`analyze_import_topology.py` automatically generates `runs.csv` and `summary.csv` after a direct/p71 sweep. It **recomputes rank-max-per-iteration medians** from the raw CSV, independently pairs the P6.2 barrier and completed Import by rank and iteration, and validates the NxN traffic graph. A reported P6.2+barrier median is calculated from paired values, **never the sum of separately reduced medians**. Independent MPI launches, not 30–50 in-process iterations, are the replications. The analyzer refuses missing PASS results, incorrect traffic totals, unmatched rank samples or failed jobs.

With `--mode=p71`, the runner also calls `analyze_p71_profiles.py --output-stem p71`, so its diagnostic `p71_runs.csv`, `p71_summary.csv`, `p71_probe_distributions.csv`, `p71_analysis.md` do **not** overwrite topology comparison files. Its grouping keys now include degree, payload size, source pattern, volume definition and target layout. Profile samples are approximate SM clock intervals, not direct NIC latency measurements.

## Suggested controlled experiments

| MPI placement | Allowed off-node degree(s) | Character |
|---|---|---|
| 2 nodes × 1 GPU | 1 | Pure off-node single-peer best case for MPI |
| 3 nodes × 1 GPU | 1, 2 | Pure off-node peer-count scaling; directed ring at degree 1 |
| 4 nodes × 1 GPU | 1, 2, 3 | Pure off-node increasing peer count |
| 2 nodes × 4 GPUs | 1, 2, 4 | Pure off-node with many GPU endpoints |
| 3 nodes × 4 GPUs | 1, 4, 8 | Pure off-node intermediate scale |
| 4 nodes × 4 GPUs | 1, 4, 8, 12 | Up to 12 off-node peers per GPU |

Start with `--source-pattern=contiguous --target-layout=remote-only`, so any difference is primarily communication/receiver overhead rather than difficult pack access. Then introduce strided/permuted sources and local copies one at a time. Run both equal **per-peer** sizes (1, 8, 64 KiB) and equal **total** remote volume (64 KiB / PE), without pooling those two metrics or mixing physical placements. Use three independent MPI launches and multiple orderings; test IBRC and CPU-handler IBGDA on actual physical nodes.

## Correctness and acceptance

- CPU graph tests demonstrate non-contiguous rank ordering, PPN=1/4, 2/3/4 nodes, multi-degree bijective in/out graphs, fixed-total and per-peer modes, all three source access patterns, interleaved/shuffled target layouts, no unexpected rank-0 fanout, deterministic and unique requested GIDs.
- During the actual MPI run, compare the **constructed Importer's real** export PIDs/owner-local source LIDs and received counts against graph expectations. Also verify per-peer cross-node bytes from P6.2 plan after the Import.
- Validate same input values and output equality against stock `Tpetra::doImport()`; the `--verify-every=1` setting checks every iteration with epoch-specific values.
- Compare completed GPU Imports first, then include the required **start-of-generation MPI barrier** when evaluating P6.2 as an algorithm. A barrier-excluded P6.2 value that beats MPI is **not yet a production replacement**.
- Once the new traffic benchmark establishes the relevant communication regimes, reuse exactly these Maps to evaluate P7.2 (receiver), P7.3 (NBI+quiet+notification), and later a barrier-free protocol. Do not conflate topology, storage allocation changes and completion protocol modifications in the same experiment.

## Files and limitations

Added: `SyntheticImportGraph.hpp`, `NvshmemImportTopologyBenchmark.cpp`, `test_synthetic_import_graph.cpp`, `run_import_topology_sweep.sh`, `analyze_import_topology.py`, `test_analyze_import_topology.py`, this README. Modified: `CMakeLists.txt` to register the separate executable/optional local-only CTest, `analyze_p71_profiles.py` and its fixture test to accept both `matrix` and `topology` without merging dissimilar workloads, and source-file manifest/active README.

**Hardware validation pending:** this container has `g++` but not the project's native Trilinos, NVCC, MPI or NVSHMEM build. The verified work here is host-side logic/testing, launcher dry run, syntax/static checks, and patch application against the supplied P7.1 archive. It is not a claim of successful native compilation or measured speedups.
