# P5.2 — Point-to-point, generation/credit NVSHMEM packed PUT

**Status: SOURCE IMPLEMENTED; ACTUAL CUDA/NVSHMEM MULTI-NODE ACCEPTANCE PENDING.**

This implementation supersedes the rejected *barrier-based* P5.2 sketch.
Apply the patch to the supplied **P5.0-complete/P5.1** Trilinos checkout.
Older modes P0–P5.1 remain independently selectable. Nothing in this
implementation needs a global barrier, MPI request or collective GPU kernel
launch **per Import**. The only collective steps are: one-time P5.1 metadata,
NVSHMEM symmetric memory allocation/free on each setup/retirement cycle,
plus MPI reference/test reporting external to the PUT transport.

## Data structures (from verified P5.1)

Each nonempty outgoing peer has a disjoint contiguous source packet; each
nonempty incoming peer owns an exclusive, contiguous segment of the symmetric
receiver inbox. All source LIDs, column LID scatter indices, pack offsets,
receiver inbox offsets and MPI rank->NVSHMEM PE translations come unchanged
from `PackedPutPlan.hpp` and `PackedPutPlanMpi.hpp`.

P5.2 adds **two different symmetric 64-bit signal arrays** of `numPes`
elements, allocated on every PE in a globally identical collective order:

- `ready[sourcePe]` at the **receiver**, updated only by that sender's
  `nvshmem_double_put_signal(..., ready + sourcePe, g, NVSHMEM_SIGNAL_SET,
  receiverPe)`. Once `ready[sourcePe] == g`, that sender's packet has arrived.
- `credit[receiverPe]` at the **sender**, updated only by that receiver's
  `nvshmemx_signal_op(credit + receiverPe, g, NVSHMEM_SIGNAL_SET,
  senderPe)` **after the GPU has finished unpacking the inbox segment**.
  Once `credit[receiverPe] == g`, that receiver has consumed the packet.

The two signals cannot be conflated. There is exactly one writer for each
remote slot, so there is no shared critical section and no need for a CAS
lock. `nvshmem_calloc` initializes both arrays to zero, with the first
valid generation being **1**. Generation monotonicity and overflow are
checked; allocation/free cycles reset generation to 1 only after a
collective deallocation/zeroed reallocation of the signals.

## Steady-state Import (no global synchronization)

```text
Sending PE A                                      Receiving PE B
-----------                                       --------------
For generation g>1:
  wait_on_stream(credit_A[B] == g-1)  <----------- ACK(g-1), after unpack
  cudaStreamSynchronize(control stream)
Kokkos pack owned X[exportLIDs] into symmetric packet
Kokkos GPU PUT_SIGNAL(packet,g) ----------------> symmetric inbox segment
                                                 ready_B[A] = g (with PUT)
...
Kokkos PUT for ALL A's outgoing peers finished
wait_on_stream(ready_A[sourcePE] == g)            ... B does same for its peers
  for each actual incoming source PE
cudaStreamSynchronize(control stream)
Kokkos GPU inbox-to-column scatter and local copies
Kokkos GPU signal_op(credit_source[A],g) --------> sender's credit slot
Return: all this PE's expected incoming values are correct
```

The local GPU pack and send are both **enqueued/completed before any ready
wait on the same rank**. The stream-based wait is the NVSHMEM host API
`nvshmemx_signal_wait_until_on_stream`, not a CUDA kernel calling the device
`nvshmem_signal_wait_until`. Thus **no NVSHMEM collective CUDA launch** is
necessary and the destination PE need not be executing any kernel when the
signal arrives. A local `cudaStreamSynchronize(0)` after the ready/credit
waits orders that stream's completion before subsequent Kokkos CUDA access;
it is NOT a PE collective. Ordinary Tpetra vectors remain in normal CUDA
space; MPI never touches a symmetric allocation for payload movement.

**Inbox safety proof:** The receiver's ACK(g) is issued only after all
GPU reads of its g packet have completed. The sender waits for ACK(g)
before issuing PUT(g+1) into the same inbox segment. Hence PUT(g+1)
cannot overwrite any packet storage being consumed by that receiver.

**Deadlock reasoning:** For a given g, all sends are issued *before* the
same PE's ready waits. Waiting on credits for g-1 cannot depend on an
unissued g PUT: each required ACK(g-1) is produced after the peer has
received all of its previous-generation PUTs, which were already issued.
PEs not participating in a relationship neither send nor wait for it.
This is a single-slot, per-neighbor pipeline; a slow destination may
backpressure its senders, but cannot stop unrelated peers as a global
rendezvous would.

**Sender scratch safety:** GPU `put_signal` is blocking with respect to
reusing the local source once the issuing threads have returned. A
Kokkos CUDA fence finishes the local sending kernel. The receiver's
completion signal establishes remote payload visibility. It is distinct
from the receiver-consumed credit.

**Remaining collectives:** NVSHMEM `malloc/calloc/free` are *by API
contract* collective and include implicit barriers. They run only during
setup/retirement, not per Import. `Exchange::release()` finishes all local
CUDA work and uses `nvshmemx_quiet_on_stream` before collective retirement
to drain device-issued ACKs. The correctness test separately uses MPI
`Allreduce` for static packet counts and final reports, plus standard
Tpetra/P2 oracles. None is invoked inside the `Exchange::import` transport.

## New files and controls

- `NvshmemPackedPut.hpp`: the reusable packet exchange, symmetric allocations,
  CUDA/Kokkos GPU packing, GPU PUTs, point-to-point on-stream waits, GPU
  scatter/local copy and GPU credit acknowledgment.
- `NvshmemP52.hpp`: the independent correctness suite with changed binary64
  values, complete destination-LID checks against stock `doImport`, ordinary
  CUDA-space P2, and per-peer/operation statistics.
- `test_packed_put_plan.cpp`: includes the original P5.1 deterministic plan
  tests plus an *adversarial, asynchronously interleaved rank scheduler* for
  the P5.2 ready/credit protocol over 900 generated distributions. This is a
  CPU-side protocol model; it cannot replace a real GPU/InfiniBand test.
- `cg_solve_file.cpp` and `CMakeLists.txt`: `--nvshmem-p52`, optional
  `PERFORMANCE_CGSOLVE_NVSHMEM_P52_CTEST=ON`, and configuration switches.

Controls: `--nvshmem-p52-case={all,matrix,identity,permute,sparse-mixed,
one-way,uneven,zero-owner,all-empty,multi-owner,fanout}`;
`--nvshmem-p52-epochs=4` (minimum 4);
`--nvshmem-p52-cycles=2` (minimum 2);
`--nvshmem-p52-min-nodes=1`;
`--nvshmem-p52-max-nodes=0` (unlimited);
`--nvshmem-p52-cross-node`;
`--nvshmem-p52-peers` / `--no-nvshmem-p52-peers`;
`--nvshmem-p52-skew-ms=0` (optional, 0–3000).

With `--nvshmem-p52-skew-ms=250`, rank 0 deliberately delays entering its
first PUT Import *after* both MPI reference oracles. Another PE may PUT
and signal rank 0 while rank 0 is sleeping. This directly checks the
premise that **signal reception does not require an active receiving kernel**.
Run this once with a topology having rank 0's actual incoming packets.

## Host-side checks completed with this source

```bash
cd packages/tpetra/core/test/PerformanceCGSolve
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
    test_packed_put_plan.cpp -o /tmp/p52_plan_test
/tmp/p52_plan_test
```

The model tests 1, 2, 3, 4, 8, and 16 simulated ranks; empty/one-way
traffic, arbitrary LIDs and rank-to-PE permutations, source duplicates,
packet ordering, fanout, two independent zeroed-signal allocation cycles
and four changed-value generations per cycle. Each PE advances through
credit wait, pack, PUT, ready wait, unpack and ACK independently in a
randomized adversarial schedule. Stale signals and premature inbox reuse
are explicitly rejected. These host tests passed GCC, Clang and GCC
AddressSanitizer/UndefinedBehaviorSanitizer. A mocked NVSHMEM/CUDA API
single-PE compile and eight imports also passed GCC/Clang.

## GPU acceptance (still required)

Use the same build, Kokkos CUDA Rdc settings, GPU binding script, and
NVSHMEM/MPI library installation already tested for P5.1. Confirm that the NVSHMEM implementation installed on the cluster exposes
`nvshmemx_signal_wait_until_on_stream` and `nvshmemx_signal_op`, and link
the required NVSHMEMx/device symbols. Preserve the established environment:

```bash
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=1
# Do not force NVSHMEM_IBGDA_NIC_HANDLER=gpu on this system.
EXE=./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe
BIND="$HOME/cronus/gpu_bind.sh"

# Single GPU, one rank: local-only and all-empty paths, zero incoming waits.
mpirun -np 1 "$BIND" "$EXE" --cuda --numgpus=8 --size=8 --nvshmem-p52 \
  --nvshmem-p52-max-nodes=1

# Intra-node, two/four ranks:
mpirun --map-by ppr:2:node --bind-to none -np 2 "$BIND" "$EXE" \
  --cuda --numgpus=8 --size=8 --nvshmem-p52 --nvshmem-p52-max-nodes=1
mpirun --map-by ppr:4:node --bind-to none -np 4 "$BIND" "$EXE" \
  --cuda --numgpus=8 --size=8 --nvshmem-p52 --nvshmem-p52-max-nodes=1

# Inter-node, one PE per node:
mpirun --map-by ppr:1:node --bind-to none -np 2 "$BIND" "$EXE" \
  --cuda --numgpus=8 --size=8 --nvshmem-p52 \
  --nvshmem-p52-min-nodes=2 --nvshmem-p52-cross-node

# Inter-node, eight PEs per node: 16 PE miniFE size=100 + all 9 Map cases:
mpirun --map-by ppr:8:node --bind-to none -np 16 "$BIND" "$EXE" \
  --cuda --numgpus=8 --size=100 --nvshmem-p52 \
  --nvshmem-p52-min-nodes=2 --nvshmem-p52-cross-node

# Explicit arrival-before-receiver test:
mpirun --map-by ppr:8:node --bind-to none -np 16 "$BIND" "$EXE" \
  --cuda --numgpus=8 --size=100 --nvshmem-p52 --nvshmem-p52-case=matrix \
  --nvshmem-p52-min-nodes=2 --nvshmem-p52-cross-node \
  --nvshmem-p52-skew-ms=250
```

Sample placement flags assume allocation of matching physical nodes and
8 visible CUDA devices per node. Adjust to the established cluster launcher.
**No real GPU/CUDA/NVSHMEM test has been run while authoring this artifact.**

Expected for the *previously verified* 16-PE/2-node `--size=100` matrix:

| Metric | Per Import | Default 8 Imports |
|---|---:|---:|
| Logical GPU PUT+signal calls | 44 | 352 |
| Logical PUT doubles | 309,105 | 2,472,840 |
| Logical PUT bytes | 2,472,840 | 19,782,720 |
| Receiver ready waits | 44 | 352 |
| Receiver consumption ACKs | 44 | 352 |
| Sender previous-generation credit waits | 44 (epochs 2–4 only) | 264 |
| Cross-node PUT calls (8 PEs/node) | 10 | 80 |
| Cross-node logical bytes | 164,912 | 1,319,296 |
| **Global synchronization calls in `Exchange::import()`** | **0** | **0** |

The ready/ack/credit counts refer to logical peer operations, not NIC wire
packets. The matrix's per-import peer counts can change with a different
matrix generation/distribution or Map.

The expected acceptance line includes:

```text
[NVSHMEM P5.2] SUMMARY status=PASS casesPassed=10 casesSkipped=0 \
  ... globalSyncsInImport=0 sync=POINT_TO_POINT_GENERATION_CREDIT
```

**P5.2 is not accepted until the native CUDA/NVSHMEM build and all relevant
P4 acceptance topologies pass, including the skew test.** No throughput or
speedup is claimed here. P5.3 can optimize independent per-peer progress,
reduce local CUDA fencing, use additional inflight slots, and measure the
credit/ACK overhead without changing this verified protocol's correctness.

## NVSHMEM references

- [Signaling and put_signal](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html)
- [Point-to-point signal wait on stream](https://docs.nvidia.com/nvshmem/api/latest/gen/api/sync.html)
- [CUDA stream and device ordering](https://docs.nvidia.com/nvshmem/api/latest/api/overview.html)
- [Symmetric allocation/free collective semantics](https://docs.nvidia.com/nvshmem/api/latest/gen/api/memory.html)
