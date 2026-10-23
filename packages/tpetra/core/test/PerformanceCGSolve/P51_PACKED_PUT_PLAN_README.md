# P5.1 — Sender-packed PUT planning (no payload communication yet)

This milestone constructs and validates the **general** one-packet-per-peer sender
and receiver plan for the Tpetra Import operation. The `--nvshmem-p51` mode
executes on the existing CUDA/NVSHMEM-enabled `Performance-CGSolve` program,
but **does not issue NVSHMEM RMA, allocate any symmetric data buffers, run a
CUDA packing kernel or measure Import throughput**. P5.2 will consume this
plan for `nvshmem_double_put_signal` + per-source completion waits + unpack.

## Files

- `PackedPutPlan.hpp`: STL-only `buildLocal` and `finalizeLocal`, input validation,
  stable per-peer packing, disjoint inbox allocation, exact unpack scatter.
- `PackedPutPlanMpi.hpp`: Tpetra P1 adapter and **one-time** MPI metadata
  exchange of export source LIDs, per-peer counts, receiver inbox offsets and
  collective symmetric buffer capacities. **No double payload via MPI**.
- `NvshmemP51.hpp`: P4 Map/test battery in a metadata-only P5.1 runner, with
  ordered sender/receiver and global packet accounting.
- `test_packed_put_plan.cpp`: C++17 standalone tests (no GPU, MPI, Tpetra).
- `cg_solve_file.cpp`, `CMakeLists.txt`: new opt-in mode and optional 1-PE CTest.

## Exact plan contract

Each sender keeps the **original relative order** of Tpetra's `exportLIDs`
for each destination rank, grouping those exports into one contiguous packet
per destination **NVSHMEM PE**. Grouping does not assume rank-number order,
adjacent owner-local LIDs or contiguous source segments.

Each receiver groups its P1 `(ownerRank, sourceLID, destinationLID)` entries by
actual source PE, assigning a nonoverlapping contiguous inbox segment to each
source. A one-time `MPI_Alltoallv` metadata-only transfer conveys the sender's
actual per-peer **ordered source LID list**. The receiver matches *each*
source LID occurrence against the P1 remote owner-LID lookup to build the
ordered `inboxSlot -> targetColumnLID` scatter array. Repeated source LIDs
within the same sender/receiver relationship use stable occurrence matching;
fanout to different receivers creates distinct outgoing packets.

Each receiver tells each sender its inbox segment offset using a metadata-only
`MPI_Alltoall`. All participants check counts, exact LID membership and
multiplicity, slot coverage, segment bounds and Map-local destination coverage.
Peer and signal addresses are translated using the verified
MPI-rank-to-NVSHMEM-PE permutation, not a numeric identity assumption.

Every rank computes the **same symmetric allocation capacities** using
collective `MPI_MAX`: `max(1, maxRankPackedDoubles)`,
`max(1, maxRankInboxDoubles)` and `numPes` 64-bit signal slots. P5.1
only computes those capacities, and `PackedPutPlan.hpp` checks element-to-byte
conversion overflow; P5.2 will allocate the actual symmetric memory in a
common collective order.

This plan **does not** change the Importer's packets, Tpetra domain/column
Maps, MPI Distributor, stock MPI/GPU payload path, or existing P0–P5.0 modes.

## Independent host tests

```sh
cd packages/tpetra/core/test/PerformanceCGSolve
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
    test_packed_put_plan.cpp -o /tmp/test_packed_put_plan
/tmp/test_packed_put_plan
```

The tests cover all-empty and zero-owned ranks; one-way sends; fanout and
repeat source requests; stable per-peer export order under shuffled global
exports; arbitrary destination interleaving; nonidentity PE mapping;
randomized 1, 2, 3, 4, 8 and 16-rank distributions; per-peer count and source
LID corruption; bad rank/PE assignments; duplicate destination LIDs; inbox
overrun and symmetric allocation overflow.

## GPU-cluster plan-only acceptance (required to complete P5.1)

This mode uses the current CUDA/NVSHMEM build **for bootstrap only**. Preserve
the GPU affinity launch script and compiler/library settings that passed P4.
Example invocations, assuming the environment already has the same bindings:

```sh
EXE=./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe
# smoke1; include all applicable synthetic cases
mpirun -np 1 "$EXE" --cuda --numgpus=8 --size=8 --nvshmem-p51 --nvshmem-p51-max-nodes=1
# intra2 and intra4: ensure one node, with a distinct GPU per PE
mpirun --map-by ppr:2:node --bind-to none -np 2 "$EXE" --cuda --numgpus=8 --size=8 --nvshmem-p51 --nvshmem-p51-max-nodes=1
mpirun --map-by ppr:4:node --bind-to none -np 4 "$EXE" --cuda --numgpus=8 --size=8 --nvshmem-p51 --nvshmem-p51-max-nodes=1
# inter2 and inter16: two nodes, require actual inter-node packet plans
mpirun --map-by ppr:1:node --bind-to none -np 2 "$EXE" --cuda --numgpus=8 --size=8 --nvshmem-p51 --nvshmem-p51-min-nodes=2 --nvshmem-p51-cross-node
mpirun --map-by ppr:8:node --bind-to none -np 16 "$EXE" --cuda --numgpus=8 --size=100 --nvshmem-p51 --nvshmem-p51-min-nodes=2 --nvshmem-p51-cross-node
```

Adjust `--numgpus` to the **per-node visible** CUDA device count, use your
established binding wrapper as needed, and verify placement. The sample
`--map-by ppr:...` commands presume matching node allocations. Use
`--nvshmem-p51-case=matrix` for a miniFE-only run and
`--no-nvshmem-p51-peers` to suppress individual peer records.

Expected for the previously observed `--size=100`, 16-PE miniFE workload:
**309,105 sent/received doubles** and **44 nonempty sender/receiver PE
relationships**, provided the matrix generation/distribution is unchanged.
The new actual source-LID-to-destination-LID verification is the main P5.1
acceptance, not this known-workload operation count.

Check for the final line:

```
[NVSHMEM P5.1] SUMMARY status=PASS ... planOnly=YES rmaIssued=0 performanceMeasured=NO
```

The P5.1 hardware acceptance is **not established** until all applicable P4
layout cases pass at smoke1/intra2/intra4/inter2/inter16. No speedup is
required, and no NVSHMEM PUT has yet occurred. P5.2 must separately validate
generation signaling, payload delivery and safe buffer reuse.

## Source snapshot caveat

The supplied October 23 archive includes `cg_solve_file.cpp` with a
`#include "NvshmemP50.hpp"`, but does **not** include that P5.0 header.
Apply this P5.1 overlay to your **working P5.0 Trilinos checkout**, retaining
its already accepted `NvshmemP50.hpp`; P5.1 does not provide or replace it.
