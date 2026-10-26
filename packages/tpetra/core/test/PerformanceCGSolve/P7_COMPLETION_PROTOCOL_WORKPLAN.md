# P7 workplan — Separate payload transport from receiver completion

**Project:** NVSHMEM-powered Tpetra `doImport` replacement (double, single RHS initially)
**Input source revision:** `nvshmem-pack-latest.tar(8).xz`, October 25, 2026
**State:** Workplan only. P7 transport/notification alternatives are **not implemented** by the accompanying cleanup patch.
**Fixed reference:** P6.2 descriptor-based sender/receiver plan and current `--nvshmem-p62` completed-Import benchmark.

## Decision to be answered

Can an NVSHMEM-backed Import **beat** the ordinary Tpetra/MPI Import end-to-end, with the same Map, values, correctness, and completion guarantees, on 8 PEs across two physical nodes? If not, the added code and persistent NVSHMEM allocations are not justified for this workload. Do not optimize only the GPU PUT submission phase.

Earlier experiments established that scalar GETs cost roughly 1.4× the Tpetra Import in the user's tests. The later descriptor PUT study reached ~71.2 µs for block/2048 without IBGDA versus ~63.8 µs for the *same-run* Tpetra Import (in the earlier P6.2 data, not bundled in the latest source archive). P6.2 also uses an **MPI_Barrier before each Import** to prevent previous-generation inbox reuse, outside the reported Import timer. These are a reference point, not proof that a different notification protocol will improve runtime.

The core question is whether the **number of communication operations**, **number of delivery notifications**, and **number of CUDA threads assigned to packing** should be three independent choices. Currently each descriptor's PUT, signal and receiver test are tied together.

## Safety invariants for every P7 variant

1. Plan/Map semantics: destination values match Tpetra for identity, permute, sparse-mixed, one-way, uneven, zero-owner, all-empty, multi-owner, fanout and generated/user-provided matrix cases. Preserve GID order, duplicates, nonidentity rank→PE mapping and arbitrary source/destination LIDs.
2. Receiver observability: **a notification cannot be observed until all payload PUTs it purports to cover have reached that receiver's inbox**. A kernel completing locally, a source buffer becoming reusable, or a CUDA event alone does not prove this.
3. Operation ownership: no `nvshmem_fence()` or quiet in a randomly selected GPU thread is presumed to order other blocks' PUTs. Establish the exact NVSHMEM/CUDA issuing scope and happens-before relationship for each proposed aggregation strategy using **NVSHMEM 3.8.0 installed headers and runtime tests**.
4. Inbox lifetime: while the experimental barrier remains, each PE completes its incoming unpack before entering the next epoch's MPI barrier. When the barrier is eliminated, replace it with an explicit receiver-consumed credit (initially per peer). Ready != consumed.
5. Symmetric source scratch cannot be modified while an outstanding NBI operation still depends on it. Keep persistent symmetric buffers and generation-stamped ready slots; no per-Import allocations, metadata exchange or duplicate/stale signal observations.
6. CUDA progress: device waiters and multiple streams must actually make progress without deadlock. P6.2's bounded receiver polling is a performance experiment, not a universal scheduling guarantee. Retain watchdog timeouts and check required NVSHMEM launch semantics.
7. **No MPI payload transfers** in NVSHMEM variants. MPI may be used for setup, experimental barrier-delimited epochs and oracle verification, with timing boundaries annotated. Never pass the NVSHMEM VMM-allocated symmetric heap to MPI.

## Protocol variants in controlled order

Keep the **same** export maps, descriptor sizes, ordered source LIDs, symmetric pack/inbox capacities, rank binding, warmups and Tpetra oracle. Only change one mechanism at a time.

| ID | Data operation | Receiver-delivery completion | Purpose / correctness condition |
|---|---|---|---|
| B0 — frozen P6.2 | Per-descriptor `put_signal` (thread/warp/block) | One independent ready slot and test per PUT | Functional/timing baseline, fixed throughout |
| B1 — receiver-only control | Same PUT-with-signal as B0 | Check complete set of expected descriptor slots before unpacking | Isolate receiver polling and scatter scheduling **without** changing signal traffic. Confirm device test-all API and resource/progress behavior |
| B2 — NBI + stream quiet (priority) | Fused per-descriptor `put_nbi` on producer stream | **Producer kernel completes → `nvshmemx_quiet_on_stream` in the same stream → GPU notification kernel sets one ready epoch per destination PE**. Receiver tests one slot per source PE and unpacks peer's entire packet | Test whether fewer signals outweigh the quiet and extra notification kernel. Stream quiet must cover the GPU-initiated PUTs. Use a separate producer submission/completion trace |
| B3 — block/warp-group completion | Each participating group issues multiple `put_nbi` packets for **one destination PE** | Group-local sync, correctly scoped quiet/order, one notification for the group's group of chunks. Receiver expects one signal per group | Avoid a grid-wide join but retain parallel PUT submission. Must prove source lifetime and NVSHMEM group completion semantics; may introduce per-group quiet overhead |
| B4 — fence + peer notification (conditional) | `put_nbi` followed by ordered notification | One per destination PE **only after a demonstrated cross-issuer ordering mechanism**. A fence issued by one arbitrary block is **not** sufficient merely because the kernel launched other blocks | Evaluate ordering-only approach **only after** a microtest proves all participating PUTs precede the signal; otherwise reject this variant |
| B5 — transfer-granularity control | Cooperatively pack in warp/block; transmit with fewer larger `put_nbi` calls | Use B2's known-correct peer completion | Separate GPU packing width from NIC submission width. Compare e.g. 1 / 4 / 16 descriptors per peer independently of packing team width |

B2's initial implementation can legitimately use **three kernels plus a stream-quiet operation** (producer, notifier, receiver), rather than preserving the exact two-kernel P6.2 count at the expense of an invalid completion proof. Minimize kernel count *after* establishing a correct, faster protocol. The signal-setting notifier can be `nvshmemx_signal_op` invoked in device code; verify the actual CUDA signatures on the installed NVSHMEM 3.8.0. Alternative host-stream signaling must likewise be checked against the installed headers rather than assumed available.

**Alternative low-level feature:** NVSHMEM 3.8 counted-write signaling is directly relevant to aggregate completion, but NVIDIA documents it as requiring CFT counted-operation support and no fallback to ordinary IBRC. Treat it as a separate capability-gated research branch, not a substitute for B2 on the current IBRC/CPU-assisted IBGDA setup. Likewise, application-controlled RMA regions should be a separate experiment after basic NBI completion is understood.

## Milestones / deliverables / gates

### P7.0 — Repository cleanup and frozen baseline

- Tag the current source branch before archiving legacy entry points. Use the supplied **dry-run-first** `git_rm_legacy_nvshmem.sh` and the `P7_SOURCE_CLEANUP.patch`; do not alter actual P6.2 payload kernels or plans.
- Keep ordinary CG solver, matched Tpetra Import benchmark, P6.2 Map scenarios, standalone packed-plan tests, and the existing P6.2 run script. Remove obsolete P0–P6.1 CLI modes and historical CTest sweep inclusion.
- Document transitive dependencies. **Do not delete P0–P4, P6 or the original `NvshmemPackedPut.hpp` yet**; active P6.2 transitively calls their utility functions. A subsequent extraction step can shrink them safely.
- Gate: patch applies to the exact uploaded source snapshot; host tests and static include-closure checks pass; a cluster rebuild proves the stock CG path and P6.2 paths remain functional before beginning P7 kernels.

### P7.1 — Completion cost measurement / no algorithm change

- Add instrumentation around send-side pack, NBI/PUT issue, host/on-stream quiet, ready polling, scatter, and final GPU synchronization, without placing per-transfer host fences in the hot path.
- Quantify `K = {512, 1024, 2048}` doubles, warp and block variants on `inter8`; include `inter2` for a mostly network-oriented control. Keep IBRC and IBGDA-on/CPU-handler separately. Confirm actual NIC-handler selection.
- Gate: show paired full-Import latency and a GPU timeline explaining where the difference between B0 and MPI arises (receiver wait, signal updates, kernel launch, network, local work, or barrier).

### P7.2 — Receiver-only change (B1)

- Keep identical B0 `put_signal` traffic. Test alternate receiver scheduling only: independent tests versus an all-descriptors-completed predicate, then scatter. If test-all would serialize processing, record that explicitly.
- Gate: correctness and no progress regression on same-node, inter2, inter8. Decide if receiver scheduling is a meaningful independent bottleneck before introducing NBI.

### P7.3 — First aggregate protocol (B2): NBI → quiet → per-peer signal

- Reuse P6.2's sorted descriptor mapping and persistent symmetric buffers, but provide per-peer delivery slots (exactly one writer/source-peer per slot/generation).
- Sender on one stream: fused pack + `put_nbi` kernel → `nvshmemx_quiet_on_stream` → small device notification kernel that sets the generation for each destination PE. Receiver on a separate stream tests expected sources, reads inbox after ready, scatters and exits. The experiment still keeps the start-of-epoch MPI barrier; it is a buffer-reuse simplification only.
- Validate a delayed last PUT, repeated source generations, asymmetric peers, all-empty plans and partial/complete absent peers. Check that late ready cannot advance early. Test stalled producer, receiver scheduling, and on-stream quiet progress under timeout.
- Gate: end-to-end completed Import (not just issuer time) consistently improves B0, and preferably reaches or beats the matched Tpetra measurement **with** protocol costs accounted for.

### P7.4 — Fine-grained completion without serializing the entire sender (B3/B4)

- If B2 improves but loses too much time in global sender quiet, try group-level completion with explicit NVSHMEM issuer ownership and group-local synchronization. Test ordered fence/signal **only after** proving ordering across all covered operations.
- If the group model introduces its own quiet or notification bottleneck, reject it; don't accumulate clever but slower mechanisms.
- Gate: repeated independent passes and end-to-end improvement beyond reference drift.

### P7.5 — Separate packing parallelism from transfer granularity (B5)

- Hold best *correct* completion protocol constant, then vary work distribution: 1 warp/block packs a chunk while another group/subset issues NBI PUTs of larger contiguous regions, versus 1 warp/block issuing each chunk.
- Control number of PUTs and total bytes separately. Avoid assume a block collective provides extra NIC issue parallelism when transport uses CPU-assisted IBGDA or IBRC proxy.
- Gate: identify whether packing, communication submission, or receiver scatter is the dominant remaining term; record A/B benchmark and NIC handler.

### P7.6 — Remove experimental MPI barrier and make the operation usable in CG/SpMV

- Introduce peer-level **consumed** credits (not one credit per descriptor) or generation-indexed ring inboxes. Receiver publishes credit only *after* its final inbox read, and a sender can reuse that peer's inbox only after credit for previous generation. No world rendezvous and no MPI payload.
- Preserve independent peer progress, no deadlock from listener kernels, and no reuse of source scratch while NBI operations are in flight.
- Evaluate end-to-end `CrsMatrix::apply()`/CG against existing MPI with no artificial per-iteration MPI barrier on either method. Use multiple iterations and real solver workloads.
- **Go/no-go:** demand a statistically repeatable performance advantage over standard Tpetra/MPI on representative communication-bound cases. If not faster end-to-end, document why and stop protocol work rather than extend tuning indefinitely.

## Measurement protocol

1. **Isolation study (P7.1–P7.5):** barrier-delimit both alternatives equally for matched start times; report the experimental barrier separately and keep an additional full-generation measurement **including the barrier** when relevant to actual operation cost. Never claim a barrier-excluded result is a deployable faster Import.
2. **Deployment study (P7.6):** after removing the barrier from P7, time full *usable* Import and `CrsMatrix::apply` with stock Tpetra without adding a barrier to MPI. Same inputs/Maps, same GPU residency, allocated scratch amortized, no hidden host/device extra copy for one method.
3. Rank metric: per-iteration `max(rank_time)` with median, p10, p90 and max; retain raw rank/epoch CSVs. At least 3 independent MPI launches and reversed/randomized order for comparisons; never count 50 epochs within one MPI launch as 50 independent replication samples.
4. Record total remote bytes, distinct sending and receiving peers, PUT count, signal count, number of quiet/fence operations, actual remote cross-node payload, source/destination memory type, pack/unpack bytes, GPU kernel launches and synchronization calls.
5. Profile with Nsight Systems (and NVSHMEM transport diagnostics when available) to prove or reject overlap. MPI barrier time, kernel submissions, remote completion and observed readiness must be distinct numbers.
6. Configuration controls: preserve device selection, rank placement, actual node names, NIC handler, NVSHMEM version, GPU SM capacity, batch/RC QP defaults (unset for baseline), and repeat on inter2 then inter8. Give timeouts to signal-wait experiments.
7. Correctness gate: verify every epoch in the short stress runs before timing with less frequent verification. No wrong or stale unpack values, lost peer notifications, or partial completion passing as ready.

## Lightweight source structure after P7.0, then later extraction

Initial P7.0 cleanup intentionally leaves the legacy utility include chain:

```
cg_solve_file.cpp
  NvshmemP62.hpp
    NvshmemP6.hpp             # timing / MPI sample aggregation
    NvshmemP4.hpp             # synthetic Maps, pattern, topology
      NvshmemP3.hpp -> NvshmemP2.hpp -> NvshmemP1.hpp -> NvshmemP0.hpp
    NvshmemPackedPutP62.hpp
      NvshmemPackedPut.hpp    # cudaCheck and NvshmemP3::upload
      PackedPutP62Plan.hpp -> PackedPutPlan.hpp
    PackedPutP62PlanMpi.hpp
```

Subsequent **P7.0b, separately compiled and reviewed**, should factor stable functions into narrow headers:

- `NvshmemImportTestUtils.hpp`: topology, rank-aware source pattern, synthetic Map fixtures, Tpetra oracle.
- `NvshmemImportPlan.hpp`: authoritative P5.1 plan and P6.2 chunks with a single metadata reconciliation API.
- `NvshmemImportCudaUtils.hpp`: checked CUDA calls, upload and tiny Kokkos helpers.
- `NvshmemImportBenchmark.hpp`: sample gathering, max-rank medians, comparison/reporting.
- `NvshmemImportCompletion.hpp`: pluggable per-descriptor, per-peer, and group completion policy (P7 work, not preemptively implemented in cleanup).

Once these are genuinely self-contained and the build/stress suite passes, older P2/P3/P4/P6 experiment bodies can be deleted without breaking the active P6.2 and P7 harness. Avoid doing extraction and deletion in the same unvalidated large patch.

## Sources / assumptions

- Latest uploaded source `nvshmem-pack-latest.tar(8).xz`, including `NvshmemP62.hpp`, `NvshmemPackedPutP62.hpp`, `PackedPutP62Plan*.hpp`, `cg_solve_file.cpp`, CMake and the design handoffs.
- Earlier conversation's P6.2 IBGDA on/off benchmark analysis (timing figures described above) is **not contained in this source archive**.
- NVIDIA NVSHMEM 3.8 API docs: [GPU/stream ordering](https://docs.nvidia.com/nvshmem/api/latest/api/overview.html), [PUT/NBI operations](https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html), [put-with-signal and counted-write restrictions](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html), [CUDA deadlock warnings](https://docs.nvidia.com/nvshmem/api/latest/cuda-interactions.html).

**No new P7 kernels are written by the cleanup patch.** This document is a handoff and implementation order, not a validated alternative to current P6.2.
