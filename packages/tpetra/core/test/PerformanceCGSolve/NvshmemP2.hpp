// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
//
// P2 correctness and timing driver. Keep P0/P1 and the stock CrsMatrix intact.
// No NVSHMEM pointer is given to the Distributor/MPI. No GID handshake or
// independent MPI ordering is needed; P1 already validated the Import plan.

#ifndef TPETRA_PERFORMANCE_NVSHMEM_P2_HPP
#define TPETRA_PERFORMANCE_NVSHMEM_P2_HPP

#include "StandaloneMpiImport.hpp"
#include "NvshmemP1.hpp"
#include "Teuchos_OrdinalTraits.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace NvshmemP2 {

// Retain the already-integrated command-line structure. The default is now
// CUDA-memory Distributor communication, not host staging. The legacy
// --no-nvshmem-p2-gpu-aware-mpi option is rejected rather than silently
// requesting an unsupported staged pathway.
struct Options {
  bool runSynthetic = true;
  bool gpuAwareMpi = true;
  bool dumpExports = false;
  int warmup = 2;
  int repetitions = 10;
};

inline void checkOptions(const Options& opts) {
  StandaloneMpiP2::check(opts.warmup >= 0 && opts.repetitions >= 1,
                         "warmup must be >= 0 and repetitions must be >= 1");
  StandaloneMpiP2::check(opts.gpuAwareMpi,
                         "revised P2 uses Kokkos::CudaSpace buffers with the "
                         "Tpetra Distributor; host-staged MPI mode was removed");
}

template<class LO, class GO, class Node>
void fillOwned(Tpetra::Vector<double, LO, GO, Node>& X,
               const NvshmemP1::ValueEncoding& encoding,
               const NvshmemP0::Session& session) {
  const std::size_t ownedLength = X.getMap()->getLocalNumElements();
  const std::uint64_t base = static_cast<std::uint64_t>(session.mpiRank()) *
                             encoding.stride;
  // The Kokkos view must die before Tpetra doImport or host validation.
  {
    auto owned = X.getLocalViewDevice(Tpetra::Access::OverwriteAll);
    Kokkos::parallel_for(
        "P2 fill rank/source-LID distinct values",
        Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<std::size_t>>(0, ownedLength),
        KOKKOS_LAMBDA(const std::size_t i) {
          owned(i, 0) = static_cast<double>(base + static_cast<std::uint64_t>(i) + 1ULL);
        });
    Kokkos::fence("P2 source Vector initialized");
  }
}

template<class LO, class GO, class Node>
void verifyOracle(const std::string& name,
                  const ImportGatherP1::ImportGatherPlan<LO, GO, Node>& plan,
                  const NvshmemP1::ValueEncoding& encoding,
                  const NvshmemP0::Session& session,
                  const Tpetra::Vector<double, LO, GO, Node>& reference) {
  const auto expected = reference.getLocalViewHost(Tpetra::Access::ReadOnly);
  const int rank = session.mpiRank();
  for (const auto& item : plan.local) {
    const double value = encoding.value(rank, item.sourceLid);
    StandaloneMpiP2::check(expected(static_cast<std::size_t>(item.destinationLid), 0) == value,
                           name + ": Tpetra local reference does not match P1 owner/LID");
  }
  for (const auto& item : plan.remote) {
    const double value = encoding.value(item.ownerRank, item.sourceLid);
    StandaloneMpiP2::check(expected(static_cast<std::size_t>(item.destinationLid), 0) == value,
                           name + ": Tpetra remote reference does not match P1 owner/LID");
  }
}

template<class LO, class GO, class Node>
void assertEqual(const std::string& name,
                 const Tpetra::Vector<double, LO, GO, Node>& reference,
                 const Tpetra::Vector<double, LO, GO, Node>& actual) {
  const auto expected = reference.getLocalViewHost(Tpetra::Access::ReadOnly);
  const auto found = actual.getLocalViewHost(Tpetra::Access::ReadOnly);
  const std::size_t n = reference.getMap()->getLocalNumElements();
  StandaloneMpiP2::check(n == actual.getMap()->getLocalNumElements(),
                         name + ": target lengths differ");
  for (std::size_t lid = 0; lid < n; ++lid) {
    StandaloneMpiP2::check(std::isfinite(expected(lid, 0)) &&
                               found(lid, 0) == expected(lid, 0),
                           name + ": standalone Import mismatch at column-map LID " +
                               std::to_string(lid));
  }
}

struct Quantiles { double median, p10, p90; };
inline double quantile(const std::vector<double>& sorted, double x) {
  const double pos = x * static_cast<double>(sorted.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(pos);
  const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
  const double fraction = pos - static_cast<double>(lo);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * fraction;
}
inline Quantiles summarize(std::vector<double> samples) {
  StandaloneMpiP2::check(!samples.empty(), "no timing samples");
  std::sort(samples.begin(), samples.end());
  return Quantiles{quantile(samples, 0.5), quantile(samples, 0.1),
                   quantile(samples, 0.9)};
}

// The per-iteration maximum time over all MPI ranks is collected for each
// stage. Setup/allocation, metadata creation and correctness are excluded.
template<class Fn>
void measure(const std::string& testName, const std::string& mode,
             const Options& opts, const NvshmemP0::Session& session, Fn action) {
  for (int i = 0; i < opts.warmup; ++i) (void)action();
  static constexpr int stages = 5;
  std::vector<double> local(static_cast<std::size_t>(opts.repetitions * stages));
  for (int i = 0; i < opts.repetitions; ++i) {
    const auto t = action();
    const std::size_t k = static_cast<std::size_t>(i * stages);
    local[k] = t.local;
    local[k + 1] = t.pack;
    local[k + 2] = t.exchange;
    local[k + 3] = t.unpack;
    local[k + 4] = t.total;
  }
  std::vector<double> global(local.size());
  StandaloneMpiP2::check(local.size() <=
                             static_cast<std::size_t>(std::numeric_limits<int>::max()),
                         "too many P2 timing measurements for MPI_Allreduce");
  const int mpiErr = MPI_Allreduce(local.data(), global.data(),
                                   static_cast<int>(global.size()), MPI_DOUBLE,
                                   MPI_MAX, session.comm());
  StandaloneMpiP2::check(mpiErr == MPI_SUCCESS, "MPI_Allreduce(P2 timing)");
  if (session.mpiRank() == 0) {
    static const char* const labels[stages] = {
      "local", "pack", "exchange", "unpack", "total"};
    const auto flags = std::cout.flags();
    const auto precision = std::cout.precision();
    std::cout << std::fixed << std::setprecision(2)
              << "[NVSHMEM P2 TIMING] case=" << testName
              << " mode=" << mode << " scope=max_rank_per_iteration"
              << " unit=us reps=" << opts.repetitions << " warmup=" << opts.warmup;
    for (int stage = 0; stage < stages; ++stage) {
      std::vector<double> values(static_cast<std::size_t>(opts.repetitions));
      for (int iteration = 0; iteration < opts.repetitions; ++iteration)
        values[static_cast<std::size_t>(iteration)] =
            global[static_cast<std::size_t>(iteration * stages + stage)] * 1.0e6;
      const auto q = summarize(std::move(values));
      std::cout << " " << labels[stage] << "_median=" << q.median;
      if (stage == stages - 1)
        std::cout << " total_p10=" << q.p10 << " total_p90=" << q.p90;
    }
    std::cout << "\n";
    std::cout.flags(flags);
    std::cout.precision(precision);
  }
}

template<class LO, class GO, class Node>
void runCase(const std::string& name,
             const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& domain,
             const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& column,
             const Teuchos::RCP<const Tpetra::Import<LO, GO, Node>>& importer,
             const NvshmemP0::Session& session, const Options& opts,
             bool requireRemote, bool benchmark) {
  using Vector = Tpetra::Vector<double, LO, GO, Node>;
  checkOptions(opts);
  StandaloneMpiP2::check(!importer.is_null(), name + ": null importer");
  NvshmemP1::ValueEncoding encoding(domain->getLocalNumElements(), session);

  std::vector<int> peForRank(static_cast<std::size_t>(session.numPes()), -1);
  for (int pe = 0; pe < session.numPes(); ++pe)
    peForRank[static_cast<std::size_t>(session.mpiRankForPe(pe))] = pe;
  const auto plan = ImportGatherP1::buildImportGatherPlan(
      domain, column, *importer, peForRank, session.comm());
  StandaloneMpiP2::ImportReplica<LO, GO, Node> replica(
      domain, column, *importer);

  Vector X(domain);
  Vector X_reference(column);
  Vector X_mpi(column);
  fillOwned(X, encoding, session);
  X_reference.doImport(X, *importer, Tpetra::INSERT);
  verifyOracle(name, plan, encoding, session, X_reference);
  (void)replica.import(X, X_mpi);
  assertEqual(name, X_reference, X_mpi);

  const unsigned long long counts[2] = {
    static_cast<unsigned long long>(replica.remote()),
    static_cast<unsigned long long>(replica.exported())};
  unsigned long long global[2] = {0, 0};
  const int err = MPI_Allreduce(counts, global, 2, MPI_UNSIGNED_LONG_LONG,
                                MPI_SUM, session.comm());
  StandaloneMpiP2::check(err == MPI_SUCCESS, "MPI_Allreduce(P2 packet counts)");
  StandaloneMpiP2::check(global[0] == global[1],
                         name + ": global remote and export counts differ");
  StandaloneMpiP2::check(!requireRemote || global[0] != 0,
                         name + ": expected remote entries but found none");

  if (opts.dumpExports) {
    std::cout << "[NVSHMEM P2 EXPORTS] case=" << name
              << " rank=" << session.mpiRank();
    for (std::size_t k = 0; k < plan.exportLids.size(); ++k)
      std::cout << " (k=" << k << ",pid=" << plan.exportPids[k]
                << ",lid=" << plan.exportLids[k] << ")";
    std::cout << "\n";
  }
  std::cout << "[NVSHMEM P2] case=" << name
            << " rank=" << session.mpiRank()
            << " owned=" << plan.sourceLength
            << " target=" << plan.targetLength
            << " same=" << plan.numSame
            << " permute=" << plan.numPermute
            << " remote=" << replica.remote()
            << " exports=" << replica.exported()
            << " sourcePEs=" << plan.numRemotePeers
            << " remoteBytes=" << (counts[0] * sizeof(double))
            << " exportBytes=" << (counts[1] * sizeof(double))
            << " memory=Kokkos::CudaSpace"
            << " distributor=Tpetra"
            << " status=PASS\n";
  if (session.mpiRank() == 0)
    std::cout << "[NVSHMEM P2] case=" << name
              << " globalRemote=" << global[0]
              << " globalExports=" << global[1]
              << " globalRemoteBytes=" << (global[0] * sizeof(double))
              << " status=PASS\n";

  if (benchmark) {
    measure(name, "tpetra_doImport", opts, session, [&]() {
      StandaloneMpiP2::Timings t;
      const double begin = MPI_Wtime();
      X_reference.doImport(X, *importer, Tpetra::INSERT);
      Kokkos::fence("P2 stock Tpetra import completed");
      t.total = MPI_Wtime() - begin;
      return t;
    });
    measure(name, "cuda_pack_distributor_cuda_unpack", opts, session, [&]() {
      return replica.import(X, X_mpi);
    });
    assertEqual(name + " timed", X_reference, X_mpi);
  }
}

enum class SyntheticKind { Mixed, OneWay, LocalOnly, Interleaved };

template<class LO, class GO, class Node>
void runSynthetic(SyntheticKind kind, const NvshmemP0::Session& session,
                  const Options& opts) {
  using Map = Tpetra::Map<LO, GO, Node>;
  using Import = Tpetra::Import<LO, GO, Node>;
  const int rank = session.mpiRank();
  const int nRanks = session.numPes();
  StandaloneMpiP2::check(
      static_cast<long long>(nRanks) * 1024 + 100 <
          static_cast<long long>(std::numeric_limits<GO>::max()),
      "too many ranks for synthetic noncontiguous GIDs");
  const int nOwned = 3 + rank % 3;
  Teuchos::Array<GO> ownedGids;
  for (int i = 0; i < nOwned; ++i)
    ownedGids.push_back(NvshmemP1::syntheticGid<GO>(rank, i));
  Teuchos::Array<GO> targetGids;
  targetGids.push_back(ownedGids[0]);
  if (nRanks > 1) {
    const int next = (rank + 1) % nRanks;
    if (kind == SyntheticKind::Mixed)
      targetGids.push_back(NvshmemP1::syntheticGid<GO>(next, 1));
    else if (kind == SyntheticKind::OneWay && rank == 0)
      targetGids.push_back(NvshmemP1::syntheticGid<GO>(1, 1));
    else if (kind == SyntheticKind::Interleaved)
      targetGids.push_back(NvshmemP1::syntheticGid<GO>(next, 2));
  }
  targetGids.push_back(ownedGids[nOwned - 1]);
  if (kind == SyntheticKind::Interleaved && nRanks > 1) {
    const int other = nRanks > 2 ? (rank + 2) % nRanks : (rank + 1) % nRanks;
    targetGids.push_back(NvshmemP1::syntheticGid<GO>(other, 1));
  }
  for (int i = nOwned - 2; i >= 1; --i) targetGids.push_back(ownedGids[i]);

  const auto comm = Tpetra::getDefaultComm();
  const auto invalid = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  auto own = Teuchos::rcp(new Map(invalid, ownedGids(), static_cast<GO>(7), comm));
  auto tgt = Teuchos::rcp(new Map(invalid, targetGids(), static_cast<GO>(7), comm));
  Teuchos::RCP<const Map> domain = own;
  Teuchos::RCP<const Map> column = tgt;
  Teuchos::RCP<const Import> importer = Teuchos::rcp(new Import(domain, column));
  std::string name;
  if (kind == SyntheticKind::Mixed) name = "synthetic-mixed";
  else if (kind == SyntheticKind::OneWay) name = "synthetic-one-way";
  else if (kind == SyntheticKind::LocalOnly) name = "synthetic-local-only";
  else name = "synthetic-interleaved";
  runCase(name, domain, column, importer, session, opts,
          kind != SyntheticKind::LocalOnly && nRanks > 1, false);
}

} // namespace NvshmemP2

#endif // TPETRA_PERFORMANCE_NVSHMEM_P2_HPP

