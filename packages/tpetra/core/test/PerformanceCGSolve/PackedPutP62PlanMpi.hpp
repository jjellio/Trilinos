// P6.2 one-time peer signal slot and sorted-GID reconciliation.
#ifndef TPETRA_PERFORMANCE_PACKED_PUT_P62_PLAN_MPI_HPP
#define TPETRA_PERFORMANCE_PACKED_PUT_P62_PLAN_MPI_HPP
#include "PackedPutP62Plan.hpp"
#include "NvshmemP0.hpp"
#include "Tpetra_Map.hpp"
#include <mpi.h>
#include <string>
#include <vector>

namespace PackedPutP62 {
inline void mpiCheck(int code, const char* what) {
  require(code == MPI_SUCCESS, std::string("P6.2 MPI setup failed: ") + what);
}

template<class LO, class GO, class Node>
Plan build(const PackedPutP51::Plan& old,
           const Tpetra::Map<LO, GO, Node>& domain,
           const Tpetra::Map<LO, GO, Node>& column,
           U64 chunkDoubles, const NvshmemP0::Session& session) {
  require(old.rank == session.mpiRank() && old.pe == session.pe() &&
          session.numPes() == static_cast<int>(old.peForRank.size()),
          "P6.2 old plan/session disagree");
  std::vector<GO> domainGids(PackedPutP51::asSize(old.ownedLength, "domain GIDs"));
  std::vector<GO> targetGids(PackedPutP51::asSize(old.targetLength, "column GIDs"));
  for (std::size_t i = 0; i < domainGids.size(); ++i)
    domainGids[i] = domain.getGlobalElement(static_cast<LO>(i));
  for (std::size_t i = 0; i < targetGids.size(); ++i)
    targetGids[i] = column.getGlobalElement(static_cast<LO>(i));
  Plan result = buildLocal(old, domainGids, targetGids, chunkDoubles);
  const std::size_t n = old.peForRank.size();
  std::vector<U64> fromSender(n, 0), senderDigest(n, 0), basesAtSender(n, PackedPutP51::noOffset);
  mpiCheck(MPI_Alltoall(result.sendChunksForRank.data(), 1, MPI_UINT64_T,
                        fromSender.data(), 1, MPI_UINT64_T, session.comm()),
           "MPI_Alltoall(P6.2 descriptor counts)");
  require(fromSender == result.receiveChunksFromRank,
          "P6.2 matching peers expect different descriptor counts");
  mpiCheck(MPI_Alltoall(result.sendGidDigestForRank.data(), 1, MPI_UINT64_T,
                        senderDigest.data(), 1, MPI_UINT64_T, session.comm()),
           "MPI_Alltoall(P6.2 GID fingerprints)");
  require(senderDigest == result.recvGidDigestFromRank,
          "P6.2 sorted sender/receiver GID order differs");
  mpiCheck(MPI_Alltoall(result.readyBaseForSourceRank.data(), 1, MPI_UINT64_T,
                        basesAtSender.data(), 1, MPI_UINT64_T, session.comm()),
           "MPI_Alltoall(P6.2 ready slot bases)");
  U64 maximumReceives = 0;
  mpiCheck(MPI_Allreduce(&result.localReadyElements, &maximumReceives,
                         1, MPI_UINT64_T, MPI_MAX, session.comm()),
           "MPI_Allreduce(P6.2 symmetric ready capacity)");
  finalize(result, old, basesAtSender, maximumReceives);
  return result;
}
} // namespace PackedPutP62
#endif
