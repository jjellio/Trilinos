// Experimental VMM-backed local CRS SpMV with an MPI-scoped arena.
// The matrix owns this context in the eventual CrsMatrix integration; the
// context *borrows* its finalized local CSR storage, not the matrix itself.
#ifndef TPETRA_EXPERIMENT_VMM_SPMV_CONTEXT_HPP
#define TPETRA_EXPERIMENT_VMM_SPMV_CONTEXT_HPP

#ifdef HAVE_TPETRA_INST_CUDA

#include "Tpetra_Details_VmmDistributedArena.hpp"
#include "Tpetra_CrsMatrix_fwd.hpp"
#include "Tpetra_MultiVector.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spmv.hpp"
#include <limits>
#include <type_traits>

namespace VmmExperiment {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
class VmmSpmvContext {
 public:
  using matrix_type = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using multivector_type = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using scalar_type = typename matrix_type::scalar_type;
  using impl_scalar_type = typename matrix_type::impl_scalar_type;
  using LO          = typename matrix_type::local_ordinal_type;
  using GO          = typename matrix_type::global_ordinal_type;
  using local_matrix_type = typename matrix_type::local_matrix_device_type;
  using device_type = typename local_matrix_type::device_type;
  using memory_space = typename device_type::memory_space;
  using entries_type = typename local_matrix_type::index_type::non_const_type;

  // Do not retain managed Kokkos views into Tpetra's WrappedDualView storage.
  // The source matrix owns its row pointers and values; VMM borrows them.
  using unmanaged_matrix_type = KokkosSparse::CrsMatrix<
      typename local_matrix_type::value_type,
      typename local_matrix_type::ordinal_type,
      device_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      typename local_matrix_type::size_type>;
  using unmanaged_vector_type =
      Kokkos::View<impl_scalar_type*, device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  static_assert(std::is_integral<LO>::value && std::is_signed<LO>::value,
                "KokkosSparse local ordinal must be a signed integer");

  VmmSpmvContext(const matrix_type& A, IpcMode ipcMode)
      : comm_(A.getDomainMap()->getComm()),
        arena_(A.getDomainMap()->getLocalNumElements(), ipcMode, comm_) {
    rank_ = comm_.rank();
    size_ = comm_.size();
    buildAddressedMatrix(A);

    xGlobal_ = unmanaged_vector_type(arena_.globalPtr(), arena_.totalElements());
    xLocal_  = unmanaged_vector_type(arena_.localPtr(), arena_.logicalLocalCount());
    // test faulting it
    Kokkos::deep_copy(xLocal_, Scalar(0));
  }

  // Publish the current Tpetra vector into this rank's physical VMM allocation.
  // This is scaffolding for the first implementation; a native VMM-backed
  // Tpetra/solver vector would eliminate this deep copy.
  void publish(const multivector_type& x) {

    auto x2d = x.getLocalViewDevice(Tpetra::Access::ReadOnly);
    
    if (x2d.extent(1) != 1 ||
        x2d.extent(0) != xLocal_.extent(0)) {
        throw std::runtime_error(
            "VMM publish: unexpected Tpetra vector local shape");
    }
    
    auto x1d = Kokkos::subview(x2d, Kokkos::ALL(), 0);
    
    // Synchronous by Kokkos definition.
    Kokkos::deep_copy(xLocal_, x1d);

    #ifdef HAVE_MPI
    // Establish a simple global epoch: all owner writes are complete before any
    // rank starts ordinary remote loads.  This is intentionally conservative.
    MPI_Barrier(comm_.mpi());
    #endif
  }

  void applyPublished(multivector_type& y, const Scalar alpha, const Scalar beta) const {
    auto y2d = y.getLocalViewDevice(Tpetra::Access::OverwriteAll);
    if (y2d.extent(1) != 1 || y2d.extent(0) != static_cast<std::size_t>(vmmA_.numRows())) {
      throw std::runtime_error("VMM SpMV: unexpected Tpetra output vector local shape");
    }
    auto y1d = Kokkos::subview(y2d, Kokkos::ALL(), 0);


    using ATS = KokkosKernels::ArithTraits<impl_scalar_type>;

    //const auto one  = ATS::one();
    //const auto zero = ATS::zero();

    KokkosSparse::spmv("N", alpha, vmmA_, xGlobal_, beta, y1d);
    Kokkos::fence("VMM direct SpMV fence");
  }

  void apply(const multivector_type& x, multivector_type& y, const Scalar alpha, const Scalar beta) {
    publish(x);
    applyPublished(y,alpha,beta);
  }

 private:
  void buildAddressedMatrix(const matrix_type& A) {
    auto domainMap = A.getDomainMap();
    auto colMap    = A.getColMap();

    if (!domainMap->isContiguous()) {
      throw std::runtime_error(
          "VMM prototype currently requires a contiguous Tpetra domain Map. "
          "The built-in miniFE generator satisfies this; general Maps can be "
          "supported later with Tpetra owner/LID lookup metadata.");
    }

    const std::size_t myCount = domainMap->getLocalNumElements();
    auto myGids = domainMap->getLocalElementList();
    long long myFirst = 0;
    if (myCount != 0) myFirst = static_cast<long long>(myGids[0]);

    std::vector<unsigned long long> counts(size_);
    std::vector<long long> firstGid(size_);
    unsigned long long myCountUll = static_cast<unsigned long long>(myCount);
#ifdef HAVE_MPI
    MPI_Allgather(&myCountUll, 1, MPI_UNSIGNED_LONG_LONG,
                  counts.data(), 1, MPI_UNSIGNED_LONG_LONG, comm_.mpi());
    MPI_Allgather(&myFirst, 1, MPI_LONG_LONG,
                  firstGid.data(), 1, MPI_LONG_LONG, comm_.mpi());
#else
    counts[0] = myCountUll;
    firstGid[0] = myFirst;
#endif

    if (arena_.totalElements() > static_cast<std::uint64_t>(std::numeric_limits<LO>::max())) {
      throw std::runtime_error(
          "Padded VMM arena exceeds the 32-bit/signed LocalOrdinal range. "
          "Use a wider VMM ordinal for this configuration.");
    }

    const std::size_t numColLids = colMap->getLocalNumElements();
    std::vector<LO> colLidToVmm(numColLids);
    auto colGids = colMap->getLocalElementList();
    const auto& base = arena_.baseElements();

    for (std::size_t lid = 0; lid < numColLids; ++lid) {
      const long long gid = static_cast<long long>(colGids[lid]);
      int owner = -1;
      std::uint64_t ownerLocal = 0;
      for (int p = 0; p < size_; ++p) {
        if (counts[p] == 0) continue;
        const long long begin = firstGid[p];
        const long long end   = begin + static_cast<long long>(counts[p]);
        if (gid >= begin && gid < end) {
          owner = p;
          ownerLocal = static_cast<std::uint64_t>(gid - begin);
          break;
        }
      }
      if (owner < 0) {
        throw std::runtime_error("Failed to resolve a column-map GID to a domain-map owner");
      }
      const std::uint64_t addressOrdinal = base[owner] + ownerLocal;
      if (addressOrdinal > static_cast<std::uint64_t>(std::numeric_limits<LO>::max())) {
        throw std::runtime_error("VMM address ordinal exceeds LocalOrdinal range");
      }
      colLidToVmm[lid] = static_cast<LO>(addressOrdinal);
    }

    // Acquire managed Tpetra views only for the duration of this method.
    // They are released on return, so subsequent getLocalMatrixHost() calls
    // do not encounter a permanently live view of CrsGraph::lclInds.
    const auto localA = A.getLocalMatrixDevice();
    const std::size_t nnz = localA.nnz();
    vmmEntries_ = entries_type("VMM direct column ordinals", nnz);

    auto entriesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                            localA.graph.entries);
    auto vmmEntriesHost = Kokkos::create_mirror_view(vmmEntries_);
    for (std::size_t k = 0; k < nnz; ++k) {
      const LO colLid = entriesHost(k);
      if (colLid < 0 || static_cast<std::size_t>(colLid) >= colLidToVmm.size()) {
        throw std::runtime_error("Local CRS column ordinal is outside the Tpetra column Map");
      }
      vmmEntriesHost(k) = colLidToVmm[static_cast<std::size_t>(colLid)];
    }
    Kokkos::deep_copy(vmmEntries_, vmmEntriesHost);
    Kokkos::fence("build VMM column ordinals");

    const LO numRows = static_cast<LO>(localA.numRows());
    const LO numCols = static_cast<LO>(arena_.totalElements());

    // The VMM matrix contains only UNMANAGED views.  Construct each view
    // from a raw pointer, not by copying managed views from localA.  The
    // translated column indices are allocated and owned by vmmEntries_.
    // The original row offsets and values remain owned by the CrsMatrix.
    using values_view = typename unmanaged_matrix_type::values_type;
    using row_map_view = typename unmanaged_matrix_type::row_map_type;
    using indices_view = typename unmanaged_matrix_type::index_type;
    const values_view values(localA.values.data(), localA.values.extent(0));
    const row_map_view rowMap(localA.graph.row_map.data(),
                              localA.graph.row_map.extent(0));
    const indices_view indices(vmmEntries_.data(), vmmEntries_.extent(0));

    vmmA_ = unmanaged_matrix_type("VMM addressed local matrix",
                                  numRows, numCols, nnz,
                                  values, rowMap, indices);

    if (rank_ == 0) {
      std::cout << "VMM CRS translation: reusing Tpetra row_map + values; "
                << "translated local column ordinals now index padded global VMM X"
                << std::endl;
    }
  }

  VmmComm comm_;
  DistributedVmmArena<impl_scalar_type> arena_;
  entries_type vmmEntries_;
  unmanaged_matrix_type vmmA_;
  unmanaged_vector_type xGlobal_;
  unmanaged_vector_type xLocal_;
  int rank_ = 0;
  int size_ = 1;
};

}  // namespace VmmExperiment

#endif  // HAVE_TPETRA_INST_CUDA
#endif  // TPETRA_EXPERIMENT_VMM_SPMV_CONTEXT_HPP


