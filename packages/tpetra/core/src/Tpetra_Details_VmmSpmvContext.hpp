// Experimental, allocation-independent VMM-addressed local CRS SpMV.
// The matrix owns the translated column indices and borrows its own row map
// and values.  The input vector (and its VMM allocation) will be supplied
// at apply time, once native VMM-backed MultiVectors are implemented.
#ifndef TPETRA_EXPERIMENT_VMM_SPMV_CONTEXT_HPP
#define TPETRA_EXPERIMENT_VMM_SPMV_CONTEXT_HPP

#ifdef HAVE_TPETRA_INST_CUDA

#include "Tpetra_Details_VmmLayout.hpp"
#include "Tpetra_Details_VmmMatrixInfo.hpp"
#include "Tpetra_Details_Behavior.hpp"
#include "Tpetra_CrsMatrix_fwd.hpp"
#include "Tpetra_DistObject_fwd.hpp"
#include "Tpetra_MultiVector.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spmv.hpp"
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace VmmExperiment {

// KokkosSparse CRS ordinals are signed, and numCols() has the ordinal
// type as well.  Test the full (padded) global X extent, not just the
// largest column referenced by this rank.  A large local row count also
// requires the wider ordinal, independent of the Tpetra LocalOrdinal.
enum class VmmCrsOrdinalWidth { Int32, Int64, Unsupported };

inline VmmCrsOrdinalWidth chooseVmmCrsOrdinalWidth(
    std::uint64_t totalElements, std::uint64_t localNumRows) {
  const auto extent = std::max(totalElements, localNumRows);
  if (extent <= static_cast<std::uint64_t>(
                    std::numeric_limits<std::int32_t>::max())) {
    return VmmCrsOrdinalWidth::Int32;
  }
  if (extent <= static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max())) {
    return VmmCrsOrdinalWidth::Int64;
  }
  return VmmCrsOrdinalWidth::Unsupported;
}

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
  // The VMM column ordinal is deliberately independent of Tpetra's LO.
  // Preserve the original row-offset and value types in both variants.
  template <class VmmOrdinal>
  struct AddressedCrs {
    using entries_type = Kokkos::View<VmmOrdinal*, device_type>;
    using matrix_type = KokkosSparse::CrsMatrix<
        typename local_matrix_type::value_type,
        VmmOrdinal,
        device_type,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>,
        typename local_matrix_type::size_type>;

    entries_type entries;  // owns translated column indices
    matrix_type matrix;    // borrows Tpetra row offsets and values
  };

  using global_vector_type = Kokkos::View<
      const impl_scalar_type*, device_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  VmmSpmvContext(const matrix_type& A, VmmLayout<impl_scalar_type> layout)
      : comm_(layout.comm()), layout_(std::move(layout)) {
    rank_ = comm_.rank();
    size_ = comm_.size();
    if (layout_.logicalLocalCount() !=
        A.getDomainMap()->getLocalNumElements()) {
      throw std::invalid_argument(
          "VMM SpMV layout does not match the matrix domain Map");
    }
    buildAddressedMatrix(A);

    // Access-pattern analysis is collective and only needed for statistics.
    // Do not run it (or print the matrix statistics) unless explicitly asked.
    if (Tpetra::Details::Behavior::experimentalVmmStatistics()) {
      const auto& analysis = TpetraMatrixInfo::analyzeAccessPattern(A);
      if (analysis.myRank == 0) {
        std::ostringstream os;

        using dist_object_type = Tpetra::DistObject<
            char,
            typename matrix_type::local_ordinal_type,
            typename matrix_type::global_ordinal_type,
            typename matrix_type::node_type>;

        // Use the same address representation as the
        // DistObject::setObjectLabel() hook.
        const auto* identity =
            static_cast<const dist_object_type*>(&A);

        // The analysis already contains the globally reduced
        // reference counts, so this also gives global NNZ.
        TpetraMatrixInfo::count_type globalNnz = 0;
        for (const auto count : analysis.references) {
          globalNnz += count;
        }

        os << "\n[TPETRA_MATRIX] event=fillComplete"
           << " object=" << static_cast<const void*>(identity)
           << " label=" << std::quoted(A.getObjectLabel())
           << " rows=" << A.getGlobalNumRows()
           << " cols=" << A.getGlobalNumCols()
           << " nnz=" << globalNnz
           << '\n';

        print(os, analysis);
        std::cout << os.str();
      }
    }
  }

  // The context owns no CUDA allocation or virtual mappings.  The caller
  // must supply a compatible, peer-readable global X pointer and arrange
  // synchronization before invoking this method.  Milestone 3 will wire
  // this to native VMM-backed Tpetra Vector/MultiVector storage.
  const VmmLayout<impl_scalar_type>& layout() const { return layout_; }
  bool uses64BitCrsOrdinals() const { return static_cast<bool>(crs64_); }

  void applyGlobalPointer(const impl_scalar_type* globalX,
                          multivector_type& y,
                          const Scalar alpha, const Scalar beta) const {
    if (globalX == nullptr) {
      throw std::invalid_argument("VMM SpMV requires a mapped global X pointer");
    }
    if (!crs32_ && !crs64_) {
      throw std::logic_error("VMM SpMV context has no addressed CRS matrix");
    }
    const global_vector_type xGlobal(globalX, layout_.totalElements());
    // ReadWrite also preserves Y for beta != 0; OverwriteAll is not valid
    // for that case if the freshest copy of Y lives on the host.
    auto y2d = y.getLocalViewDevice(Tpetra::Access::ReadWrite);
    if (y2d.extent(1) != 1 ||
        y2d.extent(0) != static_cast<std::size_t>(
            crs32_ ? crs32_->matrix.numRows() : crs64_->matrix.numRows())) {
      throw std::runtime_error("VMM SpMV: unexpected output vector local shape");
    }
    auto y1d = Kokkos::subview(y2d, Kokkos::ALL(), 0);
    if (crs32_) {
      KokkosSparse::spmv("N", alpha, crs32_->matrix, xGlobal, beta, y1d);
    } else {
      KokkosSparse::spmv("N", alpha, crs64_->matrix, xGlobal, beta, y1d);
    }
  }

 private:
  // Build an independent signed-index CRS.  The original Tpetra matrix
  // continues using its own LO for ordinary local/Import-based SpMV.
  template <class VmmOrdinal>
  std::unique_ptr<AddressedCrs<VmmOrdinal>> makeAddressedCrs(
      const local_matrix_type& localA,
      const std::vector<std::uint64_t>& colLidToVmm) {
    using addressed_type = AddressedCrs<VmmOrdinal>;
    auto crs = std::make_unique<addressed_type>();
    const std::size_t nnz = localA.nnz();
    crs->entries = typename addressed_type::entries_type(
        "VMM direct column ordinals", nnz);

    auto entriesHost = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), localA.graph.entries);
    auto vmmEntriesHost = Kokkos::create_mirror_view(crs->entries);
    for (std::size_t k = 0; k < nnz; ++k) {
      const LO colLid = entriesHost(k);
      if (colLid < 0 || static_cast<std::size_t>(colLid) >= colLidToVmm.size()) {
        throw std::runtime_error(
            "Local CRS column ordinal is outside the Tpetra column Map");
      }
      // The width was selected using the full padded extent, so every
      // translated index is representable in this signed ordinal type.
      vmmEntriesHost(k) =
          static_cast<VmmOrdinal>(colLidToVmm[static_cast<std::size_t>(colLid)]);
    }
    Kokkos::deep_copy(crs->entries, vmmEntriesHost);
    Kokkos::fence("build VMM column ordinals");

    // KokkosSparse's numRows/numCols are also VmmOrdinal.  The width
    // selection covers both dimensions and the full padded X extent.
    const auto numRows = static_cast<VmmOrdinal>(localA.numRows());
    const auto numCols = static_cast<VmmOrdinal>(layout_.totalElements());

    // Unmanaged views borrow Tpetra row offsets and values; the separate
    // managed entries View above owns only the translated CRS ordinals.
    using values_view = typename addressed_type::matrix_type::values_type;
    using row_map_view = typename addressed_type::matrix_type::row_map_type;
    using indices_view = typename addressed_type::matrix_type::index_type;
    const values_view values(localA.values.data(), localA.values.extent(0));
    const row_map_view rowMap(localA.graph.row_map.data(),
                              localA.graph.row_map.extent(0));
    const indices_view indices(crs->entries.data(), crs->entries.extent(0));
    crs->matrix = typename addressed_type::matrix_type(
        "VMM addressed local matrix", numRows, numCols, nnz,
        values, rowMap, indices);
    return crs;
  }

  void buildAddressedMatrix(const matrix_type& A) {
    auto domainMap = A.getDomainMap();
    auto colMap = A.getColMap();

    if (!domainMap->isContiguous()) {
      throw std::runtime_error(
          "VMM prototype requires a contiguous Tpetra domain Map");
    }

    // Rank counts and padded offsets are already in VmmLayout.  Gather
    // only the first GID of each rank's domain segment to resolve
    // column-Map GIDs to owner/local pairs.
    const std::size_t myCount = domainMap->getLocalNumElements();
    auto myGids = domainMap->getLocalElementList();
    long long myFirst = 0;
    if (myCount != 0) myFirst = static_cast<long long>(myGids[0]);
    std::vector<long long> firstGid(size_);
#ifdef HAVE_MPI
    MPI_Allgather(&myFirst, 1, MPI_LONG_LONG,
                  firstGid.data(), 1, MPI_LONG_LONG, comm_.mpi());
#else
    firstGid[0] = myFirst;
#endif

    const auto& counts = layout_.logicalCounts();
    const std::size_t numColLids = colMap->getLocalNumElements();
    std::vector<std::uint64_t> colLidToVmm(numColLids);
    auto colGids = colMap->getLocalElementList();

    for (std::size_t lid = 0; lid < numColLids; ++lid) {
      const long long gid = static_cast<long long>(colGids[lid]);
      int owner = -1;
      std::uint64_t ownerLocal = 0;
      for (int p = 0; p < size_; ++p) {
        if (counts[p] == 0 || gid < firstGid[p]) continue;
        // Compare unsigned offsets rather than adding counts to a signed
        // GID; the latter could overflow for a large contiguous domain.
        const auto offset = static_cast<std::uint64_t>(gid) -
                            static_cast<std::uint64_t>(firstGid[p]);
        if (offset < counts[p]) {
          owner = p;
          ownerLocal = offset;
          break;
        }
      }
      if (owner < 0) {
        throw std::runtime_error(
            "Failed to resolve a column-map GID to a domain-map owner");
      }
      colLidToVmm[lid] = layout_.globalIndex(owner, ownerLocal);
    }

    // Acquire Tpetra's managed views only within this function.  Our
    // stored CRS representations retain unmanaged aliases, so later
    // getLocalMatrixHost() calls are not blocked by a live managed view.
    const auto localA = A.getLocalMatrixDevice();
    const auto width = chooseVmmCrsOrdinalWidth(
        layout_.totalElements(), static_cast<std::uint64_t>(localA.numRows()));
    if (width == VmmCrsOrdinalWidth::Int32) {
      crs32_ = makeAddressedCrs<std::int32_t>(localA, colLidToVmm);
    } else if (width == VmmCrsOrdinalWidth::Int64) {
      crs64_ = makeAddressedCrs<std::int64_t>(localA, colLidToVmm);
    } else {
      throw std::runtime_error(
          "VMM CRS dimensions exceed signed 64-bit indexing");
    }

    if (rank_ == 0 &&
        Tpetra::Details::Behavior::experimentalVmmStatistics()) {
      std::cout << "VMM CRS translation: borrowing Tpetra row_map + values; "
                << (crs32_ ? "signed 32-bit" : "signed 64-bit")
                << " addressed column ordinals, no VMM X allocation"
                << std::endl;
    }
  }

  VmmComm comm_;
  VmmLayout<impl_scalar_type> layout_;
  std::unique_ptr<AddressedCrs<std::int32_t>> crs32_;
  std::unique_ptr<AddressedCrs<std::int64_t>> crs64_;
  int rank_ = 0;
  int size_ = 1;
};

}  // namespace VmmExperiment

#endif  // HAVE_TPETRA_INST_CUDA
#endif  // TPETRA_EXPERIMENT_VMM_SPMV_CONTEXT_HPP


