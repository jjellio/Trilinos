#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <Tpetra_Access.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Vector.hpp>
#include <MatrixMarket_Tpetra.hpp>

#include <Galeri_XpetraMatrixTypes.hpp>
#include <Galeri_XpetraProblemFactory.hpp>

namespace my_helper {

using Scalar = typename Tpetra::Vector<>::scalar_type;
using LO     = typename Tpetra::Map<>::local_ordinal_type;
using GO     = typename Tpetra::Map<>::global_ordinal_type;

template <class Node>
using crs_matrix_type = Tpetra::CrsMatrix<Scalar, LO, GO, Node>;

// Galeri matrix families supported by Galeri::Xpetra::BuildProblem.
inline int matrix_dimension(const std::string& matrixType)
{
    if (matrixType == "Laplace1D" ||
        matrixType == "Identity") {
        return 1;
    }

    if (matrixType == "Laplace2D" ||
        matrixType == "Star2D" ||
        matrixType == "BigStar2D" ||
        matrixType == "AnisotropicDiffusion" ||
        matrixType == "Elasticity2D" ||
        matrixType == "Recirc2D") {
        return 2;
    }

    if (matrixType == "Laplace3D" ||
        matrixType == "Brick3D" ||
        matrixType == "Scalar3D_27Pt" ||
        matrixType == "HexFEM_LapStiff" ||
        matrixType == "HexFEM_Mass" ||
        matrixType == "Elasticity3D") {
        return 3;
    }

    throw std::invalid_argument(
        "my_helper::get_galeri_matrix: unsupported Galeri matrixType \"" +
        matrixType + "\"");
}

inline int dofs_per_node(const std::string& matrixType)
{
    if (matrixType == "Elasticity2D") {
        return 2;
    }
    if (matrixType == "Elasticity3D") {
        return 3;
    }
    return 1;
}

inline GO get_go_parameter(
    const Teuchos::ParameterList& params,
    const char* name)
{
    if (params.isType<int>(name)) {
        return static_cast<GO>(params.get<int>(name));
    }
    return params.get<GO>(name);
}

/*
 * Construct a contiguous Tpetra map while keeping all DOFs belonging to a
 * mesh node on the same MPI rank.
 *
 * For scalar problems dofsPerNode == 1, so this is just the usual contiguous
 * distribution.  For Elasticity2D/3D, nodes are distributed first and then
 * expanded to 2 or 3 interleaved point DOFs per node.
 */
template <class Node>
Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>
make_contiguous_dof_map(
    const Tpetra::global_size_t globalNodes,
    const int dofsPerNode,
    const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
{
    using map_type = Tpetra::Map<LO, GO, Node>;

    if (dofsPerNode <= 0) {
        throw std::invalid_argument(
            "my_helper::make_contiguous_dof_map: dofsPerNode must be positive");
    }

    const Tpetra::global_size_t numRanks =
        static_cast<Tpetra::global_size_t>(comm->getSize());
    const Tpetra::global_size_t rank =
        static_cast<Tpetra::global_size_t>(comm->getRank());

    const Tpetra::global_size_t baseNodes = globalNodes / numRanks;
    const Tpetra::global_size_t extraNodes = globalNodes % numRanks;

    Tpetra::global_size_t localNodes = baseNodes;
    if (rank < extraNodes) {
        ++localNodes;
    }

    const Tpetra::global_size_t globalDofs =
        globalNodes * static_cast<Tpetra::global_size_t>(dofsPerNode);
    const Tpetra::global_size_t localDofsGlobal =
        localNodes * static_cast<Tpetra::global_size_t>(dofsPerNode);

    const std::size_t localDofs =
        static_cast<std::size_t>(localDofsGlobal);

    return Teuchos::rcp(
        new map_type(
            globalDofs,
            localDofs,
            static_cast<GO>(0),
            comm));
}

/*
 * Build exact logical labels for the matrix GIDs.
 *
 * Each row of the returned MultiVector contains
 *
 *   [ gid, ix, iy, iz, component ]
 *
 * where unused spatial dimensions are zero.  For scalar matrices component
 * is always zero.  For elasticity, consecutive matrix GIDs are the point
 * DOFs belonging to the same mesh node.
 *
 * Because this helper constructs a zero-based contiguous Galeri map, these
 * labels provide the semantic mapping needed to compare a Matrix Market dump
 * against an independently generated matrix, regardless of that generator's
 * row/column ordering.
 */
template <class Node>
Teuchos::RCP<Tpetra::MultiVector<Scalar, LO, GO, Node>>
make_logical_labels(
    const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& map,
    const GO nx,
    const GO ny,
    const int dofsPerNode)
{
    using multivector_type = Tpetra::MultiVector<Scalar, LO, GO, Node>;

    constexpr std::size_t numLabelColumns = 5;
    Teuchos::RCP<multivector_type> labels =
        Teuchos::rcp(new multivector_type(map, numLabelColumns));

    auto view = labels->getLocalViewHost(Tpetra::Access::OverwriteAll);
    const GO dofs = static_cast<GO>(dofsPerNode);

    const std::size_t localSize = map->getLocalNumElements();
    for (std::size_t k = 0; k < localSize; ++k) {
        const LO lid = static_cast<LO>(k);
        const GO gid = map->getGlobalElement(lid);

        const GO nodeGid = gid / dofs;
        const GO component = gid % dofs;

        const GO ix = nodeGid % nx;
        const GO iy = (nodeGid / nx) % ny;
        const GO iz = nodeGid / (nx * ny);

        view(lid, 0) = static_cast<Scalar>(gid);
        view(lid, 1) = static_cast<Scalar>(ix);
        view(lid, 2) = static_cast<Scalar>(iy);
        view(lid, 3) = static_cast<Scalar>(iz);
        view(lid, 4) = static_cast<Scalar>(component);
    }

    return labels;
}

/*
 * Optional validation output.
 *
 * <prefix>.mtx         : sparse Galeri matrix
 * <prefix>.logical.mtx : dense N x 5 logical labels [gid ix iy iz component]
 *
 * MatrixMarket::Writer handles gathering the distributed Tpetra objects for
 * this diagnostic output.  This is intentionally not intended as scalable I/O.
 */
template <class Node>
void write_galeri_validation_files(
    const std::string& prefix,
    const Teuchos::RCP<crs_matrix_type<Node>>& A,
    const GO nx,
    const GO ny,
    const int dofsPerNode)
{
    if (prefix.empty()) {
        return;
    }

    using matrix_type = crs_matrix_type<Node>;
    using writer_type = Tpetra::MatrixMarket::Writer<matrix_type>;

    const std::string suffix =
        Tpetra::MatrixMarket::supportsGzipOutput() ? ".mtx.gz" : ".mtx";

    writer_type::writeSparseFile(prefix + suffix, A);

    auto labels = make_logical_labels<Node>(
        A->getRowMap(), nx, ny, dofsPerNode);
    writer_type::writeDenseFile(prefix + ".logical" + suffix, labels);
}

/*
 * General contiguous-map overload.
 *
 * The caller supplies Galeri parameters such as nx, ny, nz, stretch*, E, nu,
 * etc.  This helper deliberately constructs the Tpetra map itself instead of
 * calling Galeri::Xpetra::CreateMap(), because the VMM prototype requires a
 * contiguous domain map.
 *
 * For elasticity, the physical mesh decomposition is forced to slabs in the
 * slowest-varying lexicographic dimension.  This keeps Galeri's element
 * assembly aligned with the contiguous point-DOF ownership as closely as
 * possible:
 *
 *   Elasticity2D: mx = 1, my = nranks
 *   Elasticity3D: mx = 1, my = 1, mz = nranks
 */
template <class Node>
Teuchos::RCP<crs_matrix_type<Node>>
get_galeri_matrix(
    const std::string& matrixType,
    Teuchos::ParameterList params,
    const Teuchos::RCP<const Teuchos::Comm<int>>& comm,
    const std::string& savePrefix = "")
{
    using map_type = Tpetra::Map<LO, GO, Node>;
    using matrix_type = Tpetra::CrsMatrix<Scalar, LO, GO, Node>;
    using multivector_type = Tpetra::MultiVector<Scalar, LO, GO, Node>;

    const int dim = matrix_dimension(matrixType);
    const int dofsPerNode = dofs_per_node(matrixType);

    const GO nx = get_go_parameter(params, "nx");
    GO ny = static_cast<GO>(1);
    GO nz = static_cast<GO>(1);

    if (dim >= 2) {
        ny = get_go_parameter(params, "ny");
    }
    if (dim >= 3) {
        nz = get_go_parameter(params, "nz");
    }

    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument(
            "my_helper::get_galeri_matrix: nx, ny, and nz must be positive");
    }

    if ((matrixType == "Elasticity2D" || matrixType == "Elasticity3D") &&
        (nx < 2 || ny < 2 || (dim == 3 && nz < 2))) {
        throw std::invalid_argument(
            "my_helper::get_galeri_matrix: elasticity requires at least "
            "two grid points in each active dimension");
    }

    const Tpetra::global_size_t globalNodes =
        static_cast<Tpetra::global_size_t>(nx) *
        static_cast<Tpetra::global_size_t>(ny) *
        static_cast<Tpetra::global_size_t>(nz);

    // Elasticity's BuildMesh() uses mx/my/mz to decide which finite elements
    // each rank assembles.  A slab decomposition in the slowest-varying
    // dimension best matches the contiguous lexicographic point-DOF map.
    if (matrixType == "Elasticity2D") {
        params.set("mx", static_cast<GO>(1));
        params.set("my", static_cast<GO>(comm->getSize()));
    }
    else if (matrixType == "Elasticity3D") {
        params.set("mx", static_cast<GO>(1));
        params.set("my", static_cast<GO>(1));
        params.set("mz", static_cast<GO>(comm->getSize()));
    }

    Teuchos::RCP<const map_type> map =
        make_contiguous_dof_map<Node>(globalNodes, dofsPerNode, comm);

    auto problem =
        Galeri::Xpetra::BuildProblem<
            Scalar,
            LO,
            GO,
            map_type,
            matrix_type,
            multivector_type>(
                matrixType,
                map,
                params);

    Teuchos::RCP<matrix_type> A = problem->BuildMatrix();

    if (! savePrefix.empty() ){
        std::ostringstream oss;
        oss << savePrefix << matrixType << "_" << nx << "x" << ny << "x" << nz;
    
        write_galeri_validation_files<Node>(
            oss.str(), A, nx, ny, dofsPerNode);
    }

    return A;
}

/*
 * Convenience overload.
 *
 * nsize is the number of mesh/grid points in each active dimension.
 *
 * Examples for nsize = 100:
 *
 *   Laplace1D      ->       100 rows
 *   Laplace2D      ->    10,000 rows
 *   Laplace3D      -> 1,000,000 rows
 *   Elasticity2D   ->    20,000 rows  (2 DOFs/node)
 *   Elasticity3D   -> 3,000,000 rows  (3 DOFs/node)
 */
template <class Node>
Teuchos::RCP<crs_matrix_type<Node>>
get_galeri_matrix(
    const std::string& matrixType,
    const GO nsize,
    const Teuchos::RCP<const Teuchos::Comm<int>>& comm,
    const std::string& savePrefix = "")
{
    if (nsize <= 0) {
        throw std::invalid_argument(
            "my_helper::get_galeri_matrix: nsize must be positive");
    }

    const int dim = matrix_dimension(matrixType);

    Teuchos::ParameterList params;
    params.set("nx", nsize);

    if (dim >= 2) {
        params.set("ny", nsize);
    }
    if (dim >= 3) {
        params.set("nz", nsize);
    }

    return get_galeri_matrix<Node>(matrixType, params, comm, savePrefix);
}

}  // namespace my_helper


