#pragma once

#include <string>
#include <stdexcept>

#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Vector.hpp>

#include <Galeri_XpetraMaps.hpp>
#include <Galeri_XpetraMatrixTypes.hpp>
#include <Galeri_XpetraProblemFactory.hpp>

namespace my_helper {

using Scalar = typename Tpetra::Vector<>::scalar_type;
using LO     = typename Tpetra::Map<>::local_ordinal_type;
using GO     = typename Tpetra::Map<>::global_ordinal_type;


template <class Node>
using crs_matrix_type =
    Tpetra::CrsMatrix<Scalar, LO, GO, Node>;


/*
 * General overload.
 *
 * The caller may put nx, ny, nz, mx, my, mz, stencil coefficients,
 * etc. into galeriList.
 */
template <class Node>
Teuchos::RCP<crs_matrix_type<Node>>
get_galeri_matrix(
    const std::string& matrixName,
    const std::string& mapType,
    Teuchos::ParameterList galeriList,
    const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
{
    using Map =
        Tpetra::Map<LO, GO, Node>;

    using Matrix =
        Tpetra::CrsMatrix<Scalar, LO, GO, Node>;

    using MultiVector =
        Tpetra::MultiVector<Scalar, LO, GO, Node>;

    Teuchos::RCP<const Map> map =
        Galeri::Xpetra::CreateMap<LO, GO, Map>(
            mapType,
            comm,
            galeriList);

    auto problem =
        Galeri::Xpetra::BuildProblem<
            Scalar,
            LO,
            GO,
            Map,
            Matrix,
            MultiVector>(
                matrixName,
                map,
                galeriList);

    Teuchos::RCP<Matrix> A =
        problem->BuildMatrix();

    return A;
}


/*
 * Convenience overload.
 *
 * Convention:
 *     nsize is the number of grid points in each active dimension.
 *
 * Thus:
 *
 * Laplace1D, nsize=100 -> ~100 unknowns
 * Laplace2D, nsize=100 -> ~10,000 unknowns
 * Laplace3D, nsize=100 -> ~1,000,000 unknowns
 */
template <class Node>
Teuchos::RCP<crs_matrix_type<Node>>
get_galeri_matrix(
    const std::string& matrixName,
    const GO nsize,
    const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
{
    Teuchos::ParameterList galeriList;

    galeriList.set("nx", nsize);

    std::string mapType;

    if (matrixName == "Laplace1D" ||
        matrixName == "Identity")
    {
        mapType = "Cartesian1D";
    }
    else if (matrixName == "Laplace2D" ||
             matrixName == "Star2D" ||
             matrixName == "BigStar2D" ||
             matrixName == "AnisotropicDiffusion" ||
             matrixName == "Recirc2D")
    {
        galeriList.set("ny", nsize);

        mapType = "Cartesian2D";
    }
    else if (matrixName == "Laplace3D" ||
             matrixName == "Brick3D" ||
             matrixName == "Scalar3D_27Pt" ||
             matrixName == "HexFEM_LapStiff" ||
             matrixName == "HexFEM_Mass")
    {
        galeriList.set("ny", nsize);
        galeriList.set("nz", nsize);

        mapType = "Cartesian3D";
    }
    else
    {
        throw std::invalid_argument(
            "my_helper::get_galeri_matrix: "
            "don't know which Galeri map to use for matrix \"" +
            matrixName + "\"");
    }

    return get_galeri_matrix<Node>(
        matrixName,
        mapType,
        galeriList,
        comm);
}

} // namespace my_helper
