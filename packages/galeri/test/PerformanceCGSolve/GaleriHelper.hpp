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

bool is_1d(const std::string& name)
{
    return name == "Laplace1D" ||
           name == "Identity";
}

bool is_2d(const std::string& name)
{
    return name == "Laplace2D" ||
           name == "Star2D" ||
           name == "BigStar2D" ||
           name == "AnisotropicDiffusion" ||
           name == "Recirc2D";
}

bool is_3d(const std::string& name)
{
    return name == "Laplace3D" ||
           name == "Brick3D" ||
           name == "Scalar3D_27Pt" ||
           name == "HexFEM_LapStiff" ||
           name == "HexFEM_Mass";
}

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
  
  using map_type =
      Tpetra::Map<LO, GO, Node>;
  
  using matrix_type =
      Tpetra::CrsMatrix<Scalar, LO, GO, Node>;
  
  using multivector_type =
      Tpetra::MultiVector<Scalar, LO, GO, Node>;
  
  Teuchos::ParameterList params;
  
  GO nx = nsize;
  GO ny = 1;
  GO nz = 1;
  
  if (is_2d(matrixName)) {
      ny = nsize;
  }
  else if (is_3d(matrixName)) {
      ny = nsize;
      nz = nsize;
  }
  
  params.set("nx", nx);
  
  if (ny != 1)
      params.set("ny", ny);
  
  if (nz != 1)
      params.set("nz", nz);
  
  Tpetra::global_size_t N =
      static_cast<Tpetra::global_size_t>(nx) *
      static_cast<Tpetra::global_size_t>(ny) *
      static_cast<Tpetra::global_size_t>(nz);
  
  auto map =
      Teuchos::rcp(
          new map_type(
              N,
              static_cast<GO>(0),
              comm));
  
  auto problem =
      Galeri::Xpetra::BuildProblem<
          Scalar,
          LO,
          GO,
          map_type,
          matrix_type,
          multivector_type>(
              matrixName,
              map,
              params);
  
  auto A = problem->BuildMatrix();

  return A;
}

} // namespace my_helper
