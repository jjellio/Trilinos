// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
// @HEADER

/*!
\example lesson02_read_modify_vec_kokkos.cpp
\brief Read and modify the entries of a vector (Tpetra::Vector),
  using Kokkos::View to access local data.

\ref Tpetra_Lesson02 explains this example in detail.
*/

#include <Tpetra_Core.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <cublas_v2.h>
#include <KokkosBlas1_dot.hpp>
#include <random>    // random
#include <algorithm> // shuffle
#include "descriptive_stats.hpp"

bool header_flag = false;

template<typename ds_array_type, typename experiment_type>
void analyze_timings(      std::map<experiment_type, ds_array_type>& exp_data,
                     const std::vector<experiment_type>& exps,
                           std::map<experiment_type, std::string>& exp_to_string,
                     const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
                     std::ostream& out)
{
  // setup the stats stuff
  using Time = DescriptiveStats::Time;
  using clock_resolution = DescriptiveStats::ns;
  using ds_map_type = DescriptiveStats::descriptive_stat_map_type;
  using DescriptiveStats::get_descriptive_stats;
  using DescriptiveStats::print_descriptive_stats;

  const double timer_ratio = double(clock_resolution::period::num) / double(clock_resolution::period::den);

  std::stringstream oss;
  
  for (const auto& e : exps) {

    std::stringstream ss;
    // map of statistics
    ds_map_type stats;


    // compute the stats
    get_descriptive_stats(exp_data[e], exp_data[e].size(), stats);

    // dump the CSV banner first
    if ( (!header_flag) && comm->getRank () == 0) {
      print_descriptive_stats (oss, stats, timer_ratio, "", true, true);
    }
    header_flag = true;

    print_descriptive_stats (oss, stats, timer_ratio, exp_to_string[e], true);
  }

  if (comm->getRank () == 0) {
    out << oss.str();
    out.flush();
    MPI_Barrier(MPI_COMM_WORLD);
  } else {

    MPI_Barrier(MPI_COMM_WORLD);
    out << oss.str();
  }
  out.flush();
}

void
evaluate_dot(const int local_n, const int stride,
             const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
             std::ostream& out)
{
  using std::endl;
  using Teuchos::Array;
  using Teuchos::ArrayRCP;
  using Teuchos::ArrayView;
  using Teuchos::outArg;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::REDUCE_SUM;
  using Teuchos::reduceAll;

  using Time = DescriptiveStats::Time;
  using clock_resolution = DescriptiveStats::ns;
  using double_secs = DescriptiveStats::double_secs;

  using ds_array = std::vector<double>;

  const int myRank = comm->getRank ();

  using map_type = Tpetra::Map<>;
  using vector_type = Tpetra::Vector<double,int>;

  using global_ordinal_type = vector_type::global_ordinal_type;

  // perfectly partition the vector
  const Tpetra::global_size_t numGlobalEntries = comm->getSize () * local_n * stride;

  // Construct a Map that puts the same number of equations on each
  // MPI process.
  RCP<const map_type> contigMap =
    rcp (new map_type (numGlobalEntries, local_n, 0, comm));

  // create two vectors
  vector_type x (contigMap);
  vector_type y (contigMap);

  // grab the device views so we can call cublas directly
  typedef typename vector_type::dual_view_type dual_view_type;
  typedef typename dual_view_type::t_dev device_view_type;
  auto cuX = subview ( x.getLocalView<Kokkos::Cuda> (), Kokkos::ALL (), 0);
  auto cuY = subview ( y.getLocalView<Kokkos::Cuda> (), Kokkos::ALL (), 0);

  cublasHandle_t cublas_handle;
  const auto cublas_status = cublasCreate(&cublas_handle);


  //////////////////////////////////////////////////////////////////////
  // Fill the Vector with a single number, or with random numbers
  //////////////////////////////////////////////////////////////////////

  // Set the entries of x to (pseudo)random numbers.  Please don't
  // consider this a good parallel pseudorandom number generator.
  x.randomize (0,4);
  y.randomize (0,4);

  //x.putScalar(1.0);
  //y.putScalar(1.0);
  if (!header_flag) {
    out << x.description() << "\n"
        << cuY.extent(0) << "\n";
  }

  std::vector<double> dots;
  dots.resize(1);
  
  double cublas_dot_result;
  double kk_dot_result;

  enum class Experiments { CUBLAS_LOCAL, KK_LOCAL, ALL_REDUCE, TPETRA_DOT };
  std::vector<Experiments> exps;
  exps.push_back(Experiments::CUBLAS_LOCAL);
  exps.push_back(Experiments::KK_LOCAL);
  exps.push_back(Experiments::ALL_REDUCE);
  exps.push_back(Experiments::TPETRA_DOT);

  // create a mapping for experiments to strings
  std::map<Experiments,std::string> exps_to_string;
  exps_to_string[Experiments::CUBLAS_LOCAL] = std::to_string(local_n) + "-CuBLAS_dot";
  exps_to_string[Experiments::KK_LOCAL] = std::to_string(local_n) +"-KK_dot";
  exps_to_string[Experiments::ALL_REDUCE] = "1-Allreduce";
  exps_to_string[Experiments::TPETRA_DOT] = std::to_string(local_n) + "-Tpetra_dot";

  // randomize the kernel runs
  std::random_device rd;
  std::mt19937 g(rd());
  std::vector<Experiments> randomized_exps (exps);

  std::map<Experiments, ds_array> exp_data;

  const bool skip_first = true;

  int num_samples = 1000;

  if (skip_first) num_samples++;

  for(int nsample=0; nsample < num_samples; ++nsample) {  
    std::shuffle(randomized_exps.begin(), randomized_exps.end(), g);
    MPI_Bcast(randomized_exps.data(), randomized_exps.size()*sizeof(randomized_exps[0]), MPI_BYTE,
              0, MPI_COMM_WORLD);

    dots[0] = 0.0;
    cublas_dot_result = 0.0;
    kk_dot_result = 0.0;
    double_secs elapsed_time;

    for (const auto& e : randomized_exps) {
      switch(e) {
      case Experiments::KK_LOCAL:
      {
        const auto t0 = Time::now ();
        kk_dot_result = KokkosBlas::dot(cuX,cuY);
        const auto t1 = Time::now ();
        elapsed_time = t1-t0;
      }
      break;
      case Experiments::CUBLAS_LOCAL:
      {
        const auto t0 = Time::now ();
        auto cuRC = cublasDdot (cublas_handle,
                                local_n,
                                cuX.data(), stride,
                                cuY.data(), stride,
                                &cublas_dot_result);
        const auto t1 = Time::now ();
        elapsed_time = t1-t0;
      }
      break;
      case Experiments::ALL_REDUCE:
      {
        MPI_Barrier(MPI_COMM_WORLD);
        const auto t0 = Time::now ();
        MPI_Allreduce(MPI_IN_PLACE, &cublas_dot_result, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        const auto t1 = Time::now ();
        elapsed_time = t1-t0;
      }
      break;
      case Experiments::TPETRA_DOT:
      {
        MPI_Barrier(MPI_COMM_WORLD);
        const auto t0 = Time::now ();
        x.dot(y, dots);
        const auto t1 = Time::now ();
        elapsed_time = t1-t0;
      }
      break;
      } // switch

      if (! (skip_first && nsample == 0) )
        exp_data[e].push_back(std::chrono::duration_cast<clock_resolution>(elapsed_time).count ());
    } //randomized loop
  } // samples
  
  if (comm->getRank () == 3) {
    if (!header_flag)
      out << "x.dot(y) = " << dots[0] << ", cublas = " << cublas_dot_result << ", kk = " << kk_dot_result << "\n";
  }

  analyze_timings(exp_data, exps, exps_to_string, comm, out);
}

void
evaluate_gemm(const int local_n, const int stride,
              const int num_mv_left,
              const int num_mv_right,
              const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
              std::ostream& out)
{
  using std::endl;
  using Teuchos::Array;
  using Teuchos::ArrayRCP;
  using Teuchos::ArrayView;
  using Teuchos::outArg;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::REDUCE_SUM;
  using Teuchos::reduceAll;

  using ds_map_type = DescriptiveStats::descriptive_stat_map_type;
  using Time = DescriptiveStats::Time;
  using clock_resolution = DescriptiveStats::ns;
  using double_secs = DescriptiveStats::double_secs;

  using ds_array = std::vector<double>;

  const int myRank = comm->getRank ();

  using map_type = Tpetra::Map<>;
  using mv_type = Tpetra::MultiVector<double,int>;

  using global_ordinal_type = vector_type::global_ordinal_type;

  // perfectly partition the vector
  const Tpetra::global_size_t numGlobalEntries = comm->getSize () * local_n * stride;

  // Construct a Map that puts the same number of equations on each
  // MPI process.
  RCP<const map_type> contigMap =
    rcp (new map_type (numGlobalEntries, local_n, 0, comm));

  // create two vectors
  mv_type x (contigMap, num_mv_left);
  mv_type y (contigMap, num_mv_right);

  // grab the device views so we can call cublas directly
  typedef typename vector_type::dual_view_type dual_view_type;
  typedef typename dual_view_type::t_dev device_view_type;
  auto cuX = subview ( x.getLocalView<Kokkos::Cuda> (), Kokkos::ALL (), Kokkos::ALL());
  auto cuY = subview ( y.getLocalView<Kokkos::Cuda> (), Kokkos::ALL (), Kokkos::ALL());

  cublasHandle_t cublas_handle;
  const auto cublas_status = cublasCreate(&cublas_handle);


  //////////////////////////////////////////////////////////////////////
  // Fill the Vector with a single number, or with random numbers
  //////////////////////////////////////////////////////////////////////

  // Set the entries of x to (pseudo)random numbers.  Please don't
  // consider this a good parallel pseudorandom number generator.
  x.randomize (0,4);
  y.randomize (0,4);

  //x.putScalar(1.0);
  //y.putScalar(1.0);
  if (!header_flag) {
    out << x.description() << "\n"
        << "X: " << cuX.extent(0) << " x " << cuX.extent(1) << "\n"
        << "Y: " << cuY.extent(0) << " x " << cuY.extent(1) << "\n";
  }

  Teuchos::RCP<const map_type> locallyreplicated_map (new map_type (num_mv_left,
                                                             0,
                                                             comm,
                                                   Tpetra::LocallyReplicated));
  mv_type solution(locallyreplicated_map, num_mv_right);
  mv_type cublas_solution_mv(locallyreplicated_map, num_mv_right);
  mv_type kk_solution_mv(locallyreplicated_map, num_mv_right);
  device_view_type cublas_solution_view = cublas_solution_mv.getLocalView<Kokkos::Cuda> ();
  device_view_type kk_solution_view = kk_solution_mv.getLocalView<Kokkos::Cuda> ();

 

  enum class Experiments { CUBLAS_LOCAL, KK_LOCAL, ALL_REDUCE, TPETRA_DOT };
  std::vector<Experiments> exps;
  exps.push_back(Experiments::CUBLAS_LOCAL);
  exps.push_back(Experiments::KK_LOCAL);
  exps.push_back(Experiments::ALL_REDUCE);
  exps.push_back(Experiments::TPETRA_DOT);

  // create a mapping for experiments to strings
  // create a mapping for experiments to strings
  std::map<Experiments,std::string> exps_to_string;
  exps_to_string[Experiments::CUBLAS_LOCAL] = std::to_string(local_n) + "-CuBLAS_gemm";
  exps_to_string[Experiments::KK_LOCAL] = std::to_string(local_n) +"-KK_gemm";
  exps_to_string[Experiments::ALL_REDUCE] = "1-Allreduce";
  exps_to_string[Experiments::TPETRA_DOT] = std::to_string(local_n) + "-Tpetra_multiply";

  std::map<Experiments,std::string> exps_to_string;
  exps_to_string[Experiments::CUBLAS_LOCAL] = "CuBLAS_local";
  exps_to_string[Experiments::KK_LOCAL] = "KK_local";
  exps_to_string[Experiments::ALL_REDUCE] = "Allreduce";
  exps_to_string[Experiments::TPETRA_DOT] = "Tpetra_dot";

  // randomize the kernel runs
  std::random_device rd;
  std::mt19937 g(rd());
  std::vector<Experiments> randomized_exps (exps);

  std::map<Experiments, ds_array> exp_data;

  const bool skip_first = true;

  int num_samples = 1000;

  if (skip_first) num_samples++;

  for(int nsample=0; nsample < num_samples; ++nsample) {  
    std::shuffle(randomized_exps.begin(), randomized_exps.end(), g);
    MPI_Bcast(randomized_exps.data(), randomized_exps.size()*sizeof(randomized_exps[0]), MPI_BYTE,
              0, MPI_COMM_WORLD);

    dots[0] = 0.0;
    cublas_dot_result = 0.0;
    kk_dot_result = 0.0;
    double_secs elapsed_time;

    for (const auto& e : randomized_exps) {
      switch(e) {
      case Experiments::KK_LOCAL:
      {
        const auto t0 = Time::now ();
        kk_dot_result = KokkosBlas::dot(cuX,cuY);
        const auto t1 = Time::now ();
        elapsed_time = t1-t0;
      }
      break;
      case Experiments::CUBLAS_LOCAL:
      {
        const auto t0 = Time::now ();
        auto cuRC = cublasDdot (cublas_handle,
                                local_n,
                                cuX.data(), stride,
                                cuY.data(), stride,
                                &cublas_dot_result);
        const auto t1 = Time::now ();
        elapsed_time = t1-t0;
      }
      break;
      case Experiments::ALL_REDUCE:
      {
        MPI_Barrier(MPI_COMM_WORLD);
        const auto t0 = Time::now ();
        MPI_Allreduce(MPI_IN_PLACE, &cublas_dot_result, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        const auto t1 = Time::now ();
        elapsed_time = t1-t0;
      }
      break;
      case Experiments::TPETRA_DOT:
      {
        MPI_Barrier(MPI_COMM_WORLD);
        const auto t0 = Time::now ();
        x.dot(y, dots);
        solution.multiply(Teuchos::TRANS,
                          Teuchos::NO_TRANS,
                          1.0,
                          x,y,
                          0.0);
        const auto t1 = Time::now ();
        elapsed_time = t1-t0;
      }
      break;
      } // switch

      if (! (skip_first && nsample == 0) )
        exp_data[e].push_back(std::chrono::duration_cast<clock_resolution>(elapsed_time).count ());
    } //randomized loop
  } // samples
  
  if (comm->getRank () == 3) {
    if (!header_flag)
      out << "x.dot(y) = " << dots[0] << ", cublas = " << cublas_dot_result << ", kk = " << kk_dot_result << "\n";
  }

  analyze_timings(exp_data, exps, exps_to_string, comm, out);
}

int
main (int argc, char *argv[])
{
  MPI_Init(&argc,&argv);
  const int stride = 1;
  Tpetra::ScopeGuard tpetraScope (&argc, &argv);
  {
    auto comm = Tpetra::getDefaultComm ();
    for (int N = 10000000; N <= 20000000; N += 10000000) {
      evaluate_dot (N, stride, comm, std::cout);
    }
  }
  MPI_Finalize();
  return 0;
}
