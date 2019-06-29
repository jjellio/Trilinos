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


#include <Tpetra_Core.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Vector.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <cublas_v2.h>
#include <KokkosBlas.hpp>
#include <random>    // random
#include <algorithm> // shuffle

//#define USE_TIMEOFDAY
#include "descriptive_stats/descriptive_stats.hpp"

#if defined(ENABLE_SYSTEM_BLAS)
extern "C" double cblas_ddot (const int n, const double *x, const int incx, const double *y, const int incy);
extern "C" int openblas_set_num_threads(int);
#endif

std::deque<std::string> base_fields {
    "host",
    "num_procs",
    "rank",
    #if defined(KOKKOS_HAVE_CUDA)
     #warning Adding Cuda Device ID to csv output options 
    "device_id",
    #endif
    "median",
    "mad",
    "mean",
    "min_value",
    "max_value",
    "q1",
    "q3",
    "sample_stddev",
    "num_observations",
    "perc_1",
    "perc_25",
    "perc_50",
    "perc_75",
    "perc_90",
    "perc_99"
};

#include <cxxabi.h>
template<typename T>
std::string type_name(const T& v)
{
    int status;
    std::string tname = typeid(v).name();
    char *demangled_name = abi::__cxa_demangle(tname.c_str(), NULL, NULL, &status);
    if(status == 0) {
        tname = demangled_name;
        std::free(demangled_name);
    }   
    return tname;
}


bool printed_header_flag = false;

template<typename ds_array_type, typename experiment_type>
void analyze_timings(      std::map<experiment_type, ds_array_type>& exp_data,
                     const std::vector<experiment_type>& exps,
                           std::map<experiment_type, std::string>& exp_to_string,
                     const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
                     std::ostream& out)
{
  // setup the stats stuff
  using ds_map_type = DescriptiveStats::descriptive_stat_map_type;
  using DescriptiveStats::get_descriptive_stats;
  using DescriptiveStats::descriptive_stats_csv_str;
  using DescriptiveStats::descriptive_stat_map_string_type;
  using DescriptiveStats::stat_map_to_string_map;

  std::ostringstream oss;
 
  std::shared_ptr< std::deque<std::string> > output_fields (new std::deque<std::string>(base_fields));

  // pad the 'id' field so we get aligned text output
  int max_length = -1;
  for (const auto& e : exps) {
    max_length = std::max(max_length, (int) exp_to_string[e].length());
  }

  // add a field to the output
  output_fields->emplace(output_fields->begin(), "id");

  for (const auto& e : exps) {

    std::stringstream ss;
    // map of statistics
    ds_map_type stats;


    // compute the stats
    get_descriptive_stats(exp_data[e], stats, DescStats_TICK_RESOLUTION);
    descriptive_stat_map_string_type stats_str;
    stat_map_to_string_map(stats, stats_str);
    std::string padding_str(max_length - exp_to_string[e].length(), ' ');
    stats_str["id"] = padding_str + exp_to_string[e];

    // dump the CSV banner first
    bool print_header = false;
    if ( (comm->getRank () == 0) && (!printed_header_flag)){
      printed_header_flag = true;
      print_header = true;
    }

    descriptive_stats_csv_str (oss, stats_str, output_fields, print_header);

    //print_descriptive_stats (oss, stats, timer_ratio, exp_to_string[e], true);
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

  Teuchos::RCP<const map_type> locallyreplicated_map (new map_type (1,
                                                      0,
                                                      comm,
                                                      Tpetra::LocallyReplicated));

  // multidimensional arrays to store the results
  vector_type kk_gemv_solution_mv(locallyreplicated_map, 1);

  // grab the device views so we can call cublas directly
  typedef typename vector_type::dual_view_type dual_view_type;
  typedef typename dual_view_type::t_dev device_view_type;
  auto cuX = subview ( x.getLocalView<Kokkos::Cuda> (), Kokkos::ALL (), 0);
  auto cuY = subview ( y.getLocalView<Kokkos::Cuda> (), Kokkos::ALL (), 0);

  auto cuX_rk2 = subview ( x.getLocalView<Kokkos::Cuda> (), Kokkos::ALL (), Kokkos::ALL());
  auto kk_gemv_sol_view = subview( kk_gemv_solution_mv.getLocalView<Kokkos::Cuda> (), Kokkos::ALL (), 0);

  cublasHandle_t cublas_handle;
  const auto cublas_status = cublasCreate(&cublas_handle);

  using scalar_type = typename vector_type::scalar_type;
  typedef struct dot_data_pack {
    scalar_type * x = nullptr;
    scalar_type * y = nullptr;
    scalar_type * sol = nullptr;
    scalar_type sol_stack = -1.0;
    ~dot_data_pack () {
      if (x) cudaFree(x);
      if (y) cudaFree(y);
      if (sol) cudaFree(sol);
    }
  } dot_data_pack_t;

  dot_data_pack_t cu_managed;
  dot_data_pack_t cu_hostpinned;
  dot_data_pack_t cu_reg;


  // -- create data using native cuda tools
  cudaMallocManaged(&cu_managed.x, local_n*sizeof(scalar_type));
  cudaMallocManaged(&cu_managed.y, local_n*sizeof(scalar_type));
  cudaMallocManaged(&cu_managed.sol, sizeof(scalar_type));

  cudaMalloc(&cu_reg.x, local_n*sizeof(scalar_type));
  cudaMalloc(&cu_reg.y, local_n*sizeof(scalar_type));
  cudaMalloc(&cu_reg.sol, sizeof(scalar_type));

  cudaMallocHost(&cu_hostpinned.x, local_n*sizeof(scalar_type));
  cudaMallocHost(&cu_hostpinned.y, local_n*sizeof(scalar_type));
  cudaMallocHost(&cu_hostpinned.sol, sizeof(scalar_type));

  std::vector<scalar_type> host_x;
  std::vector<scalar_type> host_y;
  host_x.reserve(local_n);
  host_x.resize(local_n);
  host_y = host_x;

  //////////////////////////////////////////////////////////////////////
  // Fill the Vector with a single number, or with random numbers
  //////////////////////////////////////////////////////////////////////

  // Set the entries of x to (pseudo)random numbers.  Please don't
  // consider this a good parallel pseudorandom number generator.
  x.randomize (0,4);
  y.randomize (0,4);

  ////x.putScalar(1.0);
  ////y.putScalar(1.0);
  //if (!header_flag) {
  //  out << x.description() << "\n"
  //      << cuY.extent(0) << "\n";
  //}

  // copy the values in x/y to our native memory locations
  cudaMemcpy (cu_hostpinned.x, cuX.data(), local_n*sizeof(scalar_type), cudaMemcpyDeviceToDevice);
  cudaMemcpy (cu_hostpinned.y, cuY.data(), local_n*sizeof(scalar_type), cudaMemcpyDeviceToDevice);

  cudaMemcpy (cu_managed.x, cu_hostpinned.x, local_n*sizeof(scalar_type), cudaMemcpyHostToHost);
  cudaMemcpy (cu_managed.y, cu_hostpinned.y, local_n*sizeof(scalar_type), cudaMemcpyHostToHost);

  cudaMemcpy (cu_reg.x, cuX.data(), local_n*sizeof(scalar_type), cudaMemcpyDeviceToDevice);
  cudaMemcpy (cu_reg.y, cuY.data(), local_n*sizeof(scalar_type), cudaMemcpyDeviceToDevice);

  memcpy(host_x.data(), cu_hostpinned.x, local_n*sizeof(scalar_type));
  memcpy(host_y.data(), cu_hostpinned.y, local_n*sizeof(scalar_type));

  std::vector<double> dots;
  dots.resize(1);

  double * cublas_dot_result;
  cudaMallocManaged(&cublas_dot_result, sizeof(double));
  double * cublas_gemv_result;
  cudaMallocManaged(&cublas_gemv_result, sizeof(double));
   
  double cublas_dot_result_host = std::numeric_limits<scalar_type>::quiet_NaN();
  cublas_gemv_result[0] = std::numeric_limits<scalar_type>::quiet_NaN();
  double kk_dot_result     = std::numeric_limits<scalar_type>::quiet_NaN();
  double kk_gemv_result    = std::numeric_limits<scalar_type>::quiet_NaN();
  double cblas_result      = std::numeric_limits<scalar_type>::quiet_NaN();

  cublas_dot_result[0] = cublas_dot_result_host;

  enum class Experiments { CUBLAS_LOCAL,
                           CUBLAS_NATIVE_MEM,
                           CUBLAS_HOST_MEM,
                           CUBLAS_UVM_MEM,
                           KK_LOCAL,
                           ALL_REDUCE,
                           TPETRA_DOT,
                           CBLAS1,
                           CBLAS2,
                           CBLAS4,
                           CBLAS8,
                           TIMER,
                           CUBLAS_GEMV,
                           KK_GEMV
 };

  // create a mapping for experiments to strings
  std::map<Experiments,std::string> exps_to_string;
  exps_to_string[Experiments::CUBLAS_LOCAL] = std::to_string(local_n) + "-CuBLAS_dot";
  exps_to_string[Experiments::CUBLAS_GEMV] = std::to_string(local_n) + "-CuBLAS_gemv";
  exps_to_string[Experiments::KK_GEMV] = std::to_string(local_n) + "-KK_gemv";
  exps_to_string[Experiments::CUBLAS_NATIVE_MEM] = std::to_string(local_n) + "-CuBLAS_dot_natmem";
  exps_to_string[Experiments::CUBLAS_HOST_MEM] = std::to_string(local_n) + "-CuBLAS_dot_nathost";
  exps_to_string[Experiments::CUBLAS_UVM_MEM] = std::to_string(local_n) + "-CuBLAS_dot_natuvm";
  exps_to_string[Experiments::KK_LOCAL] = std::to_string(local_n) +"-KK_dot";
  exps_to_string[Experiments::ALL_REDUCE] = "1-Allreduce";
  exps_to_string[Experiments::TPETRA_DOT] = std::to_string(local_n) + "-Tpetra_dot";
  exps_to_string[Experiments::CBLAS1] = std::to_string(local_n) + "-Cblas_ddot-1th";
  exps_to_string[Experiments::CBLAS2] = std::to_string(local_n) + "-Cblas_ddot-2th";
  exps_to_string[Experiments::CBLAS4] = std::to_string(local_n) + "-Cblas_ddot-4th";
  exps_to_string[Experiments::CBLAS8] = std::to_string(local_n) + "-Cblas_ddot-8th";
  exps_to_string[Experiments::TIMER] = "Timer";

  std::vector<Experiments> exps;
  exps.push_back(Experiments::TIMER);
//exps.push_back(Experiments::CBLAS1);
//exps.push_back(Experiments::CBLAS2);
//exps.push_back(Experiments::CBLAS4);
//exps.push_back(Experiments::CBLAS8);
  exps.push_back(Experiments::CUBLAS_LOCAL);
  exps.push_back(Experiments::CUBLAS_NATIVE_MEM);
  exps.push_back(Experiments::CUBLAS_HOST_MEM);
  exps.push_back(Experiments::CUBLAS_UVM_MEM);
  exps.push_back(Experiments::KK_LOCAL);
  exps.push_back(Experiments::ALL_REDUCE);
  exps.push_back(Experiments::TPETRA_DOT);
  exps.push_back(Experiments::CUBLAS_GEMV);
  exps.push_back(Experiments::KK_GEMV);


  // randomize the kernel runs
  std::random_device rd;
  std::mt19937 g(rd());
  std::vector<Experiments> randomized_exps (exps);

  std::map<Experiments, ds_array> exp_data;

  const bool skip_first = true;

  int num_samples = 1000;

  if (skip_first) num_samples++;

  DescStats_TIMEPOINT t0, t1;
  DescStats_TIMEPOINT_DELTA elapsed_time;
  double capture = 0.0;

  for(int nsample=0; nsample < num_samples; ++nsample) {  
    std::shuffle(randomized_exps.begin(), randomized_exps.end(), g);
    MPI_Bcast(randomized_exps.data(), randomized_exps.size()*sizeof(randomized_exps[0]), MPI_BYTE,
              0, MPI_COMM_WORLD);

    dots[0] = 0.0;
    kk_dot_result = 0.0;

    for (const auto& e : randomized_exps) {
      switch(e) {
      case Experiments::TIMER:
      {
        DescStats_TIMEPOINT dummy;
        DescStats_TICK(t0);
          DescStats_TICK(dummy);
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
      }
      break;
      case Experiments::CBLAS1:
      {
        #if defined(ENABLE_SYSTEM_BLAS)
        openblas_set_num_threads(1);
        DescStats_TICK(t0);
        //__sync_synchronize();
        cblas_result = cblas_ddot (local_n, host_x.data(), stride,
                                            host_y.data(), stride);
        //__sync_synchronize();
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
        capture += cblas_result;
        #endif
      }
      break;
      case Experiments::CBLAS2:
      {
        #if defined(ENABLE_SYSTEM_BLAS)
        openblas_set_num_threads(2);
        DescStats_TICK(t0);
        //__sync_synchronize();
        cblas_result = cblas_ddot (local_n, host_x.data(), stride,
                                            host_y.data(), stride);
        //__sync_synchronize();
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
        capture += cblas_result;
        #endif
      }
      break;
      case Experiments::CBLAS4:
      {
        #if defined(ENABLE_SYSTEM_BLAS)
        openblas_set_num_threads(4);
        DescStats_TICK(t0);
        //__sync_synchronize();
        cblas_result = cblas_ddot (local_n, host_x.data(), stride,
                                            host_y.data(), stride);
        //__sync_synchronize();
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
        capture += cblas_result;
        #endif
      }
      break;
      case Experiments::CBLAS8:
      {
        #if defined(ENABLE_SYSTEM_BLAS)
        openblas_set_num_threads(8);
        DescStats_TICK(t0);
        //__sync_synchronize();
        cblas_result = cblas_ddot (local_n, host_x.data(), stride,
                                            host_y.data(), stride);
        //__sync_synchronize();
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
        capture += cblas_result;
        #endif
      }
      break;
      case Experiments::KK_LOCAL:
      {
        DescStats_TICK(t0);
        //__sync_synchronize();
        kk_dot_result = KokkosBlas::dot(cuX,cuY);
        //__sync_synchronize();
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
        capture += kk_dot_result;
      }
      break;
      case Experiments::CUBLAS_NATIVE_MEM:
      {
        DescStats_TICK(t0);
        //__sync_synchronize();
        auto cuRC = cublasDdot (cublas_handle,
                                local_n,
                                cu_reg.x, stride,
                                cu_reg.y, stride,
                                &cu_reg.sol_stack);
        //__sync_synchronize();
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
        capture += cu_reg.sol_stack;
      }
      break;
      case Experiments::CUBLAS_HOST_MEM:
      {
        DescStats_TICK(t0);
        //__sync_synchronize();
        auto cuRC = cublasDdot (cublas_handle,
                                local_n,
                                cu_hostpinned.x, stride,
                                cu_hostpinned.y, stride,
                                &cu_hostpinned.sol_stack);
        //__sync_synchronize();
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
        capture += cu_hostpinned.sol_stack;
      }
      break;
      case Experiments::CUBLAS_UVM_MEM:
      {
        DescStats_TICK(t0);
        //__sync_synchronize();
        auto cuRC = cublasDdot (cublas_handle,
                                local_n,
                                cu_managed.x, stride,
                                cu_managed.y, stride,
                                &cu_managed.sol_stack);
        //__sync_synchronize();
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
        capture += cu_managed.sol_stack;
      }
      break;
      case Experiments::CUBLAS_LOCAL:
      {
        DescStats_TICK(t0);
        //__sync_synchronize();
        auto cuRC = cublasDdot (cublas_handle,
                                local_n,
                                cuX.data(), stride,
                                cuY.data(), stride,
                                &cublas_dot_result_host);
        //__sync_synchronize();
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
      }
      break;
      case Experiments::ALL_REDUCE:
      {
        MPI_Barrier(MPI_COMM_WORLD);
        DescStats_TICK(t0);
        //__sync_synchronize();
        MPI_Allreduce(MPI_IN_PLACE, &cublas_dot_result_host, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        //__sync_synchronize();
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
      }
      break;
      case Experiments::TPETRA_DOT:
      {
        MPI_Barrier(MPI_COMM_WORLD);
        DescStats_TICK(t0);
        //__sync_synchronize();
        x.dot(y, dots);
        //__sync_synchronize();
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
      }
      break;
      case Experiments::CUBLAS_GEMV:
      {
        scalar_type alpha = 1.0;
        scalar_type beta = 0.0;
        DescStats_TICK(t0);
        //__sync_synchronize();
        auto cuRC = cublasDgemv(cublas_handle, CUBLAS_OP_T,
                           local_n, 1,
                           &alpha,
                           cuX.data(), local_n,
                           cuY.data(), 1,
                           &beta,
                           cublas_gemv_result, 1);
        //__sync_synchronize();
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
      }
      break;
      case Experiments::KK_GEMV:
      {
        const char kk_trans = 'T';
        scalar_type alpha = 1.0;
        scalar_type beta = 0.0;
        DescStats_TICK(t0);
        __sync_synchronize();
        KokkosBlas::gemv (&kk_trans, alpha, cuX_rk2, cuY, beta, kk_gemv_sol_view);
        __sync_synchronize();
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
        //capture += kk_gemv_result;
      }
      break;
      default: std::cerr << "ERROR\n";
      } // switch

      if (skip_first && nsample ==  0) continue;

      exp_data[e].push_back(DescStats_TICK_TO_DOUBLE(elapsed_time));
    } //randomized loop
  } // samples

  kk_gemv_result = kk_gemv_sol_view(0);

  // always use the output variables from above so they can't be optimized out
  std::ostringstream oss;
  oss << "x.dot(y) = " << dots[0] << ", cublas = " << cublas_dot_result_host << ", kk = " << kk_dot_result
      << ", cublas_reg = " << cu_reg.sol_stack << ", cuhost = " << cu_hostpinned.sol_stack << ", cuUVM = " << cu_managed.sol_stack
      << ", cblas1 = " << cblas_result
      << ", kk_gemv = " << kk_gemv_result
      << ", cub_gemv = " << cublas_gemv_result << ", kk_gemv = " << kk_gemv_result
      << "\n"
      << capture << "\n";

  if (comm->getRank () == 3) {
    out << oss.str ();
  }

  cudaFree(cublas_dot_result);
  cudaFree(cublas_gemv_result);

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

  using ds_array = std::vector<double>;

  const int myRank = comm->getRank ();

  using map_type = Tpetra::Map<>;
  using vector_type = Tpetra::MultiVector<double,int>;

  using global_ordinal_type = vector_type::global_ordinal_type;

  // perfectly partition the vector
  const Tpetra::global_size_t numGlobalEntries = comm->getSize () * local_n * stride;

  // Construct a Map that puts the same number of equations on each
  // MPI process.
  RCP<const map_type> contigMap =
    rcp (new map_type (numGlobalEntries, local_n, 0, comm));

  // create two vectors
  vector_type x (contigMap, num_mv_left);
  vector_type y (contigMap, num_mv_right);

  // grab the device views so we can call cublas directly
  typedef typename vector_type::dual_view_type dual_view_type;
  typedef typename dual_view_type::t_dev device_view_type;
  auto cuX = subview ( x.getLocalView<Kokkos::Cuda> (),
                       std::make_pair(0, local_n),
                       std::make_pair(0, num_mv_left));
  //auto cuY = subview ( y.getLocalView<Kokkos::Cuda> (), Kokkos::ALL (), 0);
  auto cuY = subview ( y.getLocalView<Kokkos::Cuda> (),
                       std::make_pair(0, local_n),
                       std::make_pair(0, num_mv_right));

  Teuchos::RCP<const map_type> locallyreplicated_map (new map_type (num_mv_left,
                                                      0,
                                                      comm,
                                                      Tpetra::LocallyReplicated));

  // multidimensional arrays to store the results
  vector_type solution(locallyreplicated_map, num_mv_right);
  vector_type cublas_solution_mv(locallyreplicated_map, num_mv_right);
  vector_type kk_solution_mv(locallyreplicated_map, num_mv_right);
  auto cublas_solution_view =cublas_solution_mv.getLocalView<Kokkos::Cuda> ();
//  auto cublas_solution_view =  subview (cublas_solution_mv.getLocalView<Kokkos::Cuda> (),
//                                        std::make_pair(0, local_n),
//                                        std::make_pair(0, num_mv_right));
  //auto kk_solution_view =  subview (kk_solution_mv.getLocalView<Kokkos::Cuda> (), Kokkos::ALL (), 0);
  auto kk_solution_view = kk_solution_mv.getLocalView<Kokkos::Cuda> ();

  cublasHandle_t cublas_handle;
  const auto cublas_status = cublasCreate(&cublas_handle);


  //////////////////////////////////////////////////////////////////////
  // Fill the Vector with a single number, or with random numbers
  //////////////////////////////////////////////////////////////////////

  // Set the entries of x to (pseudo)random numbers.  Please don't
  // consider this a good parallel pseudorandom number generator.
  x.randomize (0,4);
  y.randomize (0,4);
  x.putScalar(1.0); y.putScalar(1.0);

  //  out << "mv_left: " << num_mv_left << " x local_n: " << local_n << "\n"
  //      << "mv_right: " << num_mv_right << "\n"
  //      << x.description() << "\n"
  //      << "x: " << type_name(x) << "\n"
  //      << "cuX: " << type_name(cuX) << "\n"
  //      << "cuY: " << type_name(cuY) << "\n"
  //      << "cublas_solution_mv: " << type_name(cublas_solution_mv) << "\n"
  //      << "cublas_solution_view: " << type_name(cublas_solution_view) << "\n"
  //      << "X: " << cuX.extent(0) << " x " << cuX.extent(1) << "\n"
  //      << "Y: " << cuY.extent(0) << " x " << cuY.extent(1) << "\n"
  //      << "cuSol: " << cublas_solution_view.extent(0) << " x " << cublas_solution_view.extent(1) << "\n";

  char xTrans = 'T';
  char yTrans = 'N';
  cublasOperation_t xTrans_cu = CUBLAS_OP_T;
  cublasOperation_t yTrans_cu = CUBLAS_OP_N;
  const double alpha = 1.0;
  const double beta = 0.0;
  cublas_solution_mv.putScalar(std::numeric_limits<double>::quiet_NaN());
  std::vector<double> gemm_host_results;
  gemm_host_results.reserve(num_mv_left*num_mv_right);
  gemm_host_results.resize(num_mv_left*num_mv_right);

  // what to test
  enum class Experiments { CUBLAS_LOCAL, KK_LOCAL, ALL_REDUCE_CUDA, ALL_REDUCE_CUDA_DEST_HOST, TPETRA, ALL_REDUCE_HOST_COPY };
  std::vector<Experiments> exps;
  exps.push_back(Experiments::CUBLAS_LOCAL);
  exps.push_back(Experiments::KK_LOCAL);
  exps.push_back(Experiments::ALL_REDUCE_CUDA);
  exps.push_back(Experiments::ALL_REDUCE_CUDA_DEST_HOST);
  exps.push_back(Experiments::TPETRA);
  exps.push_back(Experiments::ALL_REDUCE_HOST_COPY);

  // create a mapping for experiments to strings
  std::map<Experiments,std::string> exps_to_string;
  exps_to_string[Experiments::CUBLAS_LOCAL] = std::to_string(local_n) + "-CuBLAS_gemm";
  exps_to_string[Experiments::KK_LOCAL] = std::to_string(local_n) +"-KK_gemm";
  exps_to_string[Experiments::ALL_REDUCE_CUDA] = std::to_string(num_mv_left*num_mv_right) + "Allreduce-cuda";
  exps_to_string[Experiments::ALL_REDUCE_CUDA_DEST_HOST] = std::to_string(num_mv_left*num_mv_right) + "Allreduce-cuda-dest-host";
  exps_to_string[Experiments::ALL_REDUCE_HOST_COPY] = std::to_string(num_mv_left*num_mv_right)  + "allreduce-copy-host";
  exps_to_string[Experiments::TPETRA] = std::to_string(local_n) + "-Tpetra_multiply";


  // randomize the kernel runs
  std::random_device rd;
  std::mt19937 g(rd());
  std::vector<Experiments> randomized_exps (exps);

  std::map<Experiments, ds_array> exp_data;

  const bool skip_first = true;

  DescStats_TIMEPOINT t0, t1;
  DescStats_TIMEPOINT_DELTA elapsed_time;

  int num_samples = 1000;

  if (skip_first) num_samples++;

  for(int nsample=0; nsample < num_samples; ++nsample) {  
    // randomize the order of kernel launches
    std::shuffle(randomized_exps.begin(), randomized_exps.end(), g);
    MPI_Bcast(randomized_exps.data(), randomized_exps.size()*sizeof(randomized_exps[0]), MPI_BYTE,
              0, MPI_COMM_WORLD);

    for (const auto& e : randomized_exps) {
      // storage for the delta of two timepoints
      switch(e) {
      case Experiments::KK_LOCAL:
      {
        DescStats_TICK(t0);
        //__sync_synchronize();
        KokkosBlas::gemm (&xTrans, &yTrans, alpha, cuX, cuY, beta, kk_solution_view);
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
      }
      break;
      case Experiments::CUBLAS_LOCAL:
      {
        int m = cuX.extent(1);
        int n = cuY.extent(1);
        int k = cuX.extent(0);
        int lda = cuX.extent(0);
        int ldb = cuY.extent(0);
        int ldc = cublas_solution_view.extent(0);

        DescStats_TICK(t0);
        //__sync_synchronize();
        cublasDgemm(cublas_handle,
                           xTrans_cu, yTrans_cu,
                           m,n,k,
                           &alpha,
                           cuX.data(), lda,
                           cuY.data(), ldb,
                           &beta,
                           cublas_solution_view.data(),
                           ldc);
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
      }
      break;
      case Experiments::ALL_REDUCE_HOST_COPY:
      {
        double * d;
        MPI_Barrier(MPI_COMM_WORLD);
        DescStats_TICK(t0);
        //__sync_synchronize();
        {
        //cublas_solution_mv.sync_host();
        d = cublas_solution_mv.getLocalView<Kokkos::Serial> ().data();
        MPI_Allreduce(MPI_IN_PLACE, d,
                      //gemm_host_results.data(),
                      num_mv_left*num_mv_right, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        //MPI_Allreduce(cublas_solution_view.data(), gemm_host_results.data(), num_mv_left*num_mv_right, MPI_DOUBLE,
        //              MPI_SUM, MPI_COMM_WORLD);
        }
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
        gemm_host_results.assign(d, d+(num_mv_left*num_mv_right)); 
      }
      break;
      case Experiments::ALL_REDUCE_CUDA_DEST_HOST:
      {
        MPI_Barrier(MPI_COMM_WORLD);
        DescStats_TICK(t0);
        MPI_Allreduce(cublas_solution_view.data(), gemm_host_results.data(), num_mv_left*num_mv_right, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
      }
      case Experiments::ALL_REDUCE_CUDA:
      {
        MPI_Barrier(MPI_COMM_WORLD);
        DescStats_TICK(t0);
        MPI_Allreduce(MPI_IN_PLACE, cublas_solution_view.data(), num_mv_left*num_mv_right, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
      }
      break;
      case Experiments::TPETRA:
      {
        MPI_Barrier(MPI_COMM_WORLD);
        DescStats_TICK(t0);
        solution.multiply(Teuchos::TRANS,
                          Teuchos::NO_TRANS,
                          alpha,
                          x,y,
                          beta);
        DescStats_TICK(t1);
        DescStats_TICKDIFF(t0,elapsed_time,t1);
      }
      break;
      } // switch

      if (! (skip_first && nsample == 0) )
        exp_data[e].push_back(DescStats_TICK_TO_DOUBLE(elapsed_time));
    } //randomized loop
  } // samples
  
  if (comm->getRank () == 3) {
    std::stringstream oss;
    {
      oss << "Tpetra-mult = " << "\n";
      auto s = solution.getLocalView<Kokkos::Serial> ();
      for(int i=0; i < num_mv_left; ++i){
        oss << i << ": ";
        for(int k=0; k < num_mv_right; ++k){
          oss << s(i,k) << " ";
        }
        oss  << "\n";
      }
    }
    {
      oss << "KK-gemm = " << "\n";
      auto s = kk_solution_mv.getLocalView<Kokkos::Serial> ();
      for(int i=0; i < num_mv_left; ++i){
        oss << i << ": ";
        for(int k=0; k < num_mv_right; ++k){
          oss << s(i,k) << " ";
        }
        oss  << "\n";
      }
    }
    {
      oss << "cu-gemm = " << "\n";
      auto s = cublas_solution_mv.getLocalView<Kokkos::Serial> ();
      for(int i=0; i < num_mv_left; ++i){
        oss << i << ": ";
        for(int k=0; k < num_mv_right; ++k){
          oss << s(i,k) << " ";
        }
        oss  << "\n";
      }
      oss << "cu-gemm-host-all = " << "\n";
      for(int i=0; i < num_mv_left; ++i){
        oss << i << ": ";
        for(int k=0; k < num_mv_right; ++k){
          oss << gemm_host_results[i*num_mv_right + k] << " ";
        }
        oss  << "\n";
      }
    }
    out << oss.str();
    out.flush(); 
  }
  MPI_Barrier(MPI_COMM_WORLD);
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
    if (comm->getRank() == 0) std::cerr << "After Initialize\n";
    MPI_Barrier(MPI_COMM_WORLD);


    for (int N = 1000000; N < 2000000; N += 1000000) {
      evaluate_dot (N, stride, comm, std::cout);
    }

    evaluate_gemm(10000, 1,
                  2,
                  1,
                  comm,
                  std::cout);

    //std::default_random_engine generator;
    //std::normal_distribution<double> distribution(1000.0,5000.0);
    //std::vector<double> normal_samples;
    //for (int i=0; i < 10000; ++i) {
    //  normal_samples.push_back(distribution(generator));
    //}
    //using DescriptiveStats::get_descriptive_stats;
    //using DescriptiveStats::descriptive_stats_csv_str;
    //using DescriptiveStats::descriptive_stat_map_string_type;
    //using DescriptiveStats::stat_map_to_string_map;
    //using DescriptiveStats::descriptive_stat_map_type;
    //descriptive_stat_map_type stats;
    //get_descriptive_stats(normal_samples, stats);
    //descriptive_stat_map_string_type stats_str;
    //stat_map_to_string_map(stats, stats_str);
    //std::ostringstream oss;
    //descriptive_stats_csv_str (oss, stats_str, nullptr, true);
    //std::cerr << oss.str();
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}

//class DotExperiment {
//public:
//  using stat_array_type = std::vector<double>;
//
//  void run(const int n) = 0;
//  void getTimings( std::map< std::string, stat_array_type>& timing_data) = 0;
//  void getName() = 0;
//
//};
//
//template<typename view_type>>
//class KokkosBlasDot :
//  public DotExperiment {
//
//private:
//
//  using stat_array_type = DotExperiment::stat_array_type;
//
//  view_type& x;
//  view_type& y;
//  stat_array_type timings;
//  const std::string name_;
//
//public:
//  KokkosBlasDot (view_type& x, view_type& y)
//    : x (x_), y (y_), name_("KokkosBlasDot")
//  { }
//
//  virtual
//  void getName () { return(name_); }
//
//  virtual
//  void getTimings( std::map< std::string, stat_array_type>& timing_data) {
//    timing_data[name_] = timings;
//  }
//
//  virtual
//  void run (const int n)
//  {
//    const auto t0 = Time::now ();
//      const auto kk_dot_result = KokkosBlas::dot(x,y);
//    const auto t1 = Time::now ();
//    elapsed_type DescStats_TICKDIFF(t0,elapsed_time,t1);
//    add_timing(timings, elapsed_time);
//  }
//}
