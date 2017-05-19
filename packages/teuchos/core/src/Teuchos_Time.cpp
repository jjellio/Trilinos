// @HEADER
// ***********************************************************************
//
//                    Teuchos: Common Tools Package
//                 Copyright (2004) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// ***********************************************************************
// @HEADER

// Kris
// 07.08.03 -- Move into Teuchos package/namespace

#include "Teuchos_Time.hpp"

#ifdef TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS
namespace {

inline double rel_error (double truth, double obs) {
  return truth == 0.0 ? 0.0 : (std::abs(truth - obs)/truth);
}

inline double get_median (const size_t s, const size_t e, const std::vector<double>& data) {
  if (e < s) return (0.0);

  const size_t N = e - s + 1;

  if (N % 2 == 0) {
//    std::cout << "Median midpoint : " << (s + N/2) << ", " << (e- N/2)
//              << " : " << ((data[s + N/2] + data[e - N/2])/2.0)
//              << ", interval[" << s << "," << e << "]"
//              << ", size = " << N
//              << std::endl;
    return ((data[s + N/2] + data[e - N/2])/2.0);
  }
  else {
//    std::cout << "Median exact: " << (s + N/2) << ", " << (e- N/2)
//              << " : " << (data[s + N/2])
//              << ", interval[" << s << "," << e << "]"
//              << ", size = " << N
//              << std::endl;
    return (data[s + N/2]);
  }
}

void get_descriptive_stats (std::vector<double>& observations, const size_t observation_count,
                            Teuchos::Time::descriptive_stat_map_type& stat_map) {

  if (observation_count < 1) return;

  stat_map.clear ();
  if (observation_count == 1) {
    stat_map["Median"] = observations[0];

    stat_map["1st-excluded-median"] = observations[0];
    stat_map["3rd-excluded-median"] = observations[0];


    stat_map["1st-included-median"] = observations[0];
    stat_map["3rd-included-median"] = observations[0];

    stat_map["IQR-excluded-median"] = stat_map["3rd-excluded-median"] - stat_map["1st-excluded-median"];
    stat_map["IQR-included-median"] = stat_map["3rd-included-median"] - stat_map["1st-included-median"];
    stat_map["Min Value"] = observations[0];
    stat_map["Max Value"] = observations[observation_count-1];
    stat_map["Mean"] = observations[0];
    stat_map["Sample StdDev"] = 0.0;
    stat_map["Total Time"] = observations[0];
    stat_map["Num Observations"] = observation_count;

    return;
  }

  #ifdef TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS_CHECK_FP
    std::cout << "Some Timer: Count = " << observation_count << std::endl;

    double naive_total_time = observations[0];
    for (size_t i=1; i < observation_count; ++i) {
      naive_total_time += observations[i];
    }

    double naive_mean = naive_total_time/observation_count;

    double ss = 0.0;
    for (size_t i=0; i < observation_count; ++i) {
      ss += (observations[i] - naive_mean) * (observations[i] - naive_mean);
      //std::cout << "naive[" << i << "]: " << observations[i] << std::endl;
    }

    double naive_variance = (observation_count > 1) ? ss / (observation_count -1) : 0.0;
    double naive_stddev = std::sqrt(naive_variance);
  #endif
  // sort the observations
  std::sort(observations.begin (), observations.begin () + observation_count);

  // The following algorithm is taken from
  // https://www.johndcook.com/blog/standard_deviation/
  // which is taken from Knuth's book.
  // The idea is to compute the variance
  double oldMean = observations[0];
  double oldSS = 0.0;
  double newMean = 0.0;
  double newSS = 0.0;
  double total_time_sorted = observations[0];
  for (size_t i=1; i < observation_count; ++i) {

    // See Knuth TAOCP vol 2, 3rd edition, page 232
    newMean = oldMean + (observations[i] - oldMean) / (i+1);
    newSS = oldSS + (observations[i] - oldMean) * (observations[i] - newMean);

    // set up for next iteration
    oldMean = newMean;
    oldSS = newSS;

    total_time_sorted += observations[i];
  }

  double stable_mean = newMean;
  double stable_variance = (observation_count > 1) ? newSS / (observation_count - 1) : 0.0;
  double stable_stddev = std::sqrt(stable_variance);

  // compute the quartiles
  // compute the median first
  stat_map["Median"] = get_median (0, observation_count-1, observations);

  // compute the 1st and 3rd quartiles
  if (observation_count % 2 == 0) {
    stat_map["1st-excluded-median"] = get_median (0, observation_count/2 - 1, observations);
    stat_map["3rd-excluded-median"] = get_median (observation_count/2, observation_count-1, observations);


    stat_map["1st-included-median"] = get_median (0, observation_count/2 - 1, observations);
    stat_map["3rd-included-median"] = get_median (observation_count/2, observation_count-1, observations);
  }
  else {
    // there is an even number of data points, there are a few ways to compute the quartiles
    // 1) include the median, 2) exclude the median, or 3) average

    stat_map["1st-excluded-median"] = get_median (0, observation_count/2 - 1, observations);
    stat_map["3rd-excluded-median"] = get_median (observation_count/2 + 1, observation_count-1, observations);


    stat_map["1st-included-median"] = get_median (0, observation_count/2, observations);
    stat_map["3rd-included-median"] = get_median (observation_count/2, observation_count-1, observations);
  }
  stat_map["IQR-excluded-median"] = stat_map["3rd-excluded-median"] - stat_map["1st-excluded-median"];
  stat_map["IQR-included-median"] = stat_map["3rd-included-median"] - stat_map["1st-included-median"];
  stat_map["Min Value"] = observations[0];
  stat_map["Max Value"] = observations[observation_count-1];
  stat_map["Mean"] = stable_mean;
  stat_map["Sample StdDev"] = stable_stddev;
  stat_map["Total Time"] = total_time_sorted;
  stat_map["Num Observations"] = observation_count;

  #ifdef TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS_CHECK_FP
    std::cout << "Some Timer: Mean          = " << naive_mean
              << "\t\t\t" << rel_error(stable_mean, naive_mean)
              << std::endl;
    std::cout << "Some Timer: Mean (stable) = " << stable_mean << std::endl;

    std::cout << "Some Timer: Var          = " << naive_variance
              << "\t\t\t" << rel_error(stable_variance, naive_variance)
              << std::endl;
    std::cout << "Some Timer: Var (stable) = " << stable_variance << std::endl;

    std::cout << "Some Timer: StdDev          = " << naive_stddev
              << "\t\t\t" << rel_error(stable_stddev, naive_stddev)
              << std::endl;
    std::cout << "Some Timer: StdDev (stable) = " << stable_stddev << std::endl;

    std::cout << "Some Timer: Total Time          = " << naive_total_time
              << "\t\t\t" << rel_error(total_time_sorted, naive_total_time)
              << std::endl;
    std::cout << "Some Timer: Total Time (stable) = " << total_time_sorted << std::endl;
  #endif
}

} // end of anon namespace
#endif

#if defined(__INTEL_COMPILER) && defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cassert>

namespace {

bool seconds_initialized = false;
LARGE_INTEGER start_count, count_freq;  // counts per sec.

inline void seconds_initialize() {
  if( seconds_initialized ) return;
  std::cout << "\nCalling Win32 version of Teuchos::seconds_initialize()!\n";
  // Figure out how often the performance counter increments
  ::QueryPerformanceFrequency( &count_freq );
  // Set this thread's priority as high as reasonably possible to prevent
  // timeslice interruptions
  ::SetThreadPriority( ::GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL );
  // Get the first count.
  assert( QueryPerformanceCounter( &start_count ) );
  seconds_initialized = true;
}

}       // end namespace

#endif // defined(__INTEL_COMPILER) && defined(_WIN32)

namespace Teuchos {

//=============================================================================
Time::Time(const std::string& name_in, bool start_in)
  : startTime_(0), totalTime_(0), isRunning_(false), enabled_ (true), name_(name_in), numCalls_(0)
#if defined(TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS) && defined(TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS_PREALLOCATE)
    , observation_idx_ (0)
#endif
{
  #if defined(TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS) && defined(TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS_PREALLOCATE)
    observations_.reserve (DESCRIPTIVE_STATISTICS_MAX_NUM_TIMINGS);
    observations_.resize (DESCRIPTIVE_STATISTICS_MAX_NUM_TIMINGS, 0.0);
  #endif

  if(start_in) this->start();
}

void Time::start(bool reset_in)
{
  if (enabled_) {
    isRunning_ = true;
    if (reset_in) totalTime_ = 0;

    if (reset_in) {
    #ifdef TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS
      #ifdef TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS_PREALLOCATE
        // store the timing, and increment the current idx
        observations_.clear ();
        observations_.reserve (DESCRIPTIVE_STATISTICS_MAX_NUM_TIMINGS);
        observations_.resize (DESCRIPTIVE_STATISTICS_MAX_NUM_TIMINGS, 0.0);
        observation_idx_ = 0;
      #else
        observations_.clear ();
      #endif
    #endif
    }

    startTime_ = wallTime();
  }
}

double Time::stop()
{
  if (enabled_) {
    if (isRunning_) {
      const double deltaTime = ( wallTime() - startTime_ );
      totalTime_ += deltaTime;
      isRunning_ = false;
      startTime_ = 0;

      #ifdef TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS
        #ifdef TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS_PREALLOCATE
          // store the timing, and increment the current idx
          observations_[observation_idx_] = deltaTime;
          ++observation_idx_;
        #else
          observations_.push_back (deltaTime);
        #endif
      #endif
    }
  }
  return totalTime_;
}

double Time::totalElapsedTime(bool readCurrentTime) const
{
  if(readCurrentTime)
    return wallTime() - startTime_ + totalTime_;
  return totalTime_;
}

void Time::reset () {
  totalTime_ = 0;
  numCalls_ = 0;

  #ifdef TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS
    observations_.clear ();
    #ifdef TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS_PREALLOCATE
      observations_.reserve (DESCRIPTIVE_STATISTICS_MAX_NUM_TIMINGS);
      observations_.resize (DESCRIPTIVE_STATISTICS_MAX_NUM_TIMINGS, 0.0);
      observation_idx_ = 0;
    #endif
  #endif
}

void Time::computeDescriptiveStats(descriptive_stat_map_type& stat_map) const
{
  /*
   * Use this function as a testing ground
   *
   */
  #ifdef TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS
    #ifdef TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS_PREALLOCATE
      size_t observation_count = observation_idx_;
    #else
      size_t observation_count = observations_.size ();
    #endif
    std::vector<double> mutable_observations;
    mutable_observations.assign(observations_.begin(),
                                observations_.begin() + observation_count);
    assert(mutable_observations.size () == observation_count);
    ::get_descriptive_stats (mutable_observations, observation_count, stat_map);

    stat_map["1st"] = stat_map["1st-excluded-median"];
    stat_map["3rd"] = stat_map["3rd-excluded-median"];
    stat_map["IQR"] = stat_map["IQR-excluded-median"];
    stat_map["numCalls"] = numCalls_;

    std::cout
        << "Min Value" << " = " << stat_map["Min Value"] << std::endl
        << "Max Value" << " = " << stat_map["Max Value"] << std::endl
        << "Mean" << " = " << stat_map["Mean"] << std::endl
        << "Median" << " = " << stat_map["Median"] << std::endl
        << "1st" << " = " << stat_map["1st"] << std::endl
        << "3rd" << " = " << stat_map["3rd"] << std::endl
        << "IQR" << " = " << stat_map["IQR"] << std::endl
        << "Sample StdDev" << " = " << stat_map["Sample StdDev"] << std::endl
        << "Total Time" << " = " << stat_map["Total Time"] << std::endl
        << "Num Observations" << " = " << stat_map["Num Observations"] << std::endl
        << "numCalls" << " = " << stat_map["numCalls"] << std::endl;
//        << "1st-excluded-median" << " = " << stat_map["1st-excluded-median"] << std::endl
//        << "3rd-excluded-median" << " = " << stat_map["3rd-excluded-median"] << std::endl
//        << "1st-included-median" << " = " << stat_map["1st-included-median"] << std::endl
//        << "3rd-included-median" << " = " << stat_map["3rd-included-median"] << std::endl
//        << "IQR-excluded-median" << " = " << stat_map["IQR-excluded-median"] << std::endl
//        << "IQR-included-median" << " = " << stat_map["IQR-included-median"] << std::endl;
  #endif
}

void Time::disable () {
  enabled_ = false;
}

void Time::enable () {
  enabled_ = true;
}

void Time::incrementNumCalls() {
  if (enabled_) {
    ++numCalls_;
  }
}

double Time::wallTime()
{
  /* KL: warning: this code is probably not portable! */
  /* HT: have added some preprocessing to address problem compilers */
        /* RAB: I modifed so that timer will work if MPI support is compiled in but not initialized */

#ifdef HAVE_MPI

        int mpiInitialized;
        MPI_Initialized(&mpiInitialized);

        if( mpiInitialized ) {

                return(MPI_Wtime());

        }
        else {

                clock_t start;

                start = clock();
                return( (double)( start ) / CLOCKS_PER_SEC );

        }

#elif defined(__INTEL_COMPILER) && defined(_WIN32)

  seconds_initialize();
  LARGE_INTEGER count;
  QueryPerformanceCounter( &count );
  // "QuadPart" is a 64 bit integer (__int64).  VC++ supports them!
  const double
    sec = (double)( count.QuadPart - start_count.QuadPart ) / count_freq.QuadPart;
  //std::cout << "ticks = " << ticks << ", sec = " << sec << std::endl;
  return sec;

#elif ICL || defined(_WIN32)

  clock_t start;

  start = clock();
  return( (double)( start ) / CLOCKS_PER_SEC );

#else

#  ifndef MINGW
  struct timeval tp;
  static long start = 0, startu;
  if (!start)
  {
    gettimeofday(&tp, NULL);
    start = tp.tv_sec;
    startu = tp.tv_usec;
    return(0.0);
  }
  gettimeofday(&tp, NULL);
  return( ((double) (tp.tv_sec - start)) + (tp.tv_usec-startu)/1000000.0 );
#  else // MINGW
  return( (double) clock() / CLOCKS_PER_SEC );
#  endif // MINGW

#endif

}

} // namespace Teuchos
