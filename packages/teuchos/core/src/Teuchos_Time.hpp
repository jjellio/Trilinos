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

#ifndef _TEUCHOS_TIME_HPP_
#define _TEUCHOS_TIME_HPP_

/*! \file Teuchos_Time.hpp
    \brief Basic wall-clock timer class
*/

#include "Teuchos_ConfigDefs.hpp"

#include <ctime>
#ifdef HAVE_MPI
#include "mpi.h"
#else
#if ICL || defined(_WIN32)
#include <time.h>
#else
#include <sys/time.h>
#ifndef MINGW
#include <sys/resource.h>
#endif
#endif
#endif


#define TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS 1
#undef TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS_DEBUG
//#undef TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS

#define TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS_PREALLOCATE 1

#ifdef TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS
//#include <Teuchos_Array.hpp>
#include <vector>
#include <algorithm>
#endif

namespace Teuchos {


/// \class Time
/// \brief Wall-clock timer.
///
/// To time a section of code, place it in between calls to start()
/// and stop().  It is better to access this class through the
/// TimeMonitor class (which see) for exception safety and correct
/// behavior in reentrant code.
class TEUCHOSCORE_LIB_DLL_EXPORT Time {
public:
  /// \brief Constructor
  ///
  /// \param name [in] Name of the timer.
  /// \param start [in] If true, start the timer upon creating it.  By
  ///   default, the timer only starts running when you call start().
  Time (const std::string& name, bool start = false);

  /// \brief Current wall-clock time in seconds.
  ///
  /// This is only useful for measuring time intervals.  The absolute
  /// value returned is measured relative to some arbitrary time in
  /// the past.
  static double wallTime ();

  /// \brief Start the timer, if the timer is enabled (see disable()).
  ///
  /// \param reset [in] If true, reset the timer's total elapsed time
  ///   to zero before starting the timer.  By default, the timer
  ///   accumulates the total elapsed time for all start() ... stop()
  ///   sequences.
  void start (bool reset = false);

  //! Stop the timer, if the timer is enabled (see disable()).
  double stop ();

  //! "Disable" this timer, so that it ignores calls to start() and stop().
  void disable ();

  //! "Enable" this timer, so that it (again) respects calls to start() and stop().
  void enable ();

  //! Whether the timer is enabled (see disable()).
  bool isEnabled () const {
    return enabled_;
  }

  /// \brief The total time in seconds accumulated by this timer.
  ///
  /// \param readCurrentTime [in] If true, return the current elapsed
  ///   time since the first call to start() when the timer was
  ///   enabled, whether or not the timer is running or enabled.  If
  ///   false, return the total elapsed time as of the last call to
  ///   stop() when the timer was enabled.
  ///
  /// \note If start() has never been called when the timer was
  ///   enabled, and if readCurrentTime is true, this method will
  ///   return wallTime(), regardless of the actual start time.
  double totalElapsedTime (bool readCurrentTime = false) const;

  //! Reset the cummulative time and call count.
  void reset ();

  /// \brief Whether the timer is currently running.
  ///
  /// "Currently running" means either that start() has been called
  /// without an intervening stop() call, or that the timer was
  /// created already running and stop() has not since been called.
  bool isRunning() const {
    return isRunning_;
  }

  //! The name of this timer.
  const std::string& name() const {
    return name_;
  }

  /// \brief Increment the number of times this timer has been called,
  ///   if the timer is enabled (see disable()).
  void incrementNumCalls();

  //! The number of times this timer has been called while enabled.
  int numCalls() const {return numCalls_;}

private:
  double startTime_;
  double totalTime_;
  bool isRunning_;
  bool enabled_;
  std::string name_;
  int numCalls_;

  /*
   * 18 May 2017 (jjellio)
   *   Add descriptive statistics.
   *   To do this, we need to store the actual timings. We assume that is the timer is nested
   *   that the logic for incrementing the timer correctly hanldes whether increment is called or not
   *   For every increment, we store the delta.
   */
  #ifdef TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS
    std::vector<double> observations_;
public:
    typedef std::map<std::string, double> descriptive_stat_map_type;

    void computeDescriptiveStats (std::map<std::string, double>& stat_map) const;

    static constexpr const char* DS_TOTAL_TIME_KEY = "Total Time";
    static constexpr const char* DS_NUM_OBSERVATIONS_KEY = "Num Observations";
    static constexpr const char* DS_numCalls_KEY = "numCalls";

    static constexpr const char* DS_MEDIAN_KEY = "Median";
    static constexpr const char* DS_MEAN_KEY = "Mean";
    static constexpr const char* DS_STDDEV_KEY = "Sample StdDev";
    static constexpr const char* DS_MIN_KEY = "Min Value";
    static constexpr const char* DS_MAX_KEY = "Max Value";
    static constexpr const char* DS_1st_QUARTILE_KEY = "1st";
    static constexpr const char* DS_3rd_QUARTILE_KEY = "3rd";

    enum DS_STAT_IDS_enum {
      total_time = 0,
      median,
      mean,
      q1,
      q3,
      stddev,
      min_value,
      max_value,
      num_calls,
      num_observations,
      enumSize
    };

    //static DS_STAT_IDS_enum DS_STAT_IDS;

    typedef struct teuchos_descriptive_stats_mpi_struct {
      double total_time;
      double median;
      double mean;
      double q1;
      double q3;
      double stddev;
      double min_value;
      double max_value;
      double numCalls;
      double num_observations;
    } teuchos_descriptive_stats_mpi_struct_t;

//    static const MPI_Datatype * teuchos_descriptive_stats_struct_mpi_types  = {
//      MPI_DOUBLE,
//      MPI_DOUBLE,
//      MPI_DOUBLE,
//      MPI_DOUBLE,
//      MPI_DOUBLE,
//      MPI_DOUBLE,
//      MPI_DOUBLE,
//      MPI_DOUBLE,
//      MPI_DOUBLE,
//      MPI_DOUBLE };

    static
    teuchos_descriptive_stats_mpi_struct_t getDescriptiveStatsStruct (const descriptive_stat_map_type& stat_map) {
      teuchos_descriptive_stats_mpi_struct_t stats;

      descriptive_stat_map_type::const_iterator loc;

      loc = stat_map.find ("Total Time");
      stats.total_time = (loc == stat_map.end()) ? std::numeric_limits<double>::quiet_NaN() : loc->second;

      loc = stat_map.find ("Median");
      stats.median = (loc == stat_map.end()) ? std::numeric_limits<double>::quiet_NaN() : loc->second;

      loc = stat_map.find ("Mean");
      stats.mean = (loc == stat_map.end()) ? std::numeric_limits<double>::quiet_NaN() : loc->second;

      loc = stat_map.find ("1st");
      stats.q1 = (loc == stat_map.end()) ? std::numeric_limits<double>::quiet_NaN() : loc->second;

      loc = stat_map.find ("3rd");
      stats.q3 = (loc == stat_map.end()) ? std::numeric_limits<double>::quiet_NaN() : loc->second;

      loc = stat_map.find ("Sample StdDev");
      stats.stddev = (loc == stat_map.end()) ? std::numeric_limits<double>::quiet_NaN() : loc->second;

      loc = stat_map.find ("Min Value");
      stats.min_value = (loc == stat_map.end()) ? std::numeric_limits<double>::quiet_NaN() : loc->second;

      loc = stat_map.find ("Max Value");
      stats.max_value = (loc == stat_map.end()) ? std::numeric_limits<double>::quiet_NaN() : loc->second;

      loc = stat_map.find ("numCalls");
      stats.numCalls = (loc == stat_map.end()) ? std::numeric_limits<double>::quiet_NaN() : loc->second;

      loc = stat_map.find ("Num Observations");
      stats.num_observations = (loc == stat_map.end()) ? std::numeric_limits<double>::quiet_NaN() : loc->second;

      return (stats);
    }

    static
    descriptive_stat_map_type getDescriptiveStatsMap (teuchos_descriptive_stats_mpi_struct_t& stat_struct) {
      descriptive_stat_map_type stat_map;

      stat_map["Total Time"] = stat_struct.total_time;
      stat_map["Median"] = stat_struct.median;
      stat_map["Mean"] = stat_struct.mean;
      stat_map["1st"] = stat_struct.q1;
      stat_map["3rd"] = stat_struct.q3;
      stat_map["stddev"] = stat_struct.stddev;
      stat_map["Min Value"] = stat_struct.min_value;
      stat_map["Max Value"] = stat_struct.max_value;
      stat_map["numCalls"] = stat_struct.numCalls;
      stat_map["Num Observations"] = stat_struct.num_observations;
      return (stat_map);
    }

    static void printDescriptiveStatsMap (const descriptive_stat_map_type& stat_map, std::stringstream& ss) {
      printDescriptiveStatsStruct(getDescriptiveStatsStruct(stat_map), ss);
    }

    static void printDescriptiveStatsStruct (const teuchos_descriptive_stats_mpi_struct_t& stat_struct,
                                             std::stringstream& ss) {
      using std::endl;

      ss  << "Min Value"        << " = " << stat_struct.min_value        << endl
          << "Max Value"        << " = " << stat_struct.max_value        << endl
          << "Mean"             << " = " << stat_struct.mean             << endl
          << "Median"           << " = " << stat_struct.median           << endl
          << "1st"              << " = " << stat_struct.q1               << endl
          << "3rd"              << " = " << stat_struct.q3               << endl
          << "IQR"              << " = " << (stat_struct.q3 - stat_struct.q1) << endl
          << "Sample StdDev"    << " = " << stat_struct.stddev           << endl
          << "Total Time"       << " = " << stat_struct.total_time       << endl
          << "Num Observations" << " = " << stat_struct.num_observations << endl
          << "numCalls"         << " = " << stat_struct.numCalls         << endl;
    }


    static const int DS_NUM_STATS = Teuchos::Time::enumSize;

private:
    #ifdef TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS_PREALLOCATE
      static constexpr size_t DESCRIPTIVE_STATISTICS_MAX_NUM_TIMINGS = 1024*1024*100;
      int observation_idx_;
    #endif
  #endif
};


} // namespace Teuchos


#endif // TEUCHOS_TIME_HPP_
