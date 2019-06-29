#ifndef DescriptiveStats_HPP
#define DescriptiveStats_HPP

#include <map>      // for map
#include <vector>   // for vector
#include <chrono>   // for nanoseconds, steady_clock
#include <ostream>  // for ostream
#include <string>   // for string
#include <deque>
#include <memory>

namespace DescriptiveStats {

typedef std::map<std::string, double> descriptive_stat_map_type;
typedef std::map<std::string, std::string> descriptive_stat_map_string_type;

//typedef std::chrono::steady_clock Time;
typedef std::chrono:: high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;
typedef std::chrono::duration<double> double_secs;

enum class stat_types { MEAN, MEDIAN };

void get_descriptive_stats (std::vector<double>& observations,
                            descriptive_stat_map_type& stat_map,
                            const double timer_resolution,
                            const bool compute_ci=true);

void stat_map_to_string_map (const descriptive_stat_map_type& stat_map,
                                   descriptive_stat_map_string_type& string_map);

void print_descriptive_stats (std::ostream& out,
                              descriptive_stat_map_type& stat_map,
                              const double timing_ratio,
                              const std::string& id,
                              const bool csv_format=false,
                              const bool csv_header_only=false);

std::shared_ptr< std::deque<std::string> > get_default_field_order ();

void descriptive_stats_csv_str (std::ostringstream& oss,
                                descriptive_stat_map_string_type& stat_map,
                                const std::shared_ptr<std::deque<std::string> >& only_keys = nullptr,
                                const bool include_header = false);

void profile_timer ( std::vector<double>& timings,
                     const int nrepeat = 10000);

} // end namespace

//#define USE_TIMEOFDAY

#if defined(USE_GETTIME)
#include <sys/time.h>
#endif

#if defined(USE_TIMEOFDAY)
#include <time.h> // try using the c clock
#endif

#if defined(USE_GETTIME)
namespace DescriptiveStats {
void timespec_diff(const timespec& start,
                         timespec& delta,
                   const timespec& end)
{
  if ((end.tv_nsec-start.tv_nsec)<0) {
    delta.tv_sec = end.tv_sec-start.tv_sec-1;
    delta.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    delta.tv_sec = end.tv_sec-start.tv_sec;
    delta.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
}
} // namespace
#endif

#if defined(USE_TIMEOFDAY)
namespace DescriptiveStats {
void timeval_diff(const timeval& left_operand,
                        timeval& res,
                  const timeval& right_operand)
{
  if (left_operand.tv_sec > right_operand.tv_sec)
    timersub(&left_operand, &right_operand, &res);
  else if (left_operand.tv_sec < right_operand.tv_sec)
    timersub(&right_operand, &left_operand, &res);
  else  // left_operand.tv_sec == right_operand.tv_sec
  {
    if (left_operand.tv_usec >= right_operand.tv_usec)
      timersub(&left_operand, &right_operand, &res);
    else
      timersub(&right_operand, &left_operand, &res);
  }
}
} // namespace
#endif

#if defined(USE_GETTIME)
  #define DescStats_TIMEPOINT timespec
  #define DescStats_TIMEPOINT_DELTA timespec
  #define DescStats_TICK(x) ({ clock_gettime(CLOCK_MONOTONIC, &x); })
  #define DescStats_TICKDIFF(s,r,e) ({ DescriptiveStats::timespec_diff(s,r,e); })
  #define DescStats_TICK_TO_DOUBLE(r) ((double) r.tv_sec)*1000000000.0 + r.tv_nsec
  #define DescStats_TICK_RESOLUTION 1.e9
#elif defined(USE_TIMEOFDAY)
  #define DescStats_TIMEPOINT timeval
  #define DescStats_TIMEPOINT_DELTA timeval
  #define DescStats_TICK(x) ({ gettimeofday(&x, NULL); })
  #define DescStats_TICKDIFF(s,r,e) ({ DescriptiveStats::timeval_diff(s,r,e); })
  #define DescStats_TICK_TO_DOUBLE(r) ((double) r.tv_sec)*1000000.0 + r.tv_usec
  #define DescStats_TICK_RESOLUTION 1.e6
#else
  #define DescStats_TIMEPOINT std::chrono::time_point< DescriptiveStats::Time >
  #define DescStats_TIMEPOINT_DELTA DescriptiveStats::double_secs
  #define DescStats_TICK(x) ({ x =  DescriptiveStats::Time::now(); })
  #define DescStats_TICKDIFF(s,r,e) ({ r = e-s; })
  #define DescStats_TICK_TO_DOUBLE(r) std::chrono::duration_cast<DescriptiveStats::ns>(r).count ()
  #define DescStats_TICK_RESOLUTION 1.e9
#endif

#endif // DescriptiveStats_HPP
