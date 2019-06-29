#include "descriptive_stats.hpp"
#include <stdint.h>  // for uint64_t
#include <cmath>     // for sqrt, abs
#include <cstdlib>   // for getenv, exit
#include <random>    // for mt19937, random_device
#include <utility>   // for pair
#include <algorithm> // for upper_bound, lower_bound, sort
#include <mpi.h>
#include <iostream>  // for cerr
//#include <Kokkos_Macros.hpp>
#include <sstream>
#ifdef _OPENMP
#pragma message("OpenMP support in descriptive stats")
#include <omp.h>
#endif

namespace {
/*
unsigned long long rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((unsigned long long)hi << 32) | lo;
}

// The state must be seeded so that it is not all zero
uint64_t s[2];

uint64_t xorshift128plus(void) {
	uint64_t x = s[0];
	uint64_t const y = s[1];
	s[0] = y;
	x ^= x << 23; // a
	s[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
	return s[1] + y;
}
*/

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

inline double get_percentile (const double percentile, const std::vector<double>& data) {
  // use interpolated quantiles
  // The difficultly is in handling quantiles that lie between actual data points
  // previously, we computed various 'hinges'.
  // This is similar, but generalizes better.
  // The idea is to use a weighted average between two data points to arrive at the requested
  // quantile. Specifically, we will use N-1 interpolation.

  // get a 1 based index for where this percentile lies
  const double base_calc = percentile * (data.size() - 1);
  const int base_index1 = std::floor(base_calc) + 1;
  // between two indices, where does the quantity of interest lie.
  const double local_percentile = base_calc - std::floor(base_calc);
  // consider this zero
  if ( local_percentile <= 1.e-8 ) {
    return (data[base_index1-1]);
  } else {
    return ((data[base_index1] - data[base_index1-1])*local_percentile + data[base_index1-1]);
  }
}

std::pair<double,double> bootstrap_ci (const std::vector<double>& observations,
                   const DescriptiveStats::stat_types& stat,
                   const int bootstrap_iters = 10000,
                   double conf = 0.99) {

  std::vector<double> bootstrap_data (bootstrap_iters, double(0.0));
  const auto num_observations = observations.size();
  std::vector<double> workspace (num_observations, double(0.0));

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> dis(0, num_observations-1);
/*
  s[0] = rdtsc();
  while (s[0] == 0) s[0] = rdtsc();
  s[1] = rdtsc();
  while (s[1] == 0) s[1] = rdtsc();
*/
  for (int r=0; r < bootstrap_iters; ++r) {
    for (uint32_t i=0; i < num_observations; ++i) {
      workspace[i] = observations[dis(gen)];//  xorshift128plus() % num_observations]; //dis(gen)];
    }
    if (stat == DescriptiveStats::stat_types::MEDIAN) {
      std::sort(workspace.begin (), workspace.end ());
      bootstrap_data[r] = get_median(0, num_observations-1, workspace);
    } else if (stat == DescriptiveStats::stat_types::MEAN) {
      double sum = std::accumulate(workspace.begin(), workspace.end(), double(0.0));
      bootstrap_data[r] = sum / double(num_observations);
    }
  }
  std::sort(bootstrap_data.begin (), bootstrap_data.end ());
  double sig_level = (1.0 - conf)/2.0;
  // w/0.99 and 10,000 iters, this indices are 50, and 99,950 (minus 1)
  int sig_level_idx = std::round(double(sig_level * num_observations));
  size_t lower_ci_idx = sig_level_idx <= 0 ? 0 : sig_level_idx - 1;
  size_t upper_ci_idx = sig_level_idx <= 0 ? bootstrap_iters - 1 : (bootstrap_iters - sig_level_idx) - 1;

  return (std::pair<double,double>(bootstrap_data[lower_ci_idx], bootstrap_data[upper_ci_idx]));

}


} // end anon namespace

namespace DescriptiveStats {

void get_descriptive_stats (std::vector<double>& observations,
                            descriptive_stat_map_type& stat_map,
                            const double timer_resolution,
                            const bool compute_ci) {

  const size_t observation_count = observations.size();
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
    stat_map["timing_resolution"] = timer_resolution;
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

    stat_map["1st-included-median"] =  get_median (0, observation_count/2 - 1, observations);
    stat_map["3rd-included-median"] =  get_median (observation_count/2, observation_count-1, observations);

    // A reasonable question, is whether to include the values used in the midpoint calc of the median
    // above uses those values
    stat_map["1st-excluded-median"] =  get_median (0, observation_count/2 - 2, observations);
    stat_map["3rd-excluded-median"] =  get_median (observation_count/2 + 1, observation_count-1, observations);
  }
  else {
    // there is an odd number of data points, the median is easy, but we need to ensure the interval
    // for Q1 and Q3 starts correctly. E.g., w/101 points.
    // Median location = 50 (counting from zero). 101/2 = 50
    // Q1 lies in [0,50-1], Q3 [50+1, end]

    // both Q1 and Q3 should entail the average between two values, since the interval is even
    stat_map["1st-excluded-median"] =  get_median (0, observation_count/2 - 1, observations);
    // skip over the median
    stat_map["3rd-excluded-median"] =  get_median (observation_count/2 + 1, observation_count-1, observations);

    // including the median is not meaningful
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

  // I choose these as the definitions for the quartiles
  stat_map["1st"] = stat_map["1st-excluded-median"];
  stat_map["3rd"] = stat_map["3rd-excluded-median"];
  stat_map["IQR"] = stat_map["IQR-excluded-median"];

  stat_map["lower_outlier_IQR"] = stat_map["1st"] - 1.5*stat_map["IQR"];
  stat_map["upper_outlier_IQR"] = stat_map["3rd"] + 1.5*stat_map["IQR"];

  // count the number of observations that fall at or below the lower_outlier_IQR
  // upper bound returns an iterator to the first element larger than val
  const auto& lit = std::upper_bound(observations.begin(), observations.end(), stat_map["lower_outlier_IQR"]);
  // if lit==end, then all values were equal to or less than the IRQ bound (e.g., outliers)
  stat_map["lower_outlier_count"] = (lit == observations.cend()) ? observation_count : lit - observations.cbegin();

  const auto& uit = std::lower_bound(observations.begin(), observations.end(), stat_map["upper_outlier_IQR"]);
  // if lit==end, then all values were less than the upper outlier limit
  stat_map["upper_outlier_count"] = (uit == observations.cend()) ? 0 : observation_count - (uit - observations.cbegin());
  //observations.cend() - uit;

  for (int p=1; p < 100; ++p) {
    const double perc = (double) p / 100.0;
    std::stringstream ss;
    ss << "perc_" << p;
    stat_map[ss.str()] = get_percentile ( perc , observations );
  }

  if (compute_ci) {
    std::pair<double,double> median_ci = bootstrap_ci(observations, stat_types::MEDIAN);
    stat_map["median_lower_ci"] = median_ci.first;
    stat_map["median_upper_ci"] = median_ci.second;
    stat_map["median_ci_level"] = 0.99;
  } else {
    stat_map["median_lower_ci"] = std::numeric_limits<double>::quiet_NaN ();
    stat_map["median_upper_ci"] = std::numeric_limits<double>::quiet_NaN ();
    stat_map["median_ci_level"] = std::numeric_limits<double>::quiet_NaN ();
  }

  {
    int comm_size = -1;
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    const char* slurm_nodes_str = std::getenv("SLURM_STEP_NUM_NODES");
    const char* slurm_tasks_per_node_str = std::getenv("SLURM_TASKS_PER_NODE");

    stat_map["rank"] = rank;
    stat_map["num_nodes"] = (slurm_nodes_str ? std::atoi(slurm_nodes_str) : -1);
    stat_map["num_procs"] = comm_size;
    stat_map["procs_per_node"] = (slurm_tasks_per_node_str ? std::atoi(slurm_tasks_per_node_str) : -1);

//    #if defined(__CUDA_ARCH__)
//    #warning Adding Cuda Device ID to csv output options
    int cudaDev = -1;
    cudaGetDevice(&cudaDev);
    stat_map["device_id"] = cudaDev;
//    #endif

    #if defined(_OPENMP)
    stat_map["OMP_NUM_THREADS"] = omp_get_max_threads();
    #endif
  }

  // this is a hack, because I want to have an easier naming/output scheme
  // Ideally, the prior keys will get relabeled to these, and a map that defines
  // a non-csv output string is then created to map to different visualization
  stat_map["median"] = stat_map["Median"];
  stat_map["timing_resolution"] = timer_resolution;
  stat_map["mean"] = stat_map["Mean"];
  stat_map["min_value"] = stat_map["Min Value"];
  stat_map["max_value"] = stat_map["Max Value"];
  stat_map["q1"] = stat_map["1st"];
  stat_map["q3"] = stat_map["3rd"];
  stat_map["lower_outlier_iqr"] = stat_map["lower_outlier_IQR"];
  stat_map["upper_outlier_iqr"] = stat_map["upper_outlier_IQR"];
  stat_map["lower_outliers"] = stat_map["lower_outlier_count"];
  stat_map["upper_outliers"] = stat_map["upper_outlier_count"];
  stat_map["sample_stddev"] = stat_map["Sample StdDev"];
  stat_map["total_time_(s)"] = stat_map["Total Time"];
  stat_map["num_observations"] = stat_map["Num Observations"];


  // given the medians, compute the MAD
  {
    std::vector<double> diffs (observations);
    double median = stat_map["median"];
    std::for_each(diffs.begin(), diffs.end(), [median](double &v){ v = std::abs(v-median); });
    std::sort(diffs.begin(), diffs.end());
    stat_map["mad"] = get_median (0, diffs.size()-1, diffs);
    stat_map["mad_check1"] =  std::abs((stat_map["mad"] *  1.4826) - stat_map["sample_stddev"]) / std::abs(stat_map["sample_stddev"]);
    stat_map["mad_check2"] =  std::abs((stat_map["mad"]) - (stat_map["sample_stddev"]*0.67449)) / std::abs(stat_map["mad"]);
}

  #ifdef TEUCHOS_TIMEMONITOR_USE_DESCRIPTIVE_STATISTICS_CHECK_FP
    std::cout << "Some Timer: Mean          = " << naive_mean
              << "\t\t\t" << ::rel_error(stable_mean, naive_mean
              << std::endl;
    std::cout << "Some Timer: Mean (stable) = " << stable_mean << std::endl;

    std::cout << "Some Timer: Var          = " << naive_variance
              << "\t\t\t" << ::rel_error(stable_variance, naive_variance)
              << std::endl;
    std::cout << "Some Timer: Var (stable) = " << stable_variance << std::endl;

    std::cout << "Some Timer: StdDev          = " << naive_stddev
              << "\t\t\t" << ::rel_error(stable_stddev, naive_stddev)
              << std::endl;
    std::cout << "Some Timer: StdDev (stable) = " << stable_stddev << std::endl;

    std::cout << "Some Timer: Total Time          = " << naive_total_time
              << "\t\t\t" << ::rel_error(total_time_sorted, naive_total_time)
              << std::endl;
    std::cout << "Some Timer: Total Time (stable) = " << total_time_sorted << std::endl;
  #endif
}

std::shared_ptr< std::deque<std::string> > get_default_field_order() {

  std::deque<std::string> fields = {
    "host",
    #ifdef _OPENMP
    "OMP_NUM_THREADS",
    #endif
    "num_nodes", 
    "num_procs", 
    "procs_per_node",
    "rank",
    #ifdef __CUDA_ARCH__
    "device_id",
    #endif
    "median",
    "mad",
    "timing_resolution",
    "mean",
    "min_value",
    "max_value",
    "median_lower_ci",
    "median_upper_ci",
    "q1",
    "q3",
    "lower_outlier_iqr",
    "upper_outlier_iqr",
    "lower_outliers",
    "upper_outliers",
    "sample_stddev",
    "total_time_(s)",
    "num_observations",
    "perc_1",
    "perc_2",
    "perc_3",
    "perc_4",
    "perc_5",
    "perc_6",
    "perc_7",
    "perc_8",
    "perc_9",
    "perc_10",
    "perc_25",
    "perc_50",
    "perc_75",
    "perc_90",
    "perc_91",
    "perc_92",
    "perc_93",
    "perc_94",
    "perc_95",
    "perc_96",
    "perc_97",
    "perc_98",
    "perc_99"
    //,"mad_check1", "mad_check2"
    };

  return std::shared_ptr< std::deque<std::string> > (new std::deque<std::string> (fields));
}

void print_descriptive_stats (std::ostream& out,
                              descriptive_stat_map_type& stat_map,
                              const double timing_ratio,
                              const std::string& id,
                              const bool csv_format,
                              const bool csv_header_only) {

   using std::endl;
   out.precision(6);

   if (!csv_format) {
     out
        << "ID: " << id << endl
        #ifdef _OPENMP
        << "OMP_NUM_THREADS = " << stat_map["OMP_NUM_THREADS"] << endl
        #endif
        << "num_nodes = " << stat_map["num_nodes"] << endl
        << "num_procs = " << stat_map["num_procs"] << endl
        << "procs_per_node = " << stat_map["procs_per_node"] << endl
        << "Timings gathered using " << stat_map["timing_resolution"] << " (s)" << endl
        << "Min Value" << " = " << stat_map["min_value"] << endl
        << "Max Value" << " = " << stat_map["max_value"] << endl
        << "Mean" << " = " << stat_map["mean"] << endl
        << "Median" << " = " << stat_map["median"] << endl
        << "Median 99% CI" << " = "
          << "[" << stat_map["median_lower_ci"]
          << "," << stat_map["median_upper_ci"]
          << "]" << endl
        << "1st" << " = " << stat_map["q1"] << endl
        << "3rd" << " = " << stat_map["q3"] << endl
        << "1.5*IQR " << " = "
          << "[" << stat_map["lower_outlier_iqr"]
          << "," << stat_map["upper_outlier_iqr"]
          << "]" << endl
        << "Number of lower outlier = " << stat_map["lower_outlier_count"] << endl
        << "Number of upper outlier = " << stat_map["upper_outlier_count"] << endl
        << "Total Outliers = " << stat_map["lower_outlier_count"] + stat_map["upper_outlier_count"] << endl
        << "Sample StdDev" << " = " << stat_map["Sample StdDev"] << endl
        << "Total Time (s) " << " = " << stat_map["Total Time"] * timing_ratio << endl
        << "Num Observations" << " = " << stat_map["Num Observations"] << endl;
//        << "1st-excluded-median" << " = " << stat_map["1st-excluded-median"] << endl
//        << "3rd-excluded-median" << " = " << stat_map["3rd-excluded-median"] << endl
//        << "1st-included-median" << " = " << stat_map["1st-included-median"] << endl
//        << "3rd-included-median" << " = " << stat_map["3rd-included-median"] << endl
//        << "IQR-excluded-median" << " = " << stat_map["IQR-excluded-median"] << endl
//        << "IQR-included-median" << " = " << stat_map["IQR-included-median"] << endl;
  } else {
    if (csv_header_only) {
      std::ostringstream oss;
      oss  <<
        "id,"
        #ifdef _OPENMP
        "OMP_NUM_THREADS,"
        #endif
        "num_nodes," 
        "num_procs," 
        "procs_per_node,"
        "device_id,"
        "median,"
        "timing_resolution,"
        "mean,"
        "min_value,"
        "max_value,"
        "median_lower_ci,"
        "median_upper_ci,"
        "q1,"
        "q3,"
        "lower_outlier_iqr,"
        "upper_outlier_iqr,"
        "lower_outliers,"
        "upper_outliers,"
        "sample_stddev,"
        "total_time_(s),"
        "num_observations";

      for (int p=1; p <= 10; ++p) {
        std::stringstream ss;
        oss << ",perc_" << p;
      }
      oss << ",perc_25,perc_50,perc_75";

      for (int p=90; p < 100; ++p) {
        std::stringstream ss;
        oss << ",perc_" << p;
      }
      out << oss.str () << std::endl;

    } else {
      std::ostringstream oss;
      oss
        << id
        #ifdef _OPENMP
        << "," << omp_get_max_threads()
        #endif
        << "," << stat_map["num_nodes"] 
        << "," << stat_map["num_procs"]
        << ",\"" << stat_map["procs_per_node"]
        << "," << stat_map["device_id"]
        << "," << stat_map["median"]
        << "," << timing_ratio
        << "," << stat_map["mean"]
        << "," << stat_map["min_value"]
        << "," << stat_map["max_value"]
        << "," << stat_map["median_lower_ci"]
        << "," << stat_map["median_upper_ci"]
        << "," << stat_map["q1"]
        << "," << stat_map["q3"]
        << "," << stat_map["lower_outlier_iqr"]
        << "," << stat_map["upper_outlier_iqr"]
        << "," << stat_map["lower_outlier_count"]
        << "," << stat_map["upper_outlier_count"]
        << "," << stat_map["Sample StdDev"]
        << "," << stat_map["Total Time"] * timing_ratio
        << "," << stat_map["Num Observations"];

      for (int p=1; p <= 10; ++p) {
        std::stringstream ss;
        ss << "perc_" << p;
        oss << "," << stat_map[ss.str()];
      }
      {
        int p = 25;
        std::stringstream ss;
        ss << "perc_" << p;
        oss << "," << stat_map[ss.str()];
      }
      {
        int p = 50;
        std::stringstream ss;
        ss << "perc_" << p;
        oss << "," << stat_map[ss.str()];
      }
      {
        int p = 75;
        std::stringstream ss;
        ss << "perc_" << p;
        oss << "," << stat_map[ss.str()];
      }
      for (int p=90; p < 100; ++p) {
        std::stringstream ss;
        ss << "perc_" << p;
        oss << "," << stat_map[ss.str()];
      }
      out << oss.str () << std::endl;
    }
  }
}

void descriptive_stats_csv_str (std::ostringstream& out,
                                descriptive_stat_map_string_type& stat_map,
                                const std::shared_ptr< std::deque<std::string> >& only_keys,
                                const bool include_header)
{

  auto only_keys_ = (!only_keys) ? get_default_field_order() : only_keys;

  if (only_keys_->empty())  return;

  if (include_header) {
    std::string h = std::accumulate(only_keys_->begin()+1, // start
                                    only_keys_->end(),     // stop
                                    only_keys_->front(),   // initial value
                       [&](const std::string& a, std::string& b){
                             return a + ',' + b;
                       });
    out << h << "\n";
  }

  // if the keys request hostname, then we need to add it
  if (std::find(only_keys_->begin(), only_keys_->end(), "host") != only_keys_->end()) {
    char hname[MPI_MAX_PROCESSOR_NAME];
    int hlen = 0;
    MPI_Get_processor_name( hname, &hlen);
    if (hlen > 0)
      stat_map["host"] = hname;
  }

  std::string s = std::accumulate(only_keys_->begin()+1, // start
                                  only_keys_->end(),     // stop
                                  stat_map[only_keys_->front()],   // initial value
                     [&](const std::string& a, std::string& b){
                           return a + ',' + stat_map[b];
                     });
  out << s << "\n";

}

void stat_map_to_string_map (const descriptive_stat_map_type& stat_map,
                                   descriptive_stat_map_string_type& string_map)
{
  std::ostringstream ss;
  for (const auto& kv : stat_map) {
    ss.str("");
    ss << kv.second;
    string_map[kv.first] = ss.str();
  }
}

void profile_timer (
                     std::vector<double>& timing_resolutions,
                     const int nrepeat) {
  for (int i=0; i < nrepeat; ++i) {

    auto t0 = Time::now ();
    
    auto t1 = Time::now ();

    double_secs ds = t1 - t0;
    timing_resolutions.push_back(std::chrono::duration_cast<ns>(ds).count ());
  }

}

} // end namespace

