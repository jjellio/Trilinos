#include "NvshmemP6Stats.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  using namespace NvshmemP6Stats;
  const std::vector<double> rankMajor = {10, 50, 30, 20, 12, 40};
  const auto maxes = maxRankByIteration(rankMajor, 2, 3);
  assert((maxes == std::vector<double>{20, 50, 40}));
  const Summary s = summarize(maxes);
  assert(s.min == 20.0 && s.max == 50.0 && s.median == 40.0);
  assert(std::abs(s.p10 - 24.0) < 1e-12);
  assert(std::abs(s.p90 - 48.0) < 1e-12);
  assert(percentile(std::vector<double>{1, 3, 5, 7}, 0.5) == 4.0);
  bool rejected = false;
  try { (void)maxRankByIteration({1.0}, 2, 3); }
  catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);
  rejected = false;
  try { (void)summarize({-1.0}); }
  catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);
  std::cout << "P6 statistics: PASS (interpolated percentiles, per-repetition rank maxima, invalid inputs)\n";
}
