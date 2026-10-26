#include "NvshmemP71Stats.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace NvshmemP71Stats;
int main() {
  assert(planSampling(0, 64).samples == 0);
  assert(planSampling(1, 64).samples == 1);
  assert(planSampling(64, 64).stride == 1);
  assert(planSampling(65, 64).stride == 2);
  assert(planSampling(144249, 128).samples <= 128);
  for (std::size_t size : {std::size_t(0), std::size_t(1), std::size_t(3),
                           std::size_t(17), std::size_t(10000)}) {
    for (std::size_t cap : {std::size_t(1), std::size_t(4), std::size_t(64)}) {
      const auto s = planSampling(size, cap);
      assert(s.samples <= cap);
      std::size_t previous = 0;
      for (std::size_t i = 0; i < s.samples; ++i) {
        const std::size_t j = sampledId(s, i);
        assert(j < size && j % s.stride == 0);
        if (i) assert(j > previous);
        previous = j;
      }
    }
  }
  assert(std::abs(cyclesToMicroseconds(1500, 1500000) - 1.0) < 1e-10);
  assert(std::abs(quantile({1,3,7,9}, 0.5) - 5.0) < 1e-10);
  assert(quantile({}, 0.5) == 0.0);
  const auto paired = pairedFullIteration({1, 10}, {10, 1});
  assert(paired.size() == 2 && paired[0] == 11 && paired[1] == 11);
  bool caught = false;
  try { (void)planSampling(1, 0); } catch (const std::invalid_argument&) { caught = true; }
  assert(caught);
  std::cout << "[NVSHMEM P7.1 STATS] RESULT=PASS\n";
}
