#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core/parallel.hpp"
#include "microsimulator/threads.hpp"

namespace {

using microsimulator::cpu_thread_count;
using microsimulator::set_cpu_thread_count;
using microsimulator::detail::concatenate;
using microsimulator::detail::parallel_for;
using microsimulator::detail::parallel_reduce;

constexpr std::size_t thread_counts[] = {1, 2, 3, 8};

void test_every_index_runs_exactly_once() {
  for (const auto threads : thread_counts) {
    set_cpu_thread_count(threads);
    for (const std::size_t count : {0U, 1U, 7U, 64U, 1000U, 100003U}) {
      for (const std::size_t grain : {1U, 3U, 64U, 4096U}) {
        std::vector<std::atomic<std::uint32_t>> visits(count);
        parallel_for(count, grain, [&](std::size_t begin, std::size_t end) {
          assert(begin < end && end <= count && end - begin <= grain && begin % grain == 0);
          for (std::size_t index = begin; index < end; ++index) {
            visits[index].fetch_add(1, std::memory_order_relaxed);
          }
        });
        for (const auto& visit : visits) {
          assert(visit.load() == 1);
        }
      }
    }
  }
}

// A sum whose value depends on how its terms are grouped.
float grouped_sum(std::size_t count, std::size_t grain) {
  return parallel_reduce(
      count, grain, 0.0F,
      [](std::size_t begin, std::size_t end) {
        auto sum = 0.0F;
        for (std::size_t index = begin; index < end; ++index) {
          sum += 1.0F / static_cast<float>(index + 1);
        }
        return sum;
      },
      [](float left, float right) { return left + right; });
}

void test_reduction_does_not_depend_on_the_thread_count() {
  constexpr std::size_t count = 200000;
  set_cpu_thread_count(1);
  const auto reference = grouped_sum(count, 1024);
  assert(reference != grouped_sum(count, count));
  for (const auto threads : thread_counts) {
    set_cpu_thread_count(threads);
    for (int repeat = 0; repeat < 50; ++repeat) {
      assert(grouped_sum(count, 1024) == reference);
    }
  }
  assert(grouped_sum(0, 16) == 0.0F);
}

void test_boolean_reduction() {
  set_cpu_thread_count(8);
  const auto all_below = [](std::size_t count, std::size_t limit) {
    return parallel_reduce(
        count, 1, true, [&](std::size_t begin, std::size_t) { return begin < limit; },
        [](bool left, bool right) { return left && right; });
  };
  for (int repeat = 0; repeat < 200; ++repeat) {
    assert(all_below(4096, 4096));
    assert(!all_below(4096, 4095));
  }
}

void test_the_lowest_failing_chunk_is_reported() {
  for (const auto threads : thread_counts) {
    set_cpu_thread_count(threads);
    for (int repeat = 0; repeat < 50; ++repeat) {
      try {
        parallel_for(1000, 1, [](std::size_t begin, std::size_t) {
          if (begin == 17 || begin == 400 || begin == 999) {
            throw std::runtime_error(std::to_string(begin));
          }
        });
        assert(false);
      } catch (const std::runtime_error& error) {
        assert(std::string(error.what()) == "17");
      }
    }
    // The pool works again after a failure.
    assert(grouped_sum(1000, 10) == grouped_sum(1000, 10));
  }
}

void test_nested_work_runs_on_the_calling_thread() {
  set_cpu_thread_count(4);
  std::vector<std::atomic<std::uint32_t>> visits(64 * 64);
  parallel_for(64, 1, [&](std::size_t outer, std::size_t) {
    const auto caller = std::this_thread::get_id();
    parallel_for(64, 1, [&](std::size_t inner, std::size_t) {
      assert(std::this_thread::get_id() == caller);
      visits[outer * 64 + inner].fetch_add(1, std::memory_order_relaxed);
    });
  });
  for (const auto& visit : visits) {
    assert(visit.load() == 1);
  }
}

void test_concurrent_callers_all_complete() {
  set_cpu_thread_count(4);
  constexpr std::size_t callers = 6;
  constexpr std::size_t count = 20000;
  std::vector<std::uint64_t> sums(callers, 0);
  {
    std::vector<std::thread> threads;
    for (std::size_t caller = 0; caller < callers; ++caller) {
      threads.emplace_back([&, caller] {
        for (int repeat = 0; repeat < 200; ++repeat) {
          sums[caller] += parallel_reduce(
              count, 128, std::uint64_t{0},
              [](std::size_t begin, std::size_t end) {
                std::uint64_t sum = 0;
                for (std::size_t index = begin; index < end; ++index) {
                  sum += index;
                }
                return sum;
              },
              [](std::uint64_t left, std::uint64_t right) { return left + right; });
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }
  for (const auto sum : sums) {
    assert(sum == 200 * (count * (count - 1) / 2));
  }
}

void test_many_short_regions() {
  set_cpu_thread_count(8);
  std::uint64_t total = 0;
  for (std::size_t region = 0; region < 20000; ++region) {
    std::atomic<std::uint64_t> sum{0};
    parallel_for(16, 1, [&](std::size_t begin, std::size_t) {
      sum.fetch_add(begin + 1, std::memory_order_relaxed);
    });
    total += sum.load();
  }
  assert(total == 20000ULL * 136ULL);
}

void test_regions_resume_after_the_workers_sleep() {
  set_cpu_thread_count(4);
  for (int repeat = 0; repeat < 5; ++repeat) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    std::atomic<std::uint32_t> visits{0};
    parallel_for(256, 1, [&](std::size_t, std::size_t) {
      visits.fetch_add(1, std::memory_order_relaxed);
    });
    assert(visits.load() == 256);
  }
}

void test_thread_count_configuration() {
  set_cpu_thread_count(3);
  assert(cpu_thread_count() == 3);
  set_cpu_thread_count(0);
  assert(cpu_thread_count() >= 1);
  auto rejected = false;
  parallel_for(4, 1, [&](std::size_t begin, std::size_t) {
    if (begin != 0) {
      return;
    }
    try {
      set_cpu_thread_count(2);
    } catch (const std::logic_error&) {
      rejected = true;
    }
  });
  // With one thread the chunks run outside a region and the change succeeds.
  assert(rejected || cpu_thread_count() == 2);
  auto too_large = false;
  try {
    set_cpu_thread_count(1U << 20U);
  } catch (const std::invalid_argument&) {
    too_large = true;
  }
  assert(too_large);
}

void test_concatenate_keeps_piece_order() {
  const std::vector<std::vector<int>> pieces{{1, 2}, {}, {3}, {4, 5, 6}};
  assert(concatenate(pieces) == (std::vector<int>{1, 2, 3, 4, 5, 6}));
  assert(concatenate(std::vector<std::vector<int>>{}).empty());
}

}  // namespace

int main() {
  test_every_index_runs_exactly_once();
  test_reduction_does_not_depend_on_the_thread_count();
  test_boolean_reduction();
  test_the_lowest_failing_chunk_is_reported();
  test_nested_work_runs_on_the_calling_thread();
  test_concurrent_callers_all_complete();
  test_many_short_regions();
  test_regions_resume_after_the_workers_sleep();
  test_thread_count_configuration();
  test_concatenate_keeps_piece_order();
  return 0;
}
