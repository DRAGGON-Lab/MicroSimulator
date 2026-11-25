#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace microsimulator::detail {

// Calls invoke(context, chunk) once for every chunk in [0, chunks), on the
// calling thread and the pool's workers, and returns when all have finished.
//
// When chunks throw, the exception of the lowest one is rethrown, so the
// reported failure does not depend on timing; the chunks after it may or may
// not have run. A call made from inside a chunk, or while another thread is
// using the pool, runs its chunks on the calling thread in order.
void run_chunks(std::size_t chunks, void (*invoke)(const void*, std::size_t), const void* context);

template <typename Task>
void run_chunks(std::size_t chunks, const Task& task) {
  run_chunks(
      chunks,
      [](const void* context, std::size_t chunk) { (*static_cast<const Task*>(context))(chunk); },
      &task);
}

// Calls body(begin, end) over [0, count) in pieces of `grain` indices. The
// pieces depend only on `count` and `grain`, never on the thread count, so a
// body that writes only what its own indices own gives the same result on any
// number of threads.
template <typename Body>
void parallel_for(std::size_t count, std::size_t grain, Body&& body) {
  const auto chunks = (count + grain - 1) / grain;
  const auto task = [&](std::size_t chunk) {
    const auto begin = chunk * grain;
    body(begin, std::min(count, begin + grain));
  };
  run_chunks(chunks, task);
}

// Folds body(begin, end) over [0, count) in pieces of `grain` indices and
// combines the pieces in index order, starting from `identity`. The grouping
// depends only on `count` and `grain`, so a floating-point sum is the same on
// any number of threads.
template <typename Value, typename Body, typename Combine>
[[nodiscard]] Value parallel_reduce(std::size_t count, std::size_t grain, Value identity,
                                    Body&& body, Combine&& combine) {
  // Each piece is its own object, so threads never share one: a bare
  // std::vector<bool> would pack several pieces into a word.
  struct Piece {
    Value value;
  };
  const auto chunks = (count + grain - 1) / grain;
  std::vector<Piece> pieces(chunks, Piece{identity});
  const auto task = [&](std::size_t chunk) {
    const auto begin = chunk * grain;
    pieces[chunk].value = body(begin, std::min(count, begin + grain));
  };
  run_chunks(chunks, task);
  auto result = identity;
  for (const auto& piece : pieces) {
    result = combine(result, piece.value);
  }
  return result;
}

// Joins the outputs of pieces computed in parallel into one vector, in piece
// order.
template <typename Value>
[[nodiscard]] std::vector<Value> concatenate(const std::vector<std::vector<Value>>& pieces) {
  auto total = std::size_t{0};
  for (const auto& piece : pieces) {
    total += piece.size();
  }
  std::vector<Value> result;
  result.reserve(total);
  for (const auto& piece : pieces) {
    result.insert(result.end(), piece.begin(), piece.end());
  }
  return result;
}

}  // namespace microsimulator::detail
