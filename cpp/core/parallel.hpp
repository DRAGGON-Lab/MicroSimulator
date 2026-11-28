#pragma once

#include <algorithm>
#include <atomic>
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

// A barrier that the members of a team pass together. The member that
// arrives last releases the others, and every write a member makes before it
// arrives is visible to every member after it passes.
class TeamBarrier {
 public:
  explicit TeamBarrier(std::size_t members) noexcept : members_(members) {
  }

  void arrive_and_wait() noexcept;

 private:
  std::size_t members_;
  alignas(64) std::atomic<std::size_t> arrived_{0};
  alignas(64) std::atomic<std::size_t> generation_{0};
};

// Calls invoke(context, member, members, barrier) once for every member of a
// team of at most `wanted` threads, all running at once, so the members may
// wait for one another at `barrier`. The team is smaller than `wanted` when
// the pool has fewer threads, and has one member for a call made from inside
// parallel work or while another thread is using the pool. A member must not
// throw: the others would wait for it forever.
void run_team(std::size_t wanted,
              void (*invoke)(const void*, std::size_t, std::size_t, TeamBarrier&),
              const void* context);

// Runs body(member, members, barrier) on a team of at most `wanted` threads;
// see run_team.
template <typename Body>
void parallel_team(std::size_t wanted, const Body& body) {
  run_team(
      wanted,
      [](const void* context, std::size_t member, std::size_t members, TeamBarrier& barrier) {
        (*static_cast<const Body*>(context))(member, members, barrier);
      },
      &body);
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
