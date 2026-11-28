#include "core/parallel.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "microsimulator/threads.hpp"

#if defined(__APPLE__)
#include <pthread.h>
#include <sys/sysctl.h>
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace microsimulator {
namespace detail {
namespace {

constexpr std::size_t max_thread_count = 1024;
constexpr std::uint64_t chunk_mask = 0xffffffffU;
// Parallel regions arrive microseconds apart inside an iterative solver, so a
// worker spins this long for the next one before it sleeps.
constexpr auto spin_time = std::chrono::microseconds(100);

void relax() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_pause();
#elif defined(_MSC_VER) && defined(_M_ARM64)
  __yield();
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("isb");
#endif
}

[[nodiscard]] std::size_t default_thread_count() {
  // NOLINTNEXTLINE(concurrency-mt-unsafe): read once per pool configuration.
  if (const char* text = std::getenv("MICROSIMULATOR_NUM_THREADS")) {
    char* end = nullptr;
    const auto value = std::strtoull(text, &end, 10);
    if (end != text && *end == '\0' && value >= 1) {
      return static_cast<std::size_t>(std::min<unsigned long long>(value, max_thread_count));
    }
  }
#if defined(__APPLE__)
  // Efficiency cores finish their share late and hold a region open.
  int performance_cores = 0;
  std::size_t size = sizeof(performance_cores);
  if (sysctlbyname("hw.perflevel0.physicalcpu", &performance_cores, &size, nullptr, 0) == 0 &&
      performance_cores > 0) {
    return static_cast<std::size_t>(performance_cores);
  }
#endif
  return std::clamp<std::size_t>(std::thread::hardware_concurrency(), 1, max_thread_count);
}

thread_local bool inside_region = false;

// A run of chunks that one thread works through from the front. The word
// holds the next chunk above the end of the run, so a claim and the run's
// replacement by the next region cannot interleave.
struct alignas(64) ChunkRun {
  std::atomic<std::uint64_t> word{0};

  void assign(std::size_t begin, std::size_t end) noexcept {
    word.store((static_cast<std::uint64_t>(begin) << 32U) | end);
  }

  // Takes the next chunk, or returns false when none remain.
  [[nodiscard]] bool claim(std::size_t& chunk) noexcept {
    auto current = word.load(std::memory_order_acquire);
    while ((current >> 32U) < (current & chunk_mask)) {
      if (word.compare_exchange_weak(current, current + (std::uint64_t{1} << 32U),
                                     std::memory_order_acq_rel)) {
        chunk = static_cast<std::size_t>(current >> 32U);
        return true;
      }
    }
    return false;
  }
};

class ThreadPool {
 public:
  // The pool lives for the whole process: joining threads while a process
  // exits is not reliable on every platform.
  static ThreadPool& instance() {
    static auto* const pool = new ThreadPool;
    return *pool;
  }

  [[nodiscard]] std::size_t thread_count() const noexcept {
    return thread_count_.load();
  }

  void set_thread_count(std::size_t count) {
    if (inside_region) {
      throw std::logic_error("thread count cannot change inside parallel work");
    }
    if (count > max_thread_count) {
      throw std::invalid_argument("thread count is too large");
    }
    const std::lock_guard lock(busy_);
    stop_workers();
    thread_count_.store(count == 0 ? default_thread_count() : count);
  }

  void team(std::size_t wanted, void (*invoke)(const void*, std::size_t, std::size_t, TeamBarrier&),
            const void* context) {
    if (wanted > 1 && !inside_region) {
      std::unique_lock busy(busy_, std::try_to_lock);
      if (busy.owns_lock()) {
        start_workers();
        // The calling thread and each worker run one member.
        const auto members = std::min(wanted, workers_.size() + 1);
        if (members > 1) {
          run_team_region(members, invoke, context);
          return;
        }
      }
    }
    TeamBarrier alone(1);
    invoke(context, 0, 1, alone);
  }

  void run(std::size_t chunks, void (*invoke)(const void*, std::size_t), const void* context) {
    if (chunks == 0) {
      return;
    }
    if (chunks > 1 && chunks <= chunk_mask && !inside_region) {
      std::unique_lock busy(busy_, std::try_to_lock);
      if (busy.owns_lock()) {
        start_workers();
        if (!workers_.empty()) {
          run_region(chunks, invoke, context);
          return;
        }
      }
    }
    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
      invoke(context, chunk);
    }
  }

 private:
  ThreadPool() : thread_count_(default_thread_count()) {
  }

  // Requires busy_.
  void start_workers() {
    const auto count = thread_count_.load();
    if (workers_.size() + 1 >= count) {
      return;
    }
    runs_ = std::vector<ChunkRun>(count);
    while (workers_.size() + 1 < count) {
      workers_.emplace_back(
          [this, own = workers_.size() + 1, seen = region_.load()] { serve(own, seen); });
    }
  }

  // Requires busy_.
  void stop_workers() {
    stop_.store(true);
    {
      const std::lock_guard lock(mutex_);
      wake_.notify_all();
    }
    for (auto& worker : workers_) {
      worker.join();
    }
    workers_.clear();
    stop_.store(false);
  }

  // Requires busy_. The task is written before any run that leads to it, and
  // the region ends only when every chunk has been counted as done, so a
  // thread that claims a chunk always runs it with the task it belongs to.
  void run_region(std::size_t chunks, void (*invoke)(const void*, std::size_t),
                  const void* context) {
    invoke_ = invoke;
    context_ = context;
    error_ = nullptr;
    done_.store(0);
    const auto threads = runs_.size();
    for (std::size_t thread = 0; thread < threads; ++thread) {
      runs_[thread].assign(chunks * thread / threads, chunks * (thread + 1) / threads);
    }
    region_.fetch_add(1);
    if (sleepers_.load() != 0) {
      const std::lock_guard lock(mutex_);
      wake_.notify_all();
    }
    inside_region = true;
    work(0);
    inside_region = false;
    std::size_t spins = 0;
    while (done_.load(std::memory_order_acquire) != chunks) {
      relax();
      if (++spins > 4096) {
        std::this_thread::yield();
      }
    }
    if (error_) {
      std::rethrow_exception(std::exchange(error_, nullptr));
    }
  }

  // Requires busy_. Each member runs on its own thread, the calling thread
  // being member zero, and the region ends when every member has returned.
  void run_team_region(std::size_t members,
                       void (*invoke)(const void*, std::size_t, std::size_t, TeamBarrier&),
                       const void* context) {
    TeamBarrier barrier(members);
    team_invoke_ = invoke;
    context_ = context;
    team_barrier_ = &barrier;
    done_.store(0);
    // One word holds the team's region and size, so a worker that reads it
    // for the region it saw learns both at once, and one that wakes late for
    // an earlier region never takes a place in this team.
    team_word_.store(((region_.load() + 1) << 16U) | members);
    region_.fetch_add(1);
    if (sleepers_.load() != 0) {
      const std::lock_guard lock(mutex_);
      wake_.notify_all();
    }
    inside_region = true;
    invoke(context, 0, members, barrier);
    inside_region = false;
    std::size_t spins = 0;
    while (done_.load(std::memory_order_acquire) != members - 1) {
      relax();
      if (++spins > 4096) {
        std::this_thread::yield();
      }
    }
  }

  // Works through the thread's own run of chunks, then through whatever the
  // other threads have left of theirs.
  void work(std::size_t own) {
    const auto threads = runs_.size();
    std::size_t finished = 0;
    for (std::size_t offset = 0; offset < threads; ++offset) {
      auto& run = runs_[(own + offset) % threads];
      std::size_t chunk = 0;
      while (run.claim(chunk)) {
        try {
          invoke_(context_, chunk);
        } catch (...) {
          const std::lock_guard lock(mutex_);
          if (!error_ || chunk < error_chunk_) {
            error_ = std::current_exception();
            error_chunk_ = chunk;
          }
        }
        ++finished;
      }
    }
    if (finished != 0) {
      done_.fetch_add(finished, std::memory_order_acq_rel);
    }
  }

  void serve(std::size_t own, std::uint64_t seen) {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
    inside_region = true;
    while (!stop_.load(std::memory_order_acquire)) {
      auto region = region_.load(std::memory_order_acquire);
      const auto deadline = std::chrono::steady_clock::now() + spin_time;
      for (std::size_t spins = 0; region == seen; ++spins) {
        if ((spins & 63U) == 63U && (std::chrono::steady_clock::now() >= deadline ||
                                     stop_.load(std::memory_order_acquire))) {
          break;
        }
        relax();
        region = region_.load(std::memory_order_acquire);
      }
      if (region == seen) {
        // The caller publishes a region and then reads this count, while a
        // worker raises the count and then reads the region, so one of the two
        // always sees the other.
        sleepers_.fetch_add(1);
        {
          std::unique_lock lock(mutex_);
          wake_.wait(lock, [&] { return stop_.load() || region_.load() != seen; });
        }
        sleepers_.fetch_sub(1);
        continue;
      }
      seen = region;
      if (const auto word = team_word_.load(std::memory_order_acquire); (word >> 16U) == region) {
        const auto members = static_cast<std::size_t>(word & 0xffffU);
        if (own < members) {
          team_invoke_(context_, own, members, *team_barrier_);
          done_.fetch_add(1, std::memory_order_acq_rel);
        }
        continue;
      }
      work(own);
    }
  }

  // Held by the thread that is running a region or reconfiguring the pool.
  std::mutex busy_;
  std::atomic<std::size_t> thread_count_;
  std::vector<std::thread> workers_;
  // One run per thread; the calling thread's is the first.
  std::vector<ChunkRun> runs_;

  // The task of the current region and the failure of its lowest chunk.
  void (*invoke_)(const void*, std::size_t){nullptr};
  // The task of the latest team and the barrier its members meet at.
  void (*team_invoke_)(const void*, std::size_t, std::size_t, TeamBarrier&){nullptr};
  // The region of the latest team, shifted left by 16, and its size.
  std::atomic<std::uint64_t> team_word_{0};
  TeamBarrier* team_barrier_{nullptr};
  const void* context_{nullptr};
  std::exception_ptr error_;
  std::size_t error_chunk_{0};

  std::mutex mutex_;
  std::condition_variable wake_;
  std::atomic<bool> stop_{false};
  alignas(64) std::atomic<std::uint64_t> region_{0};
  alignas(64) std::atomic<std::size_t> done_{0};
  alignas(64) std::atomic<std::size_t> sleepers_{0};
};

}  // namespace

void run_chunks(std::size_t chunks, void (*invoke)(const void*, std::size_t), const void* context) {
  ThreadPool::instance().run(chunks, invoke, context);
}

void run_team(std::size_t wanted,
              void (*invoke)(const void*, std::size_t, std::size_t, TeamBarrier&),
              const void* context) {
  ThreadPool::instance().team(wanted, invoke, context);
}

void TeamBarrier::arrive_and_wait() noexcept {
  if (members_ == 1) {
    return;
  }
  const auto generation = generation_.load(std::memory_order_acquire);
  if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == members_) {
    arrived_.store(0, std::memory_order_relaxed);
    generation_.store(generation + 1, std::memory_order_release);
    return;
  }
  std::size_t spins = 0;
  while (generation_.load(std::memory_order_acquire) == generation) {
    relax();
    if (++spins > 4096) {
      std::this_thread::yield();
    }
  }
}

}  // namespace detail

std::size_t cpu_thread_count() {
  return detail::ThreadPool::instance().thread_count();
}

void set_cpu_thread_count(std::size_t count) {
  detail::ThreadPool::instance().set_thread_count(count);
}

}  // namespace microsimulator
