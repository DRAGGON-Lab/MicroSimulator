#pragma once

#include <cstddef>

namespace microsimulator {

// The number of threads the CPU backend and the host side of every backend
// spread their work over, counting the calling thread. Results do not depend
// on it.
//
// The default is the MICROSIMULATOR_NUM_THREADS environment variable when it
// holds a positive integer, and otherwise the number of performance cores the
// machine reports.
[[nodiscard]] std::size_t cpu_thread_count();

// Sets the thread count. Zero restores the default. Throws std::logic_error
// when called from inside parallel work, and waits for parallel work running
// on another thread to finish.
void set_cpu_thread_count(std::size_t count);

}  // namespace microsimulator
