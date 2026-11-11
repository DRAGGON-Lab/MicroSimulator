#pragma once

#include <nanobind/nanobind.h>

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

// Call wrappers for bound methods of stateful native objects.
//
// Long-running native calls release the Python GIL so other Python threads
// keep running. While such a call is in flight its object is registered as
// busy, and any other bound call on the same object raises RuntimeError
// rather than reading or writing state the running call is changing.
namespace microsimulator::python {

class BusyObjects {
 public:
  static void enter(const void* object) {
    const std::lock_guard lock(mutex());
    if (!objects().insert(object).second) {
      throw std::runtime_error("native object is in use by another thread");
    }
    count().fetch_add(1, std::memory_order_release);
  }

  static void leave(const void* object) noexcept {
    const std::lock_guard lock(mutex());
    objects().erase(object);
    count().fetch_sub(1, std::memory_order_release);
  }

  static void require_idle(const void* object) {
    if (count().load(std::memory_order_acquire) == 0) {
      return;
    }
    const std::lock_guard lock(mutex());
    if (objects().contains(object)) {
      throw std::runtime_error("native object is in use by another thread");
    }
  }

 private:
  static std::mutex& mutex() {
    static std::mutex value;
    return value;
  }
  static std::unordered_set<const void*>& objects() {
    static std::unordered_set<const void*> value;
    return value;
  }
  static std::atomic<std::size_t>& count() {
    static std::atomic<std::size_t> value{0};
    return value;
  }
};

// Marks an object busy, then releases the GIL, for the lifetime of the scope.
class ReleasedCall {
 public:
  explicit ReleasedCall(const void* object) : busy_(object) {}
  ReleasedCall(const ReleasedCall&) = delete;
  ReleasedCall& operator=(const ReleasedCall&) = delete;

 private:
  struct Busy {
    explicit Busy(const void* object) : object(object) { BusyObjects::enter(object); }
    ~Busy() { BusyObjects::leave(object); }
    Busy(const Busy&) = delete;
    Busy& operator=(const Busy&) = delete;
    const void* object;
  };

  // Declaration order makes the GIL return before the object is marked idle.
  Busy busy_;
  nanobind::gil_scoped_release release_;
};

// Binds a short method that runs with the GIL held and rejects a busy object.
template <typename Class, typename Result, typename... Arguments>
auto exclusive(Result (Class::*method)(Arguments...)) {
  return [method](Class& self, Arguments... arguments) -> Result {
    BusyObjects::require_idle(&self);
    return (self.*method)(std::forward<Arguments>(arguments)...);
  };
}

template <typename Class, typename Result, typename... Arguments>
auto exclusive(Result (Class::*method)(Arguments...) const) {
  return [method](const Class& self, Arguments... arguments) -> Result {
    BusyObjects::require_idle(&self);
    return (self.*method)(std::forward<Arguments>(arguments)...);
  };
}

template <typename Class, typename Result, typename... Arguments>
auto exclusive(Result (Class::*method)(Arguments...) noexcept) {
  return [method](Class& self, Arguments... arguments) -> Result {
    BusyObjects::require_idle(&self);
    return (self.*method)(std::forward<Arguments>(arguments)...);
  };
}

template <typename Class, typename Result, typename... Arguments>
auto exclusive(Result (Class::*method)(Arguments...) const noexcept) {
  return [method](const Class& self, Arguments... arguments) -> Result {
    BusyObjects::require_idle(&self);
    return (self.*method)(std::forward<Arguments>(arguments)...);
  };
}

// Binds a long-running method that runs with the GIL released.
template <typename Class, typename Result, typename... Arguments>
auto released(Result (Class::*method)(Arguments...)) {
  return [method](Class& self, Arguments... arguments) -> Result {
    const ReleasedCall call(&self);
    return (self.*method)(std::forward<Arguments>(arguments)...);
  };
}

template <typename Class, typename Result, typename... Arguments>
auto released(Result (Class::*method)(Arguments...) const) {
  return [method](const Class& self, Arguments... arguments) -> Result {
    const ReleasedCall call(&self);
    return (self.*method)(std::forward<Arguments>(arguments)...);
  };
}

}  // namespace microsimulator::python
