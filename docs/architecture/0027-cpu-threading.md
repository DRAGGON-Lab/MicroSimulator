# ADR 0027: threaded CPU execution

- Status: accepted
- Date: 2026-11-25

## Context

The CPU backend is the numerical reference, and the host side of every backend runs on the CPU too: the contact stepper, the contact search, and the flow reductions are shared code. Their loops are over cells, contacts, and grid sites, and almost every iteration is independent of the others.

A reference that changes with the machine it runs on is not a reference. A sum of floating-point terms depends on how the terms are grouped, so a thread count that changes the grouping changes the result.

## Decision

`cpp/core/parallel.hpp` provides `parallel_for` and `parallel_reduce` over a fork-join thread pool in `cpp/core/parallel.cpp`. The pool is part of the core library and uses only the standard library, so the build has no threading runtime to find, link, or ship in a wheel.

### Results do not depend on the thread count

A loop is cut into pieces of a fixed number of indices, its *grain*. The pieces depend on the loop length and the grain, never on the thread count.

- A loop whose iterations write only their own outputs gives the same result however its pieces are scheduled.
- A reduction folds each piece on its own and combines the pieces in index order, so a floating-point sum has one grouping.
- Work that gathers from several sources is arranged so that each destination sums its own terms in a fixed order. The contact solve keeps, for every cell, the rows that touch it in row order; a cell sums its own force and torque from them rather than rows adding into shared cells.
- Work that scatters into shared destinations and has no such arrangement stays on one thread. The coupled step computes every cell's exchange with the grid in parallel and adds the exchanges to the grid sites one cell at a time.
- Contact candidates and contacts are counted, the counts summed into ranges, and each cell or candidate writes its own range, as on the devices. A cell orders its own partners by identifier.
- The contact solve sums its step terms by a fixed tree over blocks of 256 rows.

`tests/cpp/thread_invariance_test.cpp` runs contact dynamics, signal transport, coupled and species rates, and both flow solves on one and on several threads and requires identical bits.

### The pool

Solver iterations enter a parallel region tens of microseconds after leaving the last one, so the pool is built for short regions.

- A worker spins for 100 microseconds waiting for the next region before it sleeps on a condition variable.
- A region gives every thread its own run of consecutive pieces. A thread works through its own run and then takes what other threads have left of theirs, so threads do not contend for a shared counter, and a thread that the operating system has descheduled does not hold the others back.
- The calling thread takes part, and finishes a region alone if no worker arrives.
- When pieces throw, the exception of the lowest piece is rethrown.
- A parallel call made from inside a region, or while another thread is using the pool, runs on the calling thread.
- A team is a region in which each of a fixed number of threads runs one member and the members meet at a spin barrier, with no chunks taken from one another, so every member is running at once. The contact solve runs all its iterations in one team, a member for every 512 rows up to the thread count: each member responds for its own bodies and sums its own blocks of rows, and the first member sums the block sums and records the step. An iteration then costs three barriers in place of two parallel regions, and the sums are those of the blocks, so the result is the same on any number of threads.

### Thread count

The default is the `MICROSIMULATOR_NUM_THREADS` environment variable when it holds a positive integer, and otherwise the number of performance cores the machine reports. `set_cpu_thread_count` and `cpu_thread_count`, in C++ and in the Python package, change and read it at run time.

### What is threaded

- The contact solve, the contact search and its adjacency lists, and the per-cell and per-contact work of the contact stepper.
- Signal transport, the Jacobi sweeps of an implicit signal step, and their residuals.
- Per-cell species and coupled rates, and the sampling of the flow field for drift.
- The operators and vector operations of the depth-averaged and resolved flow solves.

## Consequences

- A run is reproducible across thread counts and machines with the same toolchain, so a result can be checked on a laptop and produced on a workstation.
- A grain is part of the numerical definition of a reduction. Changing one changes results in the last bits, as changing the order of a sum would.
- The pool is concurrency code that the project owns. A ThreadSanitizer build of the CPU tests runs on every change to native code.
- Speedup is bounded by the work that stays on one thread and, on large systems, by memory bandwidth rather than core count.
- Metal and CUDA runs use the same threaded host code for contact search and stepping.
