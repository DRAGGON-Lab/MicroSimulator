# ADR 0031: contact dynamics steps held on the device

- Status: accepted
- Date: 2026-11-27

## Context

A contact dynamics step ([ADR 0026](0026-contact-dynamics.md)) is a sequence of substeps, each a few refinement rounds of detection, row assembly, a solve, a move, and an overlap check. When the host drives those stages, a device backend copies the geometry out and back, and waits for each stage, several times a round. The waits, not the arithmetic, then set the cost of a step on a GPU.

## Decision

**A step lives in a backend's storage.** `cpp/core/contact_step_device.hpp` declares `ContactStepDevice`. It holds the buffers that `cpp/shared/contact_step_layout.h` names, and can do four things to them:

- run a per-element stage;
- run a scan;
- run the contact solve;
- copy data in or out.

Every method is pure virtual. `MechanicsProvider::contact_step_device` gives each backend's device:

- the CPU: `cpp/cpu/cpu_contact_step.cpp`;
- Metal: `cpp/metal/metal_contact_step.mm`;
- CUDA: `cpp/cuda/cuda_contact_step.cu`.

The device persists across steps, and its buffers grow to the largest step they have held.

**Every stage of a substep is a native kernel on each backend.** `cpp/shared/contact_step_layout.h` lists the stages and lays out the table of the substep's buffers, its status words, and its parameters. Each backend writes every stage in its own language as a function of one element over that table:

- body mobilities and drift;
- detection at the base or trial geometry;
- rows and their keys;
- the per-cell row lists;
- warm starts from the last accepted substep;
- the incidence of the solve;
- the probe of a new substep length;
- the move;
- the overlap check;
- appending rows;
- accepting a substep.

The stages are listed once, in `MS_CONTACT_STEP_KERNELS`, and each backend implements them:
- the CPU in `cpp/cpu/kernels/contact_step.hpp`, run through `detail::parallel_for`;
- Metal in `cpp/metal/kernels/contact_step_kernels.metal`, as kernels that read the table through GPU addresses;
- CUDA in `cpp/cuda/kernels/contact_step.cuh`, as `__global__` kernels.

The solve sums its step terms in one fixed order, so a step repeats exactly on its backend.

**Stages report through device status.** The scalars a round needs live in the words of a device `status` buffer. These are the counts, the slack, the step length, the rotation, and the largest overlap, plus the reason an attempt failed. A stage that fails, or runs out of room, records it there, and every later stage does nothing.

`cpp/core/contact_dynamics.cpp` therefore enqueues a whole refinement round and reads its status once. The length control and the refinement logic stay in core, over that status, and the state receives the geometry only when the step is accepted.

A round that runs out of room is replayed with larger buffers. The replay starts at the stage that overflowed, whose inputs are unchanged, so it gives the same result as if the room had been there.

**A device chooses how to solve.** Metal and CUDA solve a system of up to 4,096 rows in one threadgroup or block. A larger system runs a step at a time over the device, in batches of 32 steps. The host learns the size from the previous round's status.

## Consequences

- A round costs one wait on Metal and CUDA when its system fits one threadgroup, plus one wait per 32 solver steps when it does not.
- `contact_dynamics.cpp` runs no loop over cells or contacts on the host, so it leaves the layering allowlist ([ADR 0030](0030-provider-interfaces.md)).
- The CPU runs the same stages over host buffers, so the CPU reference and the device backends share one implementation of the substep. A device stage runs over the capacity the host enqueues, its idle threads returning at once. The CPU reads status between stages, so it runs each stage only over the elements its extent in `MS_CONTACT_STEP_KERNELS` names: cells, candidates, contacts, or rows.
- `contact_dynamics_conformance` requires a filling chamber to repeat exactly on every backend and to fill to within 15% of the CPU's population.
