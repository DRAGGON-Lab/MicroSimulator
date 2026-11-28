# ADR 0030: provider interfaces

- Status: accepted
- Date: 2026-11-27

## Context

A stage that only one backend implements makes the other backends incomplete in a way the compiler cannot see. The stage either runs on the host for every backend or is reachable from Python alone, and a GPU run then pays for its round trips. A stage that every backend implements in its own language makes three implementations to keep in agreement.

## Decision

**Compute is a set of pure-virtual provider interfaces.** `cpp/include/microsimulator/backend.hpp` divides the compute a backend provides by the physics it serves:

| Interface | Stages |
| --- | --- |
| `MechanicsProvider` | growth, cell and wall contact detection, the contact force solve and its response, the device that holds a contact dynamics step, flow drift |
| `TransportProvider` | species, signal transport, cell–grid exchange |
| `FlowProvider` | the shallow and resolved flow solves |
| `CouplingProvider` | biomass deposition, the shallow-flow closure, the Brinkman drag |

`ComputeBackend` derives from all four. Each backend is a `final` class deriving from it, and its translation unit asserts `!std::is_abstract_v` of the class. A provider method that a backend leaves out therefore fails that backend's build, with a message naming the backend. The CUDA backend is compiled in continuous integration on every pull request. Every backend implements every feature, so `supports` returns true for each.

**Engine code reaches per-element compute only through a provider.** The `layering` test (`cmake/CheckLayering.cmake`) fails when a file under `cpp/core` includes a backend's headers, or runs a host loop over cells, sites, faces, or contacts outside the files it lists. The listed files hold the compute that runs on the host for every backend: contact-graph construction, the CPU signal transport, and flow-system assembly. A contact dynamics step runs on the backend's device through `ContactStepDevice` ([ADR 0031](0031-resident-contact-steps.md)).

**Each backend implements a stage in its own language.** The CPU backend writes it in C++ over `detail::parallel_for`, the CUDA backend in CUDA C++ with CUB, and the Metal backend in the Metal Shading Language, each in the idiom of its platform. The structures a stage passes between the host and a device are declared once in `cpp/shared/device_types.h`, which the host, CUDA, and Metal builds share. What a stage needs beyond its elements (validation, the layout of its inputs, and the interpretation of its outputs) is backend-independent C++ in `cpp/core`, such as `cpp/core/deposition_plan.hpp`.

**A run repeats exactly on its backend.** A sum over many elements takes its terms in an order that does not depend on scheduling. A scatter into shared sites is a gather in which each site sums its contributors in ascending index order: the CPU adds cells into sites one cell at a time, and the devices count, scan, and fill each site's contributors with integer atomics and sort them before summing. No floating-point atomic feeds the state. Backends agree with one another within stated tolerances.

## Consequences

- The biomass deposit, the shallow-flow closure, and the Brinkman drag; contact detection between cells and between cells and external constraints, the latter in both its nearest-face and every-face forms; the contact solve; and flow drift are native on every backend: `cpp/cpu/cpu_deposition.cpp` and `cpp/cpu/kernels`, `cpp/metal/kernels`, and `cpp/cuda/cuda_cell_kernels.cu` with `cpp/cuda/kernels`. The Python functions `colony_volume_fraction`, `colony_species_density`, `colony_mobility`, `colony_flow_coupling`, and `stokes.colony_drag` are wrappers that make one native call on the selected backend.
- `cell_kernels_conformance` requires the CPU to give the same bits on any number of threads, each device to repeat its results exactly, and the devices to agree with the CPU: deposits and drift to a relative 10⁻⁵, cell and external contacts in their rows, separations, and weights, and contact solves in the response they produce. It checks conservation, wall separation, truncation, the closure formulas, container faces, drift, cell contacts against the exhaustive search, and contact solves on both device paths. `python/tests/deposition_reference.py` is an independent binary64 implementation against which the native results are bounded.
- A contact solve returns the per-cell response its own kernels computed for the multipliers it reached, so contact dynamics moves cells by what the backend solved.
