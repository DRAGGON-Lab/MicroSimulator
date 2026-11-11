# ADR 0003: typed species rate plans

Status: accepted

## Context

Splicing model-provided kernel source into an integrator makes a model difficult to validate, inspect, serialize, or execute consistently through independent Metal and CUDA APIs. The species step carries an important concentration convention: growth dilution happens before rates are evaluated, and the resulting rates are applied with an explicit Euler update.

## Decision

Each simulation declares an immutable species count. Species levels are finite single-precision concentrations stored cell-major by compact slot. A new cell may provide exactly that many levels or omit them to receive zeros. Equal division copies the parent's concentrations into both daughters. Under the declared effective-volume convention, the two daughter volumes sum to the parent volume, so this also preserves total species amount.

Rate equations use a validated acyclic instruction plan rather than injected kernel text. Instructions can read species, position, length, radius, growth rate, cell type, effective volume, and effective surface area. They compose those values with arithmetic, elementary functions, comparisons, and a typed select. Every operand must refer to an earlier instruction and the plan has one declared output per species. This representation is directly inspectable and serializable, and each native backend is designed to interpret the same validated data with its own C++, MSL, or CUDA C++ implementation.

For one biological step:

1. advance rod length with the declared explicit growth recurrence;
2. dilute every concentration by `V_old / V_new`;
3. evaluate every rate output from the complete diluted pre-Euler state;
4. update every species simultaneously with `c_next = c_diluted + dt * rate`.

The effective volume and surface area are `V = pi * r^2 * (length + 2r)` and `A = 2 * pi * r * (length + 2r)`. Negative finite concentrations remain legal; the engine does not silently clamp model dynamics. Non-finite instructions or updates fail explicitly.

## Consequences

- Models are backend-neutral data rather than executable source fragments.
- Checkpoints can persist both species state and the exact rate plan.
- Native kernels can use an interpreter first and specialize or generate backend-native code later without changing model semantics.

Native Python models construct this IR through `RatePlanBuilder`. Each source method returns an immutable `RateExpression`; ordinary arithmetic operators and typed comparison/select methods append validated, backward-only instructions. `species_plan` and `coupled_plan` close the graph with the declared output vectors. Expressions carry their builder identity, so mixing graphs fails before a plan reaches native validation. The builder is authoring syntax only: the checkpoint still stores the same backend-neutral instruction data and each backend still executes its native interpreter.

- Plan validation isolates structural errors before a simulation step.
- The initial plan does not yet contain grid-signal reads; those enter with the coupled cell/grid rate contract.
