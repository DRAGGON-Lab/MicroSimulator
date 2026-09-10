# ADR 0011: compile a bounded SBML subset to typed rate plans

- Status: accepted
- Date: 2026-08-15

## Context

The CellModeller SBML importer parses a small part of kinetic-law MathML, emits Python source, and executes that source as a new module. It ignores several SBML semantics and silently replaces unknown identifiers with one. A detailed comparison is recorded in `docs/compatibility/legacy-sbml-import-audit.md`.

MicroSimulator already has a data-only `SpeciesRatePlan` instruction graph that executes independently on CPU, Metal, and CUDA and survives checkpoints. An SBML importer should be a compiler into that IR, not another runtime.

## Decision

SBML import is an optional Python feature backed by libSBML. Parsing a file or in-memory string returns immutable metadata, initial concentration levels, and a validated `SpeciesRatePlan`. It never emits or evaluates Python, C++, MSL, CUDA, or OpenCL source.

The first supported contract is intentionally narrow:

- SBML Level 3 Version 2 Core models with exactly one compartment whose constant size is one;
- species represented as concentrations, with initial concentration or an initial amount that is equivalent in the unit-volume compartment;
- constant global and kinetic-law-local parameters with explicit finite values;
- reactions with a kinetic law and constant numeric stoichiometry;
- arithmetic `+`, binary or unary `-`, `*`, `/`, power, natural logarithm, and exponential expressions over species, parameters, and finite literals; and
- constant or boundary-condition species, whose derivative is zero.

The compiler orders species by SBML model order. For reaction `r` with kinetic law `v_r`, species derivative `i` is

```text
d c_i / dt = sum_r nu[i, r] * v_r.
```

The unit compartment avoids an implicit amount-to-concentration conversion. MicroSimulator then applies its documented post-growth concentration dilution before evaluating this rate plan.

The importer rejects malformed documents, error-severity libSBML diagnostics, multiple or non-unit compartments, non-constant parameters or compartments, conversion factors, rules, events, constraints, initial assignments, function definitions, dynamic stoichiometry, missing kinetic laws, unsupported MathML, and unresolved identifiers. It does not guess defaults that affect dynamics.

## Consequences

- Imported dynamics use the same native interpreter and checkpoint format as hand-built rate plans.
- Backend parity depends on typed-plan conformance, not a backend-specific SBML parser.
- The subset is smaller than SBML Core but each accepted construct has an explicit MicroSimulator meaning.
- Expanding the subset requires fixtures that demonstrate the SBML semantic mapping, not merely successful parsing.
