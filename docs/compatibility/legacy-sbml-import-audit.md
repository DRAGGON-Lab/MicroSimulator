# SBML import behavior

## CellModeller module

CellModeller provides `CellModeller/Regulation/SBMLImport.py`, although no bundled model or test exercises it.

## CellModeller behavior

The module uses libSBML to read a document, walks reaction kinetic-law ASTs, and assembles a Python expression for each participating species derivative. It recognizes basic arithmetic and power, substitutes constant local or global parameter values, and treats species identifiers as Python variables. Reaction reactants subtract `stoichiometry * rate`; products add it. It also extracts an initial amount or concentration.

The generated text defines `getRates(cells)` and is executed into a dynamically created Python module.

## Unsupported and ambiguous semantics

The checked-in importer is not a reliable executable reference:

1. Python 3 compilation fails because indentation mixes tabs and spaces, and the removed Python 2 `new` module is imported.
2. SBML consistency checking is disabled globally by default.
3. Parse errors and a missing model are printed rather than rejected before translation continues.
4. Unknown identifiers are silently replaced by the numeric value one.
5. In-memory input is written to a fixed `tmp.sbml` path, creating collision and overwrite hazards even though modern libSBML supports string parsing.
6. Arbitrary generated Python is executed, and a separate function fetches source over unauthenticated HTTP before executing it.
7. Compartments, units, amount-versus-concentration semantics, boundary and constant species, rules, events, initial assignments, conversion factors, delays, functions, and dynamic stoichiometry are not modeled.
8. Species ordering is derived from reaction participation rather than the SBML species list. A species used only as a modifier can be referenced by a kinetic law without receiving a generated Python variable.
9. Missing or non-positive initial values are collapsed to zero, so a declared negative concentration is silently changed.

## MicroSimulator behavior

The reusable behavior is reaction-stoichiometry compilation, not source-code generation. MicroSimulator uses libSBML's document model and error log, then compiles a declared subset directly into the typed rate-plan IR. Unsupported or ambiguous constructs fail with a path-specific import error. No legacy network channel, temporary-file workaround, generated source, or `exec` path is retained.
