# Compatibility and migration

MicroSimulator preserves the useful modeling semantics of CellModeller while replacing OpenCL source injection, executable pickle state, private solver access, and UI-owned simulation control with explicit interfaces. Compatibility is based on observable model behavior, not on reproducing implementation details line for line.

## Using CellModeller material

- [Updating from CellModeller2](microsimulator-rename.md) covers the MicroSimulator package and command rename and existing output files.
- [Running CellModeller Python models](legacy-python-models.md) describes the supported callback interface and its limits.
- [Importing CellModeller pickle snapshots](legacy-pickle-import.md) explains the trusted, one-way state migration command.
- [Example compatibility matrix](legacy-example-matrix.md) lists the pinned examples exercised through the callback adapter or typed translations.
- [Typed translations of equation models](legacy-example-migrations.md) documents the equations and modeling choices used where legacy OpenCL source cannot run directly.
- [Compatibility overview](feature-ledger.md) summarizes the behavior available in MicroSimulator.

The [tutorial source reference](tutorial-source-provenance.md) records how the current teaching models relate to the CellModeller wiki, bundled examples, and SimBOL sources. Tutorial readers normally do not need this information unless they are comparing results with an older model.

## Scientific comparison records

These documents retain source-level analysis needed to interpret older models and reproduce compatibility decisions:

- [Rod mechanics](legacy-mechanics-audit.md)
- [Grid signaling](legacy-signaling-audit.md)
- [Fixed cells](legacy-fixed-position-audit.md)
- [Neighbor diffusion](legacy-neighbor-diffusion-audit.md)
- [SBML import](legacy-sbml-import-audit.md)
- [Analysis workflows](legacy-analysis-audit.md)
- [Interactive viewer behavior](legacy-viewer-audit.md)
- [Recorded trajectory comparisons](legacy-trajectory-evidence.md)
