# Tutorial sources and model translations

The MicroSimulator tutorials draw from the maintained CellModeller wiki, bundled CellModeller examples, and SimBOL examples. This page records those relationships for readers who need to compare equations or reproduce an older workflow; the tutorials themselves focus on using the current modeling interface.

## Source revisions

- CellModeller examples: commit `4896f543c6250f053eea2312e628cc3a96bf7408`
- CellModeller wiki: commit `95587c11899677b7cba87c64bc20210fb1f8f6ce`
- SimBOL: commit `54501f9da6f9809588be48b854a6c4f8abd933b5`

## Tutorial relationships

| Source material | MicroSimulator model or guide |
| --- | --- |
| Wiki Tutorial 1a | `biophysics.py`, `basics` |
| Wiki Tutorial 1b | `biophysics.py`, `competition` |
| Wiki Tutorial 1c | `biophysics.py`, `box` |
| Wiki Tutorial 2a | `gene_expression.py`, `constitutive` |
| Wiki Tutorial 2b | `gene_expression.py`, `oscillator` |
| Wiki Tutorial 3 | `signaling.py`, `mutualism` |
| Old Example 1 and its exercises | `biophysics.py`, `basics`, `two_types`, `short_cells` |
| Old Example 2 | `gene_expression.py`, `legacy_constitutive`, `dilution`, `derepression` |
| Old Example 3 | `signaling.py`, `single_gene` |
| Old Example 4 | `signaling.py`, `communication` |
| Old Example 5 | `plasmid_segregation.py` |
| Contact graph and conjugation examples | `conjugation.py` and the analysis tutorial |
| Legacy analysis scripts | analysis tutorial and analysis recipes |
| SimBOL `CM_BBa_01`–`05`, `CM_BBa_I5200` | `simbol_circuits.py` |
| SimBOL `CM_Danino.py` | `danino_clock.py` |
| SimBOL CellModeller notebook | SimBOL tutorial workflow description |

Exact equation translations used by the executable compatibility matrix remain under [`examples/legacy`](../../examples/legacy). The models under [`examples/tutorials`](../../examples/tutorials) are teaching versions: they consolidate related examples, expose parameters consistently, and support exact resume.

## Translation conventions

The teaching models preserve the biological question, initial conditions, rate equations, strain roles, division rule, and physical geometry where those are well-defined. They express those ideas through current interfaces:

- `CellInit` defines cell state;
- `NativeController` or an explicit controller owns regulation and stochastic state;
- `RatePlanBuilder` represents biological equations;
- `SignalGridSpec` and `CoupledRatePlan` define transport and cell-grid exchange;
- plane and sphere constraints define physical boundaries;
- checkpoints store restart state; and
- scene and analysis exports provide visualization and quantitative output.

Some source material uses `targetVol` as a length, treats callback attributes alternately as molecule counts and concentrations, or applies chemical updates once per GUI step without an explicit `dt`. The current tutorials name the chosen interpretation where it affects scientific meaning.

## SimBOL source workflow

At the pinned SimBOL revision, the intended workflow is:

```text
SBOL 3 document
  -> sbol3_to_json_converter
  -> summarized circuit JSON
  -> parameter preparation / UI
  -> CellModeller-specific generated Python
```

The checked-in `notebooks/CellModeller.ipynb` stops after environment setup, SBOL upload, JSON conversion, and parameter-form display. The generated `test/CM_*.py` files and matching JSON fixtures are therefore the concrete sources used for the MicroSimulator examples.

These examples are explicit translations, not a general SBOL-to-rate-plan import path. MicroSimulator does not accept arbitrary SimBOL output without a versioned intermediate schema, declared parameter units, explicit handling of unsupported SBOL semantics, and source provenance.
