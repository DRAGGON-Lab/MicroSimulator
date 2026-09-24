# Plasmid segregation, contacts, and conjugation

This tutorial uses plasmid segregation and conjugation to show how discrete biological state, stochastic events, and contact-dependent behavior fit into a MicroSimulator model.

New founders preserve their requested length unless it exceeds the single sampled division target. This also handles the rare short Gaussian target in the conjugation model without rejection sampling. Checkpoint restoration keeps stored lengths and targets. See [founder initialization](biophysics-and-growth.md#length-and-volume).

## 1. Incompatible plasmid segregation

```console
uv run microsimulator view \
  --model examples/tutorials/plasmid_segregation.py \
  --parameter copies_per_cell=10 \
  --seed 42 \
  --dt 0.02 \
  --checkpoint-output results/plasmids.json \
  --open
```

The founder contains five copies each of plasmids A and B. Before division, every copy is duplicated. The resulting `2N` discrete copies are shuffled with the checkpointed random stream, and exactly `N` are assigned to each daughter. The following invariants are checked on every event:

```text
daughter_1 total = daughter_2 total = N
daughter_1 A + daughter_2 A = 2 parent A
daughter_1 B + daughter_2 B = 2 parent B.
```

The model exposes `copies_per_cell`, so the total number of copies assigned to each daughter is explicit:

```console
--parameter copies_per_cell=6
```

### Why copy counts live in controller state

Plasmid copy numbers are integers. Treating them as concentrations would make ordinary growth dilution turn `5` into a fractional copy before division. The tutorial therefore stores the authoritative `{a, b}` counts by stable cell ID in data-only controller state. It publishes dimensionless A and B fractions to species channels 0 and 1 after each completed step for inspection, but those channels do not drive the segregation calculation.

Cell type summarizes the discrete outcome:

| Type | State                 |
| ---: | --------------------- |
|    0 | both plasmids present |
|    1 | A fixed; B absent     |
|    2 | B fixed; A absent     |

Use cell-type coloring to see sectors. A rendered sector is a stochastic realization, not evidence for a particular fixation probability. Run multiple seeds and compare a declared endpoint such as the fraction of single-plasmid cells at a fixed total cell count.

The checkpoint is ordinary JSON, so exact counts can be inspected without executing the model:

```console
jq '.controller.state.plasmids' results/plasmids.json
```

## 2. Contact graphs

MicroSimulator derives a typed contact graph from current capsule geometry:

```python
graph = simulation.find_cell_contacts()
for cell in simulation.cells():
    neighbors = graph.neighbor_ids(cell.slot)
```

Stable IDs identify cells across compaction and checkpoints. Slots are dense, temporary indices used to query one graph. Recompute the graph after growth, division, or mechanical relaxation; neighbor lists are not persistent cell state.

Parallel rods can produce two geometric contact rows. `neighbor_ids(slot)` returns ascending unique stable IDs, so a pair is one biological neighbor even when the mechanics operator needs two rows. Constraint contacts are a separate typed graph and never appear as cell neighbors.

Contact graphs are derived on demand and have no fixed scientific contact cap; allocation failures are explicit.

## 3. Contact-dependent conjugation

```console
uv run microsimulator view \
  --model examples/tutorials/conjugation.py \
  --parameter transfer_probability=0.1 \
  --seed 42 \
  --dt 0.02 \
  --open
```

The founders are an acceptor (type 0) and donor (type 1). At each regulation step, an acceptor independently tests every donor or transconjugant neighbor. A successful event changes it to a transconjugant (type 2), which can transmit on later steps.

The default `0.1` is a probability *per simulation step*, so changing `dt` changes the implied physical hazard. For a time-calibrated rate `lambda`, replace it with

```text
p(dt) = 1 - exp(-lambda dt)
```

and make `dt` available to the controller’s declared model state or custom step orchestration. Do not compare runs with different `dt` while calling the per-step probability a fixed kinetic rate.

The update is simultaneous: contacts and neighbor cell types come from one pre-update snapshot. A newly converted transconjugant cannot infect another cell until the next step, so results do not depend on callback iteration order.

## Experiments

- Estimate fixation or coexistence frequencies over a declared seed set.
- Compare per-step transfer with a time-calibrated hazard at two `dt` values.
- Make transfer probability depend on donor growth rate, but state whether the intended quantity is configured growth or mechanically realized elongation.
- Export contact edges and ask whether transfer events concentrate in cells with high degree. Avoid counting two-row parallel contacts as two neighbors.
