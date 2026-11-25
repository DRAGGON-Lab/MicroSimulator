# ADR 0026: contact dynamics with growth

- Status: accepted
- Date: 2026-11-13

## Context

A contact scheme for growing cells has to keep cells apart in a full space and let a full space stop growing. A regularized least-squares relaxation of the contacts found in [ADR 0002](0002-contact-mechanics.md), applied after growth is committed, does neither: it treats contacts as equalities and minimizes the sum of squared overlaps, so a colony that fills a closed space keeps elongating into itself, pairs pushed past one radius invert and stay inverted, and no weighting suits both a free colony and a jam.

Growing rod-shaped cells are simulated without those failures by solving contact as a complementarity problem: contact forces are non-negative, gaps stay non-negative, and a force acts only where a gap is closed ([Yan, Zhang, and Shelley 2019](https://arxiv.org/abs/1811.04736); [Yan et al. 2020](https://arxiv.org/abs/1909.06623)). Growth enters the same problem when a cell's elongation responds to the compressive load its contacts put on it ([Weady et al. 2024](https://arxiv.org/abs/2405.10158); [Lerchner](https://home.cit.tum.de/~ler/html-data/bacteria/cell-collectives.pdf)).

## Decision

`Simulation.step(dt, ContactDynamicsParameters)` advances growth and contact together. A controller configured with `MechanicsConfig` takes every step this way; `Simulation.step(dt)` alone grows cells without contact.

### The problem of one substep

Every candidate contact is a row with a multiplier `g >= 0`, the magnitude of its contact force along the contact normal. For rows `J`, cell mobility `M^-1`, and the current signed gaps `phi`, the predicted gap after a substep of length `h` is

```
gap(g) = phi + h J^T M^-1 J g - h L^T (r * f(L g)) + c g
```

- `J g` is the force and torque the rows put on each cell, and `J^T` maps cell velocities to gap rates.
- `M` is the drag of each cell on its surroundings. For a capsule of full length `L = l + 2r` it is `m = drag * L` in translation, `m r^2 / 2` about the axis, and `m (L^2 + 3 r^2) / 12` across it; the finite radius keeps rotation about the axis from being free. A fixed cell has no mobility.
- `L g` is the axial load on each cell. A row contributes its multiplier times the cell's length lever, the fraction of the cell's elongation that closes that contact: the cylinder stretches uniformly about its center and the caps ride on its ends, so the lever is one half for a contact on a cap along the axis and zero for a contact on the side.
- `r` is the nominal elongation rate `growth_rate * length`, taken from the lengths at the start of the step, and `f` is the growth load response. `L^T` maps elongation to the gap it closes.
- `c` is the compliance of the row: `1 / contact_stiffness` between cells and `1 / wall_stiffness` against a constraint.

The solution satisfies `g >= 0`, `gap >= 0`, and `g * gap = 0` row by row. Those are the optimality conditions of a convex energy whose gradient is `gap`, because `f` is non-increasing. In a full space the multipliers rise until the loads stop growth.

The growth load response `f` is one of:

- `none`: elongation ignores load;
- `exponential`: `f = exp(-sensitivity * load)`;
- `linear`: `f = 1` up to `free_load`, falling linearly to zero at `stall_load`.

Shrinkage is never scaled. Loads are forces in units of drag times length squared per time, so a stall load is chosen relative to the drag.

### Compliance

A contact is a unilateral linear spring solved implicitly: a resting contact under force `g` overlaps by `g / contact_stiffness`, and the solve is stable at any stiffness. Walls are rigid next to cells, so a contact with a constraint uses the larger `wall_stiffness`, by default a hundred times the cell value; cells then rest on a wall within a small fraction of the overlap they have with each other, at little cost to the solve. Compliance keeps the problem well posed when a space holds more than it can: the forces of an over-full region are finite, where rigid constraints have no solution. It also means an overlap `e` that a jammed pair cannot recover persists as a force `e * contact_stiffness`. Where growth responds to load, the accepted overlap is therefore held to the overlap that a tenth of the load scale of the response presses into a contact. The forces of a rigid jam above the stall load depend on the history of those small overlaps and are not otherwise determined.

A contact already deeper than `max_overlap_recovery` asks for only that much separation in one substep and is stiffened in proportion, so a pair free to move separates at a bounded rate while a pair that cannot move still meets the full spring force of its overlap.

### Solve

The energy is minimized by Barzilai–Borwein projected gradient descent with the two step lengths alternating. The iteration needs only the gap of a multiplier vector, starts from a step length bounded by the largest diagonal entry of the linear part, and stops when the largest violation of the complementarity conditions, a length, falls below a quarter of the accepted overlap. `core/contact_solver.hpp` holds the iteration; a backend supplies the gap evaluation and the reductions over its own storage.

### Substeps and refinement

A step is divided so that no cell end elongates by more than `max_growth_fraction` radii in one substep. Each substep:

1. finds the cell pairs and walls within `candidate_margin` and builds their rows at the current geometry;
2. solves for the multipliers and moves the cells: positions by velocity, directions by the axis-angle rotation, lengths by the loaded elongation;
3. finds the contacts that overlap at the moved geometry by more than the accepted overlap plus what their force accounts for;
4. appends a row for each, linearized where it was found with its offset set so that the multipliers that produced the move predict the gap the move left, and solves again from the current multipliers.

Rows are never rebuilt once added, which is the recursively generated complementarity scheme of [Palmer, Aktulga, and Gao 2025](https://arxiv.org/abs/2506.14097). A substep that does not converge, or needs more than `max_refinements` rounds, is retried at half its length, and after `max_step_halvings` the step throws. A step that throws leaves the simulation unchanged.

The stepping carries no state between steps, so a resumed run repeats an uninterrupted one.

### Walls

Constraints are rows in the same solve. A box or cylinder that cells live inside contributes one row for every face a cell end is near, from `find_external_contact_faces`: each face is an independent half-space, and a row for the nearest face alone leaves a cell end in a corner free to cross the other wall. These rows come from the host constraint geometry on every backend.

### Order within a step

Growth and contact run first, then species, coupled, and signal updates, which see the elongation that happened; uptake tied to volume change is charged for it.

### Flow

With `flow_drift` set, the velocity and Jeffery rotation that the signal grid's velocity field gives each cell ([ADR 0021](0021-flow-drift.md)) enter the solve as a known velocity `u`: the predicted gap gains `h J^T u` and each cell moves at `u` plus what its contact forces add. Contacts therefore resist the flow pressing cells together or into a wall, rather than correcting the overlap after a separate drift move. The rates are evaluated at the midpoint of each cell's free motion over the substep, fixed cells do not drift, and a substep carries no cell further than a quarter of the smallest grid spacing.

## Validation

- A cell between two walls fills the space, stalls with its axial load within 5% of the stall load, and stops consuming nutrient.
- A closed chamber fills to under one layer of cells and its population and footprint then hold, with overlap at what the contact forces press into the cells.
- An unconfined colony without load response elongates exactly as it does without mechanics, stays planar, and packs to a fraction a little under one.
- That colony's radial load profile follows the continuum solution for an incompressible growing layer dragged on its substrate, `p(r) = zeta g (R^2 - r^2) / 4` with `zeta = drag * packing / diameter`, without fitted constants.

## Limits

- Contacts are frictionless and cells are rigid apart from the contact compliance.
- Contact stiffness trades overlap against cost: in an unconfined colony of 2,000 cells without load response, ten times the stiffness leaves a fifth of the overlap and takes about eight times the iterations. Loads grow with the size of a colony whose growth ignores them, so the stiffness that keeps overlaps small grows with it.
- A cell squeezed directly between two walls, with no softer contact to yield, has its load resolved only to the accepted overlap times the wall stiffness.
- The cost of a step is the number of projected-gradient iterations, which grows with the number of cells that push on each other through the colony; published solvers of this kind report on the order of a thousand iterations per step in dense colonies.
- A perfectly collinear line of cells between two walls is an equilibrium and does not buckle. Division plans perturb daughter directions to break that symmetry.
- The exponential response never stops growth, so a closed space keeps filling slowly as its contact overlaps deepen.
- The cited growth models are two-dimensional and unconfined. Monolayers confined between a floor and a ceiling are covered by the tests here, not by that literature.
