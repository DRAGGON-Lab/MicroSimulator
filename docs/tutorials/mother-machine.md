# Mother-machine growth and washout

The [mother-machine example](../../examples/mother_machine.py) models six single-file growth channels connected to a perfusion channel. All cells can translate and rotate. A closed end and narrow side walls confine each cell row; elongation and division push descendants toward the opening. Cells entering the perfusion channel are carried downstream and removed near its outlet. There is no attachment flag, tether, position reset, or division-triggered release.

## Experimental basis and model geometry

Wang et al., [Robust Growth of Escherichia coli](https://doi.org/10.1016/j.cub.2010.04.045) (2010), describe growth channels approximately 25 µm long, 1.5 µm wide, and 1.4 µm deep in their [supplemental experimental procedures](https://dev.jun.ucsd.edu/files/publications/RobustGrowth_complete_CurrBiol2010.pdf). Their device retains the closed-end lineage while descendants leave through the open end. The authors passivated the device to prevent cell adhesion; this example therefore uses geometric confinement rather than permanent attachment.

`microsimulator.microfluidics.MotherMachineDevice` uses those growth-channel dimensions for both mechanical walls and the transport mask. Growth channels run from x=0 to x=25 µm. Their openings face the perfusion channel at negative x; medium flows along positive y. The floor is z=0. Seven rods initially occupy each growth channel, with a diameter of 1 µm and centerline lengths sampled between 2 and 2.4 µm. This represents an already loaded device; loading is not simulated.

The six-channel count, 7.5 µm pitch, 12 × 45 × 7 µm perfusion segment, 40 µm/s mean inlet speed, and cell parameters are illustrative choices. They do not reconstruct or calibrate the complete experimental chip. The perfusion segment is deliberately compact enough to inspect the growth channels and washout together.

## Run in the viewer

Build the viewer with `pnpm --dir viewer build`, then run:

```console
uv run --no-sync microsimulator view --model examples/mother_machine.py --backend cpu --seed 42 --dt 0.025 --parameter growth_rate=0.3 --frame-steps 2 --fps 20 --viewer-dist viewer/dist --checkpoint-output results/mother-machine-live.json --open
```

Click **Play**. Use **Cell type** coloring: type 0 marks the tracked closed-end lineage, and type 1 marks other cells. The horizontal branches contain the growing rows, while the vertical channel carries medium and escaped descendants. Select **XY plane (Z)** and signal slice **1**, corresponding to z=0.7 µm, to see dye or nutrient at cell height. Dye enters for three seconds, followed by three seconds of clear medium. Its penetration into the branches is predominantly diffusive.

The `growth_rate=0.3` override accelerates biology for this demonstration. It changes the relative timescales of growth, flow, and diffusion; measurements from that run must not be interpreted as experimental predictions. Omit the override to use the default maximum centerline elongation rate ln(2)/1800 s⁻¹. That is a nominal 30-minute doubling timescale at saturating nutrient, not a guarantee of 30-minute division cycles: nutrient limitation, finite end caps, and division thresholds also matter. The viewer clock is in seconds.

For a headless run at the default growth rate:

```console
uv run --no-sync microsimulator run --model examples/mother_machine.py --backend cpu --seed 42 --steps 100 --dt 0.025 --output results/mother-machine.json
```

This short run checks startup and transport; several biological generations require a much longer simulated interval. `--parameter growth_rate=0` disables elongation, and `--parameter cells=false` runs the same device without cells.

## Growth, lineage, and nutrient

The local nutrient concentration sets the centerline growth rate through `maximum × nutrient / (5 + nutrient)`. Nutrient uptake is proportional to actual cell volume increase with an illustrative yield of 0.5. A cell divides when its centerline length exceeds its sampled 3–3.6 µm target. Both daughters remain movable and inherit the parent's direction without artificial division jitter. Contact mechanics handles their interactions with neighboring cells and the device.

Each initial closed-end cell has a tracked pole facing the back wall. Division transfers that identity to the daughter inheriting that pole, regardless of its position. The other daughter becomes type 1. This bookkeeping does not hold either daughter in place or select a replacement mother by proximity. If a tracked lineage washes out, its recorded ID becomes null. `mother_generations` counts divisions since initialization, rather than absolute biological pole age.

Checkpoints include `mother_ids`, `mother_generations`, total `division_count`, `washed_out`, per-cell division thresholds, and random state. Together with the native lineage records, these support retention and washout checks without relying on the animation alone.

## Numerical scope and validation

The example uses steady, depth-averaged device flow followed by kinematic rod drift and contact relaxation. It does not solve cell-induced fluid displacement, cell-dependent hydraulic resistance, near-contact lubrication, or wall friction. Solute concentrations use the full non-wall voxel volume, without excluding cell-occupied volume. These approximations matter in tightly packed growth channels; geometry-based retention alone does not establish quantitative agreement with an experiment.

The transport lattice has 1 × 0.375 × 1.4 µm spacing: four sites across each growth channel and one through its depth. Native backward Euler transport uses an absolute residual tolerance of 1e-5 concentration units to accommodate float32 roundoff on this narrow geometry. Nutrient and dye diffusion coefficients are illustrative (40 and 10 µm²/s), and concentrations are relative. The dye inlet is a prescribed three-row reservoir. Nutrient is supplied at both external reservoir boundaries; downstream cells are removed before reaching the outlet wall.

The regression tests in [test_mother_machine.py](../../python/tests/test_mother_machine.py) check movable-cell confinement, old-pole retention through repeated divisions, descendants entering flow, outlet loss, population accounting, nutrient-dependent growth and uptake, transport through channel mouths, and exact checkpoint restart. They repeat the retention test at two timesteps. Quantitative use still requires grid and timestep convergence, measured biological and transport parameters, and comparison with experimental trajectories and residence times.
