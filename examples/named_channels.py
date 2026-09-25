"""Two intracellular reporters and two extracellular signals with explicit labels."""

from microsimulator import (
    CellInit,
    ChannelMetadata,
    CheckpointBundle,
    GridShape,
    ModelContext,
    NativeController,
    SignalGridSpec,
    Vec3,
)


def build(context: ModelContext) -> NativeController:
    simulation = context.simulation(species_count=2)
    grid = SignalGridSpec()
    grid.signal_count = 2
    shape = GridShape()
    shape.x, shape.y, shape.z = 4, 4, 1
    grid.shape = shape
    grid.spacing = Vec3(1.0, 1.0, 1.0)
    grid.diffusion = [0.1, 0.2]
    grid.advection = [Vec3(), Vec3()]
    simulation.configure_signal_grid(grid, [0.25] * 16 + [0.75] * 16)
    cell = CellInit()
    cell.length = 2.0
    cell.species = [0.25, 0.75]
    simulation.add_cell(cell)
    return NativeController(
        simulation,
        model_id="named-channels",
        model_version=1,
        rng=context.rng,
        channel_metadata=ChannelMetadata(
            species=("Green reporter", "Red reporter"),
            signals=("Nutrient", "Extracellular cue"),
        ),
    )


def resume(context: ModelContext, checkpoint: CheckpointBundle) -> NativeController:
    return NativeController.from_checkpoint(checkpoint, model_id="named-channels", model_version=1)
