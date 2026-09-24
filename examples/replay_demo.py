"""Short deterministic growth/division/removal recording for the replay tutorial."""

from microsimulator import (
    CellInit,
    ChannelMetadata,
    CheckpointBundle,
    ControllerStep,
    DivisionRequest,
    ModelContext,
    NativeController,
    StepPlan,
)


def regulate(step: ControllerStep) -> StepPlan:
    if step.completed_steps == 1:
        return StepPlan(divisions=(DivisionRequest(step.cells[0].id),))
    if step.completed_steps == 2:
        return StepPlan(removals=(step.cells[0].id,))
    return StepPlan()


def build(context: ModelContext) -> NativeController:
    simulation = context.simulation(species_count=1)
    cell = CellInit()
    cell.length = 4.0
    cell.growth_rate = 0.5
    cell.species = [0.25]
    simulation.add_cell(cell)
    return NativeController(
        simulation,
        model_id="replay-demo",
        model_version=1,
        rng=context.rng,
        regulate=regulate,
        channel_metadata=ChannelMetadata(species=("Reporter",)),
    )


def resume(context: ModelContext, checkpoint: CheckpointBundle) -> NativeController:
    return NativeController.from_checkpoint(
        checkpoint, model_id="replay-demo", model_version=1, regulate=regulate
    )
