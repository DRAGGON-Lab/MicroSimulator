# Species and signal labels

Numerical channel indices determine rate-plan inputs, storage order, and viewer preferences. Labels only describe those indices. They are never inferred from Python variable names.

## Native models

Pass immutable `ChannelMetadata` to `NativeController` after configuring the simulation's species count and signal grid:

```python
from microsimulator import ChannelMetadata, NativeController

return NativeController(
    simulation,
    model_id="my-model",
    model_version=1,
    rng=context.rng,
    channel_metadata=ChannelMetadata(
        species=("Green reporter", "Red reporter"),
        signals=("Nutrient", "Extracellular cue"),
    ),
)
```

Each supplied tuple must contain exactly one entry per corresponding numerical channel. Use `None` for an unnamed entry, or omit a whole group to leave all its channels unnamed. The constructor validates counts immediately; the runner checks again before the first step, and exporters validate against the current state. Empty strings and whitespace-only strings display the same fallback as `None`, while retaining their exact supplied text in files. Labels must be Unicode scalar strings; unpaired surrogates are rejected. Presentation collapses and trims ASCII whitespace like an HTML option label before checking for duplicates; serialized metadata retains the original text. Duplicate names are valid and display their numerical indices for disambiguation. If a supplied name imitates one of those generated labels (for example, `GFP`, `GFP`, and `GFP [0]`), all labels in that species or signal group receive their indices so every displayed name remains distinct. Labels, including HTML-like strings, render as text.

The complete [named-channel model](../../examples/named_channels.py) declares two species and two signals:

```sh
uv run microsimulator run --model examples/named_channels.py --backend cpu --seed 17 --steps 2 --dt 0.01 --output named.json
uv run microsimulator view --model examples/named_channels.py --resume named.json --backend cpu --dt 0.01
```

`NativeController.from_checkpoint` restores the persisted labels automatically. A custom controller may optionally expose a typed `channel_metadata: ChannelMetadata` attribute; this is not a required member of `SimulationController`. Its `resume` function must restore `checkpoint.channel_metadata`. The native model runner rejects a resumed model that changes the saved labels. Unnamed native models and legacy adapters need no changes.

## Data-only export and low-level APIs

Labels are stored in checkpoint version 9 independently of the controller payload, so recovering them never requires running model code. Scene version 3 carries the same ordered arrays. Use the bundle's labels explicitly when exporting or saving native state directly:

```python
from microsimulator import capture_scene, load_checkpoint_bundle, save_checkpoint, save_scene

bundle = load_checkpoint_bundle("named.json")
frame = capture_scene(bundle.simulation, channel_metadata=bundle.channel_metadata)
save_scene(frame, "named.scene.json")
save_checkpoint(
    bundle.simulation,
    "copy.json",
    provenance=bundle.provenance,
    controller=bundle.controller,
    channel_metadata=bundle.channel_metadata,
)
```

`capture_scene` and `save_checkpoint` also accept `channel_metadata` for a bare `Simulation` when no controller is needed. A bare native simulation does not own Python presentation metadata. Therefore `load_checkpoint` refuses a file with non-null labels, just as it refuses a non-null controller payload: use `load_checkpoint_bundle` to avoid silently losing labels. Such a named, bare-native checkpoint is exported or continued through the bundle API; `run --resume` without a model retains its existing unnamed-only contract. Standard named models use the controller resume command above.

Scenes support at most 4096 species and 4096 signals per frame, independently, including unnamed channels and empty colonies. `MAX_SCENE_CHANNELS` exposes this presentation budget; oversized export fails with `SceneError` before copying native state or expanding labels. Native simulation and checkpoint counts retain their existing semantics. When loading checkpoints predating v9, `CheckpointBundle.channel_metadata` keeps both unspecified groups as `None`, without allocating labels from native counts. Bounded scene export supplies the null-filled arrays; callers that explicitly need resolved metadata can use `.resolved(species_count, signal_count)`.

Within a live dataset, channel choices stay keyed by kind and index. Renaming a channel does not change its concentration or select another channel. Frames, reset, and replay retain the same labels through the shared scene parser. Opening another dataset establishes a new presentation identity.

## SBML labels

`SBMLRateModel.channel_metadata` explicitly maps nonempty species names to labels and falls back to SBML species identifiers when names are missing. It preserves the imported species order:

```python
from microsimulator import CellInit, NativeController, load_sbml

rates = load_sbml("model.xml")
simulation = context.simulation(species_count=rates.species_count)
simulation.set_species_rate_plan(rates.rate_plan)
cell = CellInit()
cell.species = list(rates.initial_levels)
simulation.add_cell(cell)
return NativeController(
    simulation,
    model_id="my-sbml-model",
    model_version=1,
    rng=context.rng,
    channel_metadata=rates.channel_metadata,
)
```

SBML species identifiers remain authoritative for compilation. If the simulation also has extracellular signals, declare those explicitly with `ChannelMetadata(species=rates.channel_metadata.species, signals=(...))`. The SBML importer does not infer extracellular signal identities.
