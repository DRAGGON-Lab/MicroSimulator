# Migrating from the CellModeller viewer

## CellModeller components

The CellModeller 4 desktop entry point and rendering layer comprise:

- `Scripts/CellModellerGUI.py`;
- `CellModeller/GUI/PyGLCMViewer.py`;
- `CellModeller/GUI/PyGLWidget.py`;
- `CellModeller/GUI/Renderers.py`;
- the renderer hooks in `CellModeller/Simulator.py`; and
- model calls to `Simulator.addRenderer`.

The goal is to preserve useful interactive behavior without preserving the old ownership graph or graphics implementation.

## Simulation and presentation ownership

`PyGLCMViewer` does more than display state. It enumerates OpenCL platforms and devices, constructs and destroys `Simulator` instances, imports and reloads model modules, advances the simulation from a zero-interval Qt timer, toggles pickle output, and restores executable pickle payloads. This makes the UI an owner of compute policy and model execution.

Models also import OpenGL renderer classes and register renderer instances on the simulator. Those renderers retain the simulator and read mutable `cellStates`, signaling integrator arrays, meshes, and arbitrary Python cell attributes directly. The simulator therefore depends on presentation objects, while presentation code depends on nearly every mutable simulation detail.

The graphics layer uses the compatibility OpenGL API: `QGLWidget`, display lists, matrix stacks, GLU quadrics, immediate-mode primitives, and the OpenGL name stack for picking. Rendering and stepping both run on the GUI thread.

## User-visible behavior

The maintained rod-cell workflows use a smaller set of capabilities:

- load a model or saved state;
- run, pause, single-step, and reset;
- save simulation state;
- orbit, pan, zoom, and frame the colony;
- draw 2D or 3D spherocylinders with per-cell color;
- draw a selected-cell outline and inspect its properties; and
- display scalar signaling-grid slices.

The renderer file also contains sphere, plant, periodic-image, static-mesh, and experimental mesh renderers. They are listed below because they require different simulation state and are not supported by the rod-and-grid viewer.

## Renderer support

The machine-readable contract is `compatibility/legacy-renderers-v1.json`. It authenticates `Renderers.py` at CellModeller commit `4896f543c6250f053eea2312e628cc3a96bf7408` and classifies all ten renderer classes. Repository-wide call-site search found that all 25 bundled examples select `GLBacteriumRenderer`, four also select `GLGridRenderer`, and none selects another renderer class.

| Family | CellModeller behavior | MicroSimulator support |
| --- | --- | --- |
| rod cells (`GLBacteriumRenderer`, `GLCelBacteriumRenderer`, `GL2DBacteriumRenderer`) | rods, stable cell IDs, selection, arbitrary color fields | Supported through scene-v1 rods, typed color mappings, and the independent viewer. |
| signal grid (`GLGridRenderer`) | direct reads from the signaling integrator | Supported through scene-v1 grids and viewer-owned channel and slice selection. |
| sphere cells (`GLSphereRenderer`) | a distinct spherical cell morphology; no bundled call site; picking references an undefined radius | Not supported as a cell morphology. Typed inside/outside sphere constraints remain available for rod cells. |
| plant cells (`GLPlantRenderer`, `GLPlantSignalRenderer`) | polygon `nodep`/`wallp` geometry and arbitrary signal attributes; no bundled call site or plant engine | Not supported; MicroSimulator has no corresponding plant-cell state model. |
| periodic cell images (`GLBacteriumRendererWithPeriodicImages`) | four visual copies offset by mutable collision-grid bounds; no declared periodic cell topology or bundled call site | Not supported for cells. Typed periodic signal boundaries remain available. |
| dynamic collision mesh (`GLWillsMeshRenderer`) | lines over mutable `CLBacterium` broad-phase bins; no bundled call site | Not part of scene data; it visualizes an implementation-specific debugging structure. |
| static triangle mesh (`GLStaticMeshRenderer`) | external mesh/regulator object graphs; no bundled call site or checkpoint representation | Not supported because there is no corresponding checkpoint or scene representation. |

Spherical-cell, plant-tissue, periodic-cell-domain, and mesh simulations require their own typed engine state, checkpoint representation, and scene semantics; they are not implied by the old renderer classes alone.

## Migration considerations

The old viewer cannot be used as a compatibility oracle for several reasons:

- it loads arbitrary pickle objects and may execute source stored inside them;
- model files choose concrete renderer classes and mutate presentation state;
- selection assumes a cell ID fits the legacy OpenGL integer name mechanism;
- render refresh depends on mutable `stepNum` and renderer-local caches;
- property inspection walks arbitrary `CellState.__dict__` values; and
- the signaling renderer reaches into integrator-owned arrays rather than a declared observation interface.

Visual pixel equality is neither defined nor scientifically meaningful. The compatibility target is equality of the typed scene data presented to the viewer, followed by behavioral tests of selection and controls.

## MicroSimulator design

MicroSimulator does not attach renderers to `Simulation`, expose backend memory to a UI, or let the UI choose CUDA or Metal devices by reaching into a runtime. Instead, the engine produces immutable scene frames in stable cell-ID order. A separate controller owns run policy and produces those frames after complete simulation steps.

The initial scene vocabulary contains rods and an optional scalar signal grid. Cell color is a presentation mapping over declared fields such as cell type, species level, growth rate, and fixed state; it is not a mutable engine field. Stable 64-bit cell IDs cross the JSON boundary as decimal strings so browser consumers cannot lose precision.

Legacy pickle viewing goes through the existing trusted, explicit one-way import command. The new viewer only accepts the non-executable MicroSimulator checkpoint and scene formats.
