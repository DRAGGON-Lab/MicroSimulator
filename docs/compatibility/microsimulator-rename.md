# Updating to MicroSimulator

CellModeller2 is now MicroSimulator. The repository is [DRAGGON-Lab/MicroSimulator](https://github.com/DRAGGON-Lab/MicroSimulator).

For an existing checkout, update the remote and reinstall the package:

```console
git remote set-url origin git@github.com:DRAGGON-Lab/MicroSimulator.git
uv sync --locked --group dev
uv run microsimulator devices
```

Change Python imports from `cellmodeller2` to `microsimulator`, including submodule imports. Replace the `cm` command with `microsimulator`; `python -m microsimulator` provides the same interface. Optional dependency groups use the new package name, for example `microsimulator[viewer]` and `microsimulator[analysis]`. Rebuild the viewer with `pnpm --dir viewer build` after updating the checkout.

Checkpoints, scenes, run manifests, analysis datasets, and controller payloads use `microsimulator-*` identifiers, and readers accept only those. A file written with a `cellmodeller2-*` identifier is rejected; regenerate it by running its model. Checkpoint outputs use ordinary `.json` names, and periodic checkpoints insert the step before that suffix.

Updating a model's imports changes its source digest. Update run manifests to authenticate the revised source; checkpoint resumes that require the original model digest still enforce that requirement. The rename does not bypass model identity validation or provide an alias for the old Python package.

The native C++ namespace and include directory are `microsimulator`. CMake options, build targets, macros, and environment variables use the `MICROSIMULATOR_` and `microsimulator_` prefixes, for example `MICROSIMULATOR_ENABLE_METAL` and `MICROSIMULATOR_LEGACY_ROOT`. References to the original CellModeller project and its `CellModeller` Python modules continue to identify that separate upstream project.
