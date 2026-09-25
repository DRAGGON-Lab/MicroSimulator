# Tutorial commands by backend and shell

Run these commands from the repository root. They use the same Python models on every backend. `--backend` selects `cpu`, `metal`, or `cuda`; `--device-index` selects the zero-based device within that backend. Model paths, JSON parameters, seeds, and timesteps do not acquire backend-specific syntax.

## Prepare and discover devices

Install Python 3.12, `uv`, CMake, Ninja, and a C++23 compiler. Windows CPU builds need the Visual Studio C++ build tools and Windows SDK available to the build process; the tested Windows route uses PowerShell 7 on `windows-2025`. Metal needs macOS and an accessible Apple GPU. CUDA additionally needs a CUDA-enabled build, an NVIDIA GPU, and a compatible toolkit/driver; see the [Metal](../../environments/metal/README.md) and [CUDA](../../environments/cuda/README.md) environment guides.

```console
uv sync --locked --group dev
```

The commands below use `uv run --no-sync` after this installation so running a tutorial does not change the environment. Run `uv sync --locked --group dev` again after changing dependencies or checking out another version. No virtual-environment activation is required.

<!-- tutorial-command: devices -->
```console
uv run --no-sync microsimulator devices
```

<!-- tutorial-command: devices-json -->
```console
uv run --no-sync microsimulator devices --json
```

An entry such as `metal:0 ...` identifies backend `metal`, device index `0`. CPU also uses index `0`. For a second enumerated GPU, change only `--device-index 0` to `--device-index 1`. `devices` reports what this installation can construct; it does not certify scientific conformance. CPU is the reference backend, Metal is the supported Apple backend, and CUDA remains under development pending the required NVIDIA runtime and application gates. A CUDA compile check alone does not establish runtime support. See the [validation policy](../development/validation.md).

Requesting an unavailable backend/device fails with an error such as `backend cuda device 0 is unavailable (0 device(s) found)` and a nonzero exit status. There is no automatic CPU fallback. Choose an available backend explicitly or install/configure the requested backend. Do not change a backend flag merely to label a CPU result as a GPU result.

## Choose a shell

Single-line `console` commands on this page work in POSIX `sh`, Bash, Zsh, and PowerShell 7.3 or newer configured as below. Multiline `sh` blocks use a trailing backslash; multiline `powershell` blocks use a trailing backtick. The continuation character must be the last character on the line, with no trailing spaces or comment. Do not paste the backslashes from a `sh` block into PowerShell; use the PowerShell block or join the command onto one line.

For PowerShell, open `pwsh` and set its native argument-passing mode once in that session:

<!-- tutorial-command: powershell-mode -->
```powershell
$PSNativeCommandArgumentPassing = 'Standard'
```

The tested Windows route is PowerShell 7.3+ (`pwsh`), not Windows PowerShell 5.1 (`powershell.exe`) or Command Prompt (`cmd.exe`). Their native JSON quoting differs; install/use `pwsh` for these examples. A shell passing the arguments correctly does not establish GPU availability on that operating system. CUDA's current hardware conformance scripts target Linux.

## Run the same trap on CPU, Metal, or CUDA

Choose one available backend. These commands all run [`examples/microfluidic_trap.py`](../../examples/microfluidic_trap.py), with seed `42`, no model parameter overrides, 100 steps, `dt=0.02`, device index `0`, and checkpoints every 20 steps. Only the backend and output filename differ. Keeping separate output names prevents one backend's result from replacing another's.

CPU:

<!-- tutorial-command: trap-cpu -->
```console
uv run --no-sync microsimulator run --model examples/microfluidic_trap.py --backend cpu --device-index 0 --seed 42 --steps 100 --dt 0.02 --checkpoint-every 20 --output "results/tutorial runs/trap-cpu.json"
```

Metal:

<!-- tutorial-command: trap-metal -->
```console
uv run --no-sync microsimulator run --model examples/microfluidic_trap.py --backend metal --device-index 0 --seed 42 --steps 100 --dt 0.02 --checkpoint-every 20 --output "results/tutorial runs/trap-metal.json"
```

CUDA:

<!-- tutorial-command: trap-cuda -->
```console
uv run --no-sync microsimulator run --model examples/microfluidic_trap.py --backend cuda --device-index 0 --seed 42 --steps 100 --dt 0.02 --checkpoint-every 20 --output "results/tutorial runs/trap-cuda.json"
```

The quoted output paths contain spaces. Parent directories are created automatically. Final and periodic outputs must be new: choose another output name on a second run. `--overwrite` is an explicit replacement option, not a prerequisite for running a tutorial. Identical seeds define the same experiment; floating-point results across backends must be compared using the project's numerical tolerances, not an assumption of byte-identical checkpoints.

## JSON parameters and model paths with spaces

`--parameter` takes one `NAME=JSON` argument. For a JSON string, the shell must preserve the inner double quotes: `'scenario="basics"'` reaches Python as `scenario="basics"`. A bare `scenario=basics` is invalid JSON. Numeric and Boolean examples are `--parameter copies_per_cell=6` and `--parameter enabled=true`, when the selected model defines those parameters. Repeat `--parameter` for additional names.

This example copies the self-contained growth model to a path containing spaces, then runs its `basics` scenario. Choose the block for your shell. Both blocks describe the same run and output, so run only one.

POSIX `sh`, Bash, or Zsh:

<!-- tutorial-command: copy-posix -->
```sh
mkdir -p "results/tutorial models"
cp examples/tutorials/biophysics.py "results/tutorial models/biophysics.py"
```

<!-- tutorial-command: basics-posix -->
```sh
uv run --no-sync microsimulator run \
  --model "results/tutorial models/biophysics.py" \
  --parameter 'scenario="basics"' \
  --backend cpu --device-index 0 --seed 42 \
  --steps 10 --dt 0.02 \
  --output "results/tutorial runs/basics.json"
```

PowerShell 7.3+ with `Standard` argument passing:

<!-- tutorial-command: copy-powershell -->
```powershell
New-Item -ItemType Directory -Force "results/tutorial models" | Out-Null
Copy-Item examples/tutorials/biophysics.py "results/tutorial models/biophysics.py"
```

<!-- tutorial-command: basics-powershell -->
```powershell
uv run --no-sync microsimulator run `
  --model "results/tutorial models/biophysics.py" `
  --parameter 'scenario="basics"' `
  --backend cpu --device-index 0 --seed 42 `
  --steps 10 --dt 0.02 `
  --output "results/tutorial runs/basics.json"
```

To run this scenario on a GPU, replace `--backend cpu` with an enumerated `metal` or `cuda` backend and choose a new output filename. Keep the model, `scenario`, seed, step count, and timestep unchanged for a comparison. Quotes also protect an absolute repository or model path containing spaces; on Windows, forward slashes in these Python CLI paths are accepted.

## Resume saved parameters and state

Resume the preceding scenario for ten additional steps:

<!-- tutorial-command: resume-basics -->
```console
uv run --no-sync microsimulator run --model "results/tutorial models/biophysics.py" --resume "results/tutorial runs/basics.json" --backend cpu --device-index 0 --steps 10 --dt 0.02 --output "results/tutorial runs/basics-resumed.json"
```

For native `--model ... --resume ...`, the CLI obtains the seed and parameters from the checkpoint, restores controller/random/native state, and checks the model file's SHA-256 before executing it. Do not pass `--parameter` on resume: the CLI rejects it. `--seed` is a construction option and does not override the saved resume seed; omit it here. The model file may move, but its bytes must match the saved digest. Keep the original model source when updating a checkout. `--steps` is an additional step count, and `--dt` remains an explicit choice; retain the original timestep when continuing the same experiment. An available backend/device can be selected explicitly for resume; that does not guarantee bitwise equality across backends.

To continue the CPU trap from above:

<!-- tutorial-command: resume-trap -->
```console
uv run --no-sync microsimulator run --model examples/microfluidic_trap.py --resume "results/tutorial runs/trap-cpu.json" --backend cpu --device-index 0 --steps 100 --dt 0.02 --output "results/tutorial runs/trap-cpu-resumed.json"
```

For a Metal or CUDA trap checkpoint, change both `--resume` and `--output` to that run's filenames and select the intended available backend. Controller-backed checkpoints need their original `--model`; a checkpoint is data, not a substitute for model behavior.

## Live view, Stop, and restart

Build the browser assets once from the repository root:

```console
pnpm --dir viewer install
pnpm --dir viewer build
```

Start a CPU trap session. Open the tokenized loopback URL printed in the terminal, or append `--open` to open it automatically. The explicit `--viewer-dist` resolves from the current directory.

<!-- tutorial-command: live-trap -->
```console
uv run --no-sync microsimulator view --model examples/microfluidic_trap.py --backend cpu --device-index 0 --seed 42 --dt 0.02 --viewer-dist viewer/dist --checkpoint-output "results/tutorial runs/live-trap.json"
```

To use a GPU, the only simulation selection changes are `--backend metal` or `--backend cuda` and, if needed, `--device-index`. `--port` defaults to `8765`; choose another free port explicitly if it is occupied. Use the new tokenized URL after each launch.

| Control or action | Effect |
| --- | --- |
| Pause | Stops continuous playback; keeps the model process and current state available. |
| Reset | Rebuilds this session's original model. For a resumed session, reloads its starting checkpoint. Does not select a different model. |
| Close the browser | Disconnects and pauses the session; the server remains available for reconnection. |
| Checkpoint | Saves to `--checkpoint-output`; use this before Stop when restartable state is needed. |
| Stop session or one terminal Ctrl+C | Finishes the current individual step or checkpoint write, drains the worker, releases the port, and returns to the prompt. No automatic checkpoint is written. |

Wait for **Stopped** and the terminal prompt before starting another command. A long `--frame-steps` batch is interrupted between individual steps; Stop does not forcibly interrupt a single model callback or solver. After stopping the trap, launch another model on the same default port:

<!-- tutorial-command: live-basics -->
```console
uv run --no-sync microsimulator view --model examples/tutorials/biophysics.py --parameter 'scenario="basics"' --backend cpu --device-index 0 --seed 42 --dt 0.02 --viewer-dist viewer/dist
```

After stopping that session, the saved headless scenario can also be opened live:

<!-- tutorial-command: live-resume -->
```console
uv run --no-sync microsimulator view --model "results/tutorial models/biophysics.py" --resume "results/tutorial runs/basics.json" --backend cpu --device-index 0 --dt 0.02 --viewer-dist viewer/dist
```

The [viewer guide](../../viewer/README.md#stop-one-model-and-start-another) describes shutdown checks, including the distinction between a generated Windows console Ctrl+C event and a human keyboard press in a particular terminal application.

## Executed-command coverage

[`test_tutorial_commands.py`](../../python/tests/test_tutorial_commands.py) executes the marked command blocks above through the actual shell, in a temporary repository path containing spaces. It checks device discovery, each available backend's trap checkpoint, unavailable-backend errors, JSON string parameters, quoted model/output/resume paths, saved provenance, continuation syntax, resume equivalence, and real live-session Stop/restart on the same port. Live CLI tests use minimal static assets; browser control behavior is covered separately by the [viewer shutdown tests](../../python/tests/test_viewer_shutdown.py).

The [Windows CLI and live-session workflow](../../.github/workflows/live-shutdown-windows.yml) runs PowerShell commands against a freshly built native CPU extension and uploads shell/platform/backend/command evidence. The [command verification record](../development/tutorial-command-validation.md) states which platforms, shells, and hardware were actually exercised. Missing GPU hardware is recorded as unavailable, never counted as a passing GPU execution. These smoke checks establish command behavior, not complete backend conformance.
