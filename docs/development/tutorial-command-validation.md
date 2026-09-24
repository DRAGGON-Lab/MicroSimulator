# Tutorial command verification

The [command guide](../tutorials/commands.md) separates shell argument handling, backend availability, and scientific backend validation. The executable examples are read directly from its marked Markdown blocks by [`test_tutorial_commands.py`](../../python/tests/test_tutorial_commands.py).

## Coverage and limits

The command tests exercise human-readable and JSON device discovery; the identical 100-step trap command on every available backend; explicit unavailable-backend failures; JSON string arguments and line continuation; repository, model, checkpoint, and output paths containing spaces; restored seed/parameters/source digest; and uninterrupted versus resumed controller/native state. They launch the documented trap, growth, and resumed live sessions in sequence on port 8765, save a checkpoint, send authenticated Stop, observe Stopped, and verify each process returns successfully before the next launch. A cleanup regression also verifies that a failed test terminates its own subprocess tree and drains inherited pipes. The CLI tests supply minimal static assets and do not claim browser rendering coverage.

PowerShell tests set `$PSNativeCommandArgumentPassing = 'Standard'` exactly as documented. Windows uses `pwsh` 7.3+; Windows PowerShell 5.1 and `cmd.exe` are not covered. POSIX examples are executed in `sh`, Bash, and Zsh when those shells are installed. A missing shell is explicitly skipped. Reports include the exact script text, exit status, stdout/stderr, platform, Python version, shell version, and enumerated devices.

The [Windows CLI and live-session workflow](../../.github/workflows/live-shutdown-windows.yml) builds a native CPU extension on `windows-2025`, executes these command tests and the existing live-session tests, and uploads reports. Its GPU backends are deliberately disabled. The shutdown tests additionally generate an actual Windows `CTRL_C_EVENT` in an isolated console; they do not simulate a human keyboard press in every terminal application. The viewer guide retains a [manual terminal procedure](../../viewer/README.md#stop-one-model-and-start-another).

## Execution record

The local command run on 2026-09-24 passed all ten tests with Python 3.12.8, Bash 3.2.57, Zsh 5.9, and PowerShell 7.6.4. The associated pull request links the exact source SHA and hosted workflow run; reports are retained as CI artifacts.

| Platform and shell | Backend execution | Command coverage |
| --- | --- | --- |
| macOS, POSIX `sh`, Bash, Zsh | Native CPU and Apple M4 Max Metal | Passed: discovery, 100-step CPU/Metal trap, JSON/space paths, headless CPU resume, live CPU checkpoint/Stop/restart |
| macOS, PowerShell 7.6.4 with Standard arguments | Native CPU and Apple M4 Max Metal | Passed: same commands, PowerShell backtick continuation and string quoting |
| Windows Server 2025, PowerShell 7 with Standard arguments | Native CPU | Hosted command verification pending; see the pull request workflow result |
| NVIDIA CUDA hardware | Not available in the command-verification campaign | Unavailable-backend error checked; no CUDA runtime or numerical claim |

The test runner reports actual Metal availability for each macOS run. This run executed Metal headless trap commands; its JSON scenario, resume, and live-session examples selected CPU. A skipped/unavailable Metal result does not count as GPU execution. Backend support still requires the independent [native and application conformance gates](validation.md), even when a tutorial smoke command succeeds.

## Reproduce

After installing the development environment, run:

```console
uv run --no-sync python -m pytest python/tests/test_tutorial_commands.py -v
```

Set `MICROSIMULATOR_COMMAND_REPORT` to a new output directory to retain per-shell JSON reports; the Windows workflow supplies this environment variable. Run from an environment with loopback networking enabled and port 8765 free. The tests use their own temporary working directories and never overwrite tutorial outputs in the source checkout.
