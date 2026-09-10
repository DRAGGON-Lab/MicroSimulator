# Run manifest v1

A run manifest is strict, data-only JSON that names reproducible batch jobs. Parsing it does not import or execute model code. Execution remains explicit: one `microsimulator run-manifest --job` invocation runs exactly one job, so a local shell, CI system, or cluster scheduler retains ownership of parallelism and retries.

## Document shape

```json
{
  "format": "microsimulator-run-manifest",
  "version": 1,
  "jobs": [
    {
      "id": "gamma-0.10-replicate-001",
      "model": {
        "path": "../models/colony.py",
        "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
      },
      "backend": "metal",
      "device_index": 0,
      "seed": 1,
      "parameters": { "gamma": 0.1 },
      "stopping": {
        "maximum_steps": 100000,
        "dt": 0.05,
        "cell_count": 10000
      },
      "checkpoint_every": 1000,
      "output": "../results/gamma-0.10-replicate-001.json"
    }
  ]
}
```

Every listed field is required. Unknown and duplicate keys, non-finite JSON numbers, unknown backends, malformed SHA-256 values, duplicate job IDs, and invalid numeric bounds are rejected. Job IDs contain 1-128 ASCII letters, digits, periods, underscores, or hyphens and must begin with a letter or digit. Seeds, maximum steps, and optional cell-count thresholds use unsigned 64-bit bounds; device indices use unsigned 32-bit bounds. `cell_count` is either null or positive. `dt` is finite and non-negative.

Model and output paths are resolved relative to the manifest directory unless already absolute. Final outputs must be unique. The validator also rejects collisions between any jobs' potential periodic and final checkpoint names over their declared maximum step ranges. Existing filesystem collisions are checked immediately before a selected job runs.

The model digest is over the exact source bytes. It is checked before those bytes are compiled or executed. The resulting checkpoint provenance includes the job ID plus the absolute manifest path and exact manifest-file SHA-256. The model remains explicitly trusted Python once its declared digest matches; the manifest is not a sandbox.

## Execution

```console
uv run microsimulator run-manifest experiments/gamma.runs.json \
  --job gamma-0.10-replicate-001
```

Use `--quiet` or `--progress-every` as with `microsimulator run`. Existing outputs require the explicit `--overwrite` execution flag. The manifest intentionally contains no worker count, queue, cloud, or cluster configuration.
