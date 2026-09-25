# Independent review of the Luiza feedback PRs

Three fresh sub-agents reviewed all 12 issue contributions against their original acceptance criteria and pinned branch bases. The coordinator checked combined application wiring and triaged findings. Two P2 defects were reproduced and corrected in their originating PRs. Both corrections passed separate independent re-review, and all 12 PRs have green hosted checks. These are recorded agent review results; the PRs remain open for landing in dependency order.

Each reviewer received the original issue text, exact head/base references and a bounded ownership area. They read implementation before author evidence, mapped the acceptance criteria, and distinguished checks they executed from source inspection or prior artifacts. Reviewers did not edit production code. Fixes received a separate independent re-review, and the corrected prerequisites were assembled on `marpaia/feedback-integration`. The original review snapshots remain recorded below and in the lane reports.

## Coverage and verdicts

| Review lane | Issues / PRs | Evidence and result |
| --- | --- | --- |
| Scientific behavior | #13 / #31, #14 / #35, #22 / #26 | [Acceptance map and independent probes](feedback-reviews/science.md). No material findings. Conservation ledgers, native empty-limit comparisons, float32/RNG boundaries and CPU/Metal stage diagnostics checked. |
| Persistence and lifecycle | #18 / #28, #16 / #33, #17 / #34, #24 / #37 | [Initial acceptance map](feedback-reviews/persistence.md). One shutdown defect; final closure below. Metadata integrity/migrations, replay ownership/races, checkpoint failure recovery and literal shell commands checked. |
| Viewer behavior | #20 / #27, #15 / #30, #23 / #29, #19 / #32, #21 / #36 | [Acceptance map and browser results](feedback-reviews/viewer.md). No material findings. Independent live/replay interactions, actual scalar/composite colors, geometry visibility, camera/grid state, capsule joins and resource cleanup checked. |
| Coordinator | Combined wiring and channel presentation | One channel-name ambiguity defect, corrected and [independently approved as fixed](feedback-reviews/labels-rereview.md). |

## Closed findings

### PERSIST-17-1: a stalled receiver prevented shutdown completion

At original #17 head `b50c72d78e84555fa050849721113897f22ed255`, a valid 40,000-cell scene and an actual TCP client with reading paused left approximately 6.56 MB in the server write buffer. A second authenticated, healthy client sent Stop and received both lifecycle notifications. After 12 seconds, the controller still had not completed shutdown: aiohttp was waiting for queued output to drain before its own close-handshake timeout started. Resuming the receiver allowed cleanup. The independent reviewer reproduced this on both the isolated PR and combined branch; the previous mocked stalled-send test did not fill a real transport buffer.

PR #34 now bounds initial frame delivery and the complete socket close, including drain. A connection that exceeds its close deadline is aborted to discard queued network bytes. Independent lifecycle sends and closes run concurrently. The initial request handler also follows bounded cleanup. Cancellation caused by aiohttp's shared drain waiter is handled as failed network delivery while explicit task cancellation still propagates. Current model steps and atomic checkpoint writes retain cooperative completion.

The committed regression keeps a real receiver paused, proves that more than 1 MB is buffered, verifies normal lifecycle/code-1000 closure for a healthy receiver, checks buffer/worker cleanup and immediately starts a different model on the same port. It covers both authenticated Stop and application-runner cleanup. Windows accepted the first frame below Python's transport accounting despite a paused receiver and bounded kernel buffers. The final Windows fixture sends at most eight real authenticated Frame requests while draining the healthy receiver, until it observes the same actual backlog. POSIX exercises stalled initial delivery; Windows exercises sustained broadcast backpressure on its default Proactor loop. One 15-second setup budget covers native capture and backlog observation; the separate production and shutdown assertion deadlines are unchanged.

The production repair is `a29ae2be39f34c6123b23b8639149b6636fa1db5`; final PR head is `46774910df12cc913ab3dc83812ddf8e42f9aa42`, whose later commits change only the regression fixture. The [independent re-review](feedback-reviews/shutdown-rereview.md) found no remaining production defect, passed 37 focused cases including six independent cancellation probes and a checkpoint-failure probe, and separately reviewed the final fixture changes. Both final Windows runs passed 30 tests. The [repair evidence](feedback-reviews/shutdown-fix.md) records the original reproduction and exact platform coverage.

### COORD-18-1: distinct channels received identical displayed names

At original #18 head `ae428e5dae170e3e58b54804c3b39b90c07f63bf`, raw names `["GFP", "GFP", "GFP [0]"]` displayed as `["GFP [0]", "GFP [1]", "GFP [0]"]`. Actual Chromium option labels also collapsed `["GFP", "GFP ", "GFP\t"]` into three identical names. Numerical identities remained separate, but users could not distinguish them as required.

PR #28 normalizes ASCII whitespace for display and detects collisions after ordinary duplicate suffix generation. If a supplied name imitates a generated result, every member of that group receives its final numerical index. Metadata remains verbatim in persistence. Weakly keyed caching over existing readonly metadata arrays avoids repeating the complete group calculation for each rendered label and does not retain frames.

Four new species/signal regressions failed before their respective fixes. All 43 isolated viewer tests, strict TypeScript, production build and actual Chromium selector/legend/inspector checks passed afterward, including actual option labels and numerical selection through renamed frames. Independent re-review at `98382896f91ab9ffb349567683041097e7f477e6` approved the fix after 4,412 adversarial group checks and an ownership/cache audit. See the [verdict](feedback-reviews/labels-rereview.md).

## Final source and validation

Final assembled source is `c127d827acb46edce57a76dc38f5c4c6825b153f`; publication adds only these review documents. No native C++ implementation changed in this review round.

- **534 Python tests passed, 57 skipped in 54.55 seconds** at `0e961edc4e1159cca4f025960085f6e29e280f08`. Subsequent commits change only the real-backpressure test setup; the complete affected server/shutdown suite was rerun on the final assembly: **30 passed in 10.51 seconds**. Product Python code is identical between those snapshots.
- **115 viewer tests passed**, strict TypeScript and production build passed at `5dc74c27ca019b7116a81046f66196547f15c00c`. The viewer tree is unchanged in the final assembly. The existing bundle warning remains at approximately 646 kB minified / 167 kB compressed.
- Actual Chromium combined replay passed after the label correction: fixed species/signal scales, composite preferences, native-exported topology changes, stable selection, backward/rapid/pending seeks, missing/returning data, camera/grid retention, malformed-frame recovery, keyboard navigation, narrow layout and new-dataset defaults. The independent viewer lane also executed five issue-owned browser checks and separate live-frame/replay cross-feature probes.
- Actual Chromium live-session verification passed against the corrected combined server: paused Stop, playing Stop, reconnect, checkpoint completion and terminal interruption, with clean process exits and same-port restart. This ran at `0e961ed`; subsequent changes affect tests only.
- Full Python Ruff lint passed. Strict Pyright with `--pythonpath .venv/bin/python` reported zero errors and the existing `flow_reference.py` native-extension source-stub warning. Changed Python formatting/type checks, viewer Prettier and diff whitespace checks passed.
- Both exact-head Windows shutdown workflows passed 30 tests: [run 36085262246](https://github.com/DRAGGON-Lab/MicroSimulator/actions/runs/36085262246) in 24.49 seconds and [run 36085257990](https://github.com/DRAGGON-Lab/MicroSimulator/actions/runs/36085257990) in 25.23 seconds. They include real measured backpressure, normal healthy-client closure, actual console Ctrl+C, worker cleanup and different-model same-port restart. All 12 issue PRs also have green hosted CUDA compilation checks; compilation is not NVIDIA runtime validation.

The scientific reviewer independently checked 250 mixed transport/reaction/storage cases, six native empty-limit cases, 831 float32 threshold cases, 1,000 preserved RNG draws, the founder seed/scenario matrix, and actual CPU/Metal planarity trajectories. The persistence reviewer independently checked checkpoint failure during concurrent cleanup, replay integrity and settings, exact documented shell commands, and the real stalled-client failure. These checks complement the committed regression suites; their scope and outcomes are in the linked lane reports.

For reproduction, run `.venv/bin/python -m pytest -q`, `.venv/bin/ruff check python`, and `.venv/bin/pyright --pythonpath .venv/bin/python` from the assembled checkout. Run `pnpm test`, `pnpm build`, and `pnpm format:check` from `viewer`. Follow the browser README for combined replay and live-session scripts, using distinct owned ports. Full command tests require the documented shells and available port 8765.

## Reviewed and final issue heads

| Issue / PR | Original reviewed head | Final head |
| --- | --- | --- |
| #13 / #31 | `4ccef4f3a95f02b810fa18eb76b75e0e9ff7a80e` | Unchanged |
| #14 / #35 | `63adb45c02f79f045df35539164385c0e8290af0` | Unchanged |
| #15 / #30 | `e96672c4e8e40a827e04a7ae95dcf9e3f27cca2d` | Unchanged |
| #16 / #33 | `3ccd5a6656a127159a42d034ddc7367167136779` | Unchanged |
| #17 / #34 | `b50c72d78e84555fa050849721113897f22ed255` | `46774910df12cc913ab3dc83812ddf8e42f9aa42` |
| #18 / #28 | `ae428e5dae170e3e58b54804c3b39b90c07f63bf` | `98382896f91ab9ffb349567683041097e7f477e6` |
| #19 / #32 | `9a27f8714de8515d6cb31a0b37ab5d1ce3bfa0c2` | Unchanged |
| #20 / #27 | `2a6bdc0340994afb1c6fe8b6d9f92ecc11257c18` | Unchanged |
| #21 / #36 | `178143550dc11f63689f8eba5b412146224b7aea` | Unchanged |
| #22 / #26 | `2d30fc7fde33abe06ddf0b05910c09d7d594e3f7` | Unchanged |
| #23 / #29 | `c1cb4d6fd20d0f0ca5f5bb59f3164413e4ca1f97` | Unchanged |
| #24 / #37 | `5e1fdecb404feb89f82e75cea698f870fb70e37d` | Unchanged |

The other ten issue branches required no review correction. Frozen bases remain `base-16` at `d65f848334a18a73c4405b5563e94b435b23795b`, `base-19` at `b0f98ee9187514d92adc562f383796ea6f41db0f`, and `base-24` at `f2c17902089d3987a247a0131ada74c0402174b3`. Their dependent PRs retain the original review boundary. The final combined branch includes the newer #17/#18 fixes; landing must also include those newer prerequisite commits before dependent contributions are retargeted.

## Landing and evidence boundaries

All 12 PRs remain open against their intended bases. Master remains `b69193b9db034b4ccd484920de3068cc11c3ea2d`. Follow the [existing landing order](feedback-campaign.md#landing-order): independent prerequisites first, then #14/#15/#16/#23/#24, then #19 after its prerequisites. Rebase or transplant only the issue-specific commits when a prerequisite has been squash-merged. Review the resulting diffs and rerun the combined checks after any landing conflict resolution. The integration branch and frozen bases are validation artifacts, not merge targets.

Issue #13 delivers its numerical design and executable reference; production occupancy remains outside that issue. Issue #14 delivers the dimensionality diagnosis and tutorial audit; strict 2D mechanics is a specified follow-up. CPU and available Apple Metal were executed. No NVIDIA runtime result is claimed; hosted CUDA checks compile only. Legacy-source tests retain their documented unavailable-input skips. Chromium uses SwiftShader, so measured timings are not physical GPU performance claims. Windows controlled console tests do not claim coverage of every interactive terminal/keyboard configuration.

The lane reports preserve exact commands, heads, acceptance maps and evidence limits. Local probe sources/logs/screenshots are retained under `/private/tmp/microsimulator-swarm/review-round-1`; those paths are not repository dependencies. Principal failure regressions are committed in the originating issue branches, and the [browser README](../../viewer/browser/README.md) documents the reproducible UI workflows. The [original campaign report](feedback-campaign.md) retains its earlier validation snapshot; this report records the subsequent review round.
