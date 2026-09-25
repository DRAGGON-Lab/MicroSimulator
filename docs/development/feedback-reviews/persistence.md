# Independent persistence review, round 1

This is an archived review/evidence record. The [round summary](../feedback-review.md) records final closure, exact heads and successful Windows validation. Statements that a check was pending describe the time of that review, not its final status. Temporary evidence paths identify local artifacts; they are not repository dependencies.

Reviewed issues **18, 16, 17, 24 / PRs 28, 33, 34, 37** against their original issue bodies. This is an independent review of the pinned implementations; no issue-worktree source or tests were edited by this reviewer. Findings and evidence below distinguish tests actually run in this review from source inspection and archived author evidence. Initial report written 2026-09-24; shutdown remediation is owned by the coordinator-assigned author and will receive a separate independent follow-up review.

| Issue / PR | Pinned reviewed head | Diff base |
| --- | --- | --- |
| 18 / 28 | `ae428e5dae170e3e58b54804c3b39b90c07f63bf` | `b69193b9db034b4ccd484920de3068cc11c3ea2d` |
| 16 / 33 | `3ccd5a6656a127159a42d034ddc7367167136779` | merge-base `d65f848334a18a73c4405b5563e94b435b23795b` of frozen `marpaia/base-16` |
| 17 / 34 | `b50c72d78e84555fa050849721113897f22ed255` | `b69193b9db034b4ccd484920de3068cc11c3ea2d` |
| 24 / 37 | `5e1fdecb404feb89f82e75cea698f870fb70e37d` | merge-base `f2c17902089d3987a247a0131ada74c0402174b3` of frozen `marpaia/base-24` |

Combined interactions were inspected at integration `5514d99a565b1911c2a8a204e885ebd37ad6476a`. New metadata and presentation prerequisites are not attributed as authored by #16 or #24. The latest #18 scene budget / compact legacy metadata and #16 pending-seek / failed-seek Play corrections were reviewed as starting implementations, not inherited conclusions.

## Confirmed finding PERSIST-17-1 — P2 — shutdown can wait indefinitely on a stalled receiver

- Issue / PR: **17 / 34**.
- Pinned head: `b50c72d78e84555fa050849721113897f22ed255`.
- Code: `python/src/microsimulator/viewer_server.py:353`, the unbounded `await socket.close(...)` in `LiveController._close`. Same behavior reproduced at integration `5514d99`, line 362.
- Trigger: a connected client stops reading while a valid large initial scene is being transmitted; a second authenticated, same-origin client sends Stop.
- Expected: after current model/checkpoint work ends, shutdown closes sockets, completes worker/application cleanup and releases the listening port, without requiring the receiver to resume.
- Actual: the healthy client receives both `stopping` and `stopped`, but `controller.stopped` remains false. The shutdown task is blocked inside `WebSocketResponse.close -> StreamWriter.drain -> BaseProtocol._drain_helper`, before aiohttp's close-handshake timeout. `serve_live` consequently cannot finish its wait/runner cleanup. Shared terminal-interruption cleanup has the same dependency. The apparent Stopped browser state can precede an indefinitely live server.
- Reproduction: `review-round-1/persistence/stalled_socket.py`. From either issue-17 or integration: `.venv/bin/python /private/tmp/microsimulator-swarm/review-round-1/persistence/stalled_socket.py`. The probe creates 40,000 ordinary native cells (no model steps or pathological callback), opens actual aiohttp WebSockets, pauses one actual client transport, observes over 6 MB queued server output, reads the valid initial frame on a second client, and sends authenticated Stop from that healthy client. It observes shutdown for 12 seconds, then resumes/closes the stalled receiver to clean up its owned application. No fake infinite coroutine or mocked socket close is involved.

Pinned #17 output:

```text
server_write_buffer_before_stop 6563031
healthy_initial_cells 40000
stopped_with_receiver_paused False 12.001375499879941
healthy_notifications ['stopping', 'stopped']
close_task_stack ['.../issue-17/python/src/microsimulator/viewer_server.py:353:_close']
await_chain ['LiveController._close', 'WebSocketResponse.close', 'StreamWriter.drain', 'BaseProtocol._drain_helper', '_asyncio.FutureIter']
cleanup_completed True
```

Integration output:

```text
server_write_buffer_before_stop 6563598
healthy_initial_cells 40000
stopped_with_receiver_paused False 12.00127000012435
healthy_notifications ['stopping', 'stopped']
close_task_stack ['.../integration/python/src/microsimulator/viewer_server.py:362:_close']
await_chain ['LiveController._close', 'WebSocketResponse.close', 'StreamWriter.drain', 'BaseProtocol._drain_helper', '_asyncio.FutureIter']
cleanup_completed True
```

The shortened await-chain strings above omit nondeterministic object addresses; function names and observed stack locations are unchanged. An earlier direct-controller version also reproduced the same wait at four seconds. The 12-second version supersedes that initial evidence and exercises authenticated network Stop.

Why existing tests miss it: `test_stop_is_not_blocked_by_a_stalled_frame_send` replaces `send_str` with an awaiting coroutine; the broadcast timeout handles that coroutine, but no real transport output buffer is filled. Existing real clients continuously consume their frames and close handshakes, so `socket.close` drain completes. The initial `_send_frame` is also awaited by the WebSocket request before entering its receive loop; any correction must account for that handler's ownership and finalization when the transport is aborted.

Bounded correction: put a deadline around the entire close, including writer drain, abort a transport that cannot drain within it, and ensure the initial-send request and command consumer can finish/cancel without leaving a pending handler. Preserve completion of the current individual step and atomic checkpoint operation. Required regression: a real nonreading transport with buffered valid scene output, authenticated Stop from a healthy client, bounded application/process exit, absence of live worker threads/pending handlers, and immediate same-port restart. Re-run ordinary disconnect/reconnect, cooperative-step, in-progress checkpoint, repeat Stop, SIGINT, and Windows console tests where available.

The coordinator accepted this finding and assigned its correction to the #17 author. This report does not claim the correction is complete.

## Issue 18 / PR 28 acceptance map

| Original criterion | Independent verification and remaining boundary |
| --- | --- |
| Native example has two named species and two named signals, all four names appear | Read `examples/named_channels.py`; Python named-model roundtrip/export test passed. Generated a fresh five-frame recording from that native example in this report directory. Browser preference probe verifies all four selectors and named legends/inspector; coordinator executed the reviewer-authored Chromium probe successfully on integration5514d99; see browser result below. |
| No native-kernel changes | Pinned diff changes Python/TypeScript/docs/fixtures only; no C++/Metal/CUDA source or ABI changes. |
| Unnamed models / legacy adapters require no mandatory new methods | `SimulationController` protocol is unchanged; metadata is obtained with a typed optional attribute/default. Bare unnamed simulation and old checkpoint tests passed. Existing legacy acceptance is supported by the unchanged protocol and fallback; no new live legacy-model browser run was performed. |
| Count mismatch precise and before running/invalid export | Constructor, runner preflight, capture/save validation inspected. `test_channels.py` and budget suite passed, including no step/output after mismatch. 4096 inclusive scene boundary and larger/tiny unsigned claims rejected before metadata expansion. |
| Labels survive continuation and standalone export | Named periodic/final checkpoints, `NativeController.from_checkpoint`, bundle-to-scene export, and live session reset/checkpoint tests all ran and passed. `load_checkpoint` deliberately rejects nonnull labels to prevent loss; bundle API is documented. |
| Python / TypeScript new schema and migrations agree | Python channel/checkpoint/budget suites and TypeScript channels/scene tests passed. Shared v3 fixture, v2 original payload verification, strict fields, count/type errors, tampering, Unicode scalar validation checked. Checkpoint versions 1–8 retain compact absent metadata until needed; their original integrity rules run before metadata handling. |
| Existing v2 scenes fallback by index | Original v2 fixtures and migration tests passed in both reader suites. No v2 fields are silently added before original digest verification. |
| Duplicate, Unicode, missing, HTML-like labels; text rendering | Unit fixtures cover all four and DOM writes use textContent. The coordinator separately confirmed a duplicate-disambiguation collision and HTML option whitespace collision at the pinned head, and owns their #18 correction. These are cross-referenced rather than duplicated under a persistence finding ID. |
| Selections/settings use channel identity | DOM option values and stored preferences use numerical indices, not label text; color tests and source trace verified. Combined composite controls consume channelLabel and numerical-index state. Fresh reviewer-authored named-replay Chromium settings check passed on integration5514d99, executed by the coordinator. |
| Authoring API and SBML metadata example | Read channel-label API guide, example, and `SBMLRateModel.channel_metadata`; focused SBML tests passed for name/identifier fallback. |

No additional persistence/schema finding beyond coordinator-owned display disambiguation issues. Untrusted legacy counts now do not amplify absent metadata during checkpoint load. Scene limits intentionally apply to presentation, independently of native simulation counts.

## Issue 16 / PR 33 acceptance map

| Original criterion | Independent verification and remaining boundary |
| --- | --- |
| Document run -> periodic checkpoints -> export -> standalone open | Read replay guide, CLI integration and bundle schema. Ran exporter tests and independently built five periodic native named checkpoints and exported them into `named-replay`. Coordinator-executed reviewer-authored Chromium probe successfully opened the exported bundle; final result below. |
| Growth/division/removal both directions | Exporter test executed actual native lifecycle and verified exact cell IDs/counts/lengths, equal-time division, removal, and unchanged checkpoint bytes. Playback tests exercised backward/forward frame navigation. Existing lifecycle browser script was read; it asserts stable-ID selection across division/removal and reverse seeks, but was not rerun by this lane. |
| Ordering from manifest, not filenames | Native exporter test uses deliberately nonlexical source filenames; manifest unit test uses z-first before a-second and passed. |
| Decreasing time rejected, equal time policy | Tests passed; equal times retain separate ordinal entries. Both exporter and manifest parser reject decreasing values, and format docs state the policy. |
| Seek no steps/source device/checkpoint mutation | Exporter imports no model source, loads native state on CPU, restores source backend only as provenance, and never steps. Test exports unavailable-CUDA provenance using CPU and checks source bytes unchanged. Browser only receives local data files. |
| Missing/malformed/integrity-invalid frame error identifies frame | Python failures include ordinal/path; TypeScript bundle tests passed missing, length, digest, time/backend mismatch. ReplayController wraps current errors with ordinal and preserves prior displayed index. Existing malformed-browser recovery assertions were inspected. |
| Rapid seeks do not show stale frames | Single drain worker/coalesced pending request, version checks and AbortSignal handling inspected. Focused tests passed stale success/failure, disposal, pending seek + Play, and retrying failed seeks. |
| Stable cell-ID selection/clear missing cell | Shared renderer stores selected ID and resolves new slot/clears absent IDs. Existing browser script asserts this for topology-changing recording. Fresh named probe checks retained ID across same-topology frames. No new topology-browser run by this lane. |
| Signal appearance/disappearance and shape controls | Presentation code clamps only effective current indices, retains stored preferences, and hides absent grids. Focused presentation-state tests passed; native exporter grid change test passed. Existing browser script checks absence, shrink, reappearance; not rerun by this lane. |
| Bounded decoded memory | Cache tests passed frame limit, weighted-byte eviction, oversized frame bypass, LRU behavior and dispose clearing. There is one active load and one coalesced pending ordinal. Docs correctly exclude renderer/current frame/manifest/one active parse from the 64 MiB accounting budget. No process-heap benchmark was claimed or run. |
| Fixed scales/channel/device/reference-grid settings stable | Combined source uses one dataset identity and index-based preferences. Fresh native named replay probe covers species/signal fixed ranges, channels, device/signal visibility, camera/grid, composite order/enablement; probe passed through coordinator execution; final result below. |
| Exporter/playback/browser workflow evidence | This review ran seven exporter tests and 31 TypeScript replay/manifest/presentation tests. Browser script and existing prior evidence were inspected; fresh reviewer-authored settings-browser probe passed through coordinator execution; result below. |

No confirmed additional replay finding. Manifest references are safe relative local file keys; no remote URL fetching. Exact file digest, internal scene digest/schema and manifest time/source-backend agreement are all checked. Existing frozen-base features were not attributed to #16.

## Issue 17 / PR 34 acceptance map

| Original criterion | Independent verification and remaining boundary |
| --- | --- |
| Browser Stop paused and playing | Closed authenticated Stop vocabulary and UI/client transition code inspected. Focused socket tests passed paused/manual batch and playing Stop. Browser tests were inspected; root owns current combined live-control browser verification. |
| Finish current slow step, not remainder of large batch | Controlled blocked model test passed for manual and active-playback 10,000-step batches; completion count exactly one. Event is checked between individual model steps. |
| Repeated Stop and concurrent disconnect idempotent | Repeated Stop, concurrent close and ten reconnect cycles ran and passed. Shared close task avoids cleanup waiting on itself. |
| Reject Play/Step/Reset/Checkpoint once stopping | Protocol reader rejects at admission; executor also checks inside serialized operation lock. All four rejection assertions passed while current step remained blocked. |
| Browser shows stopped state | Client awaits message queue before close state so asynchronous scene validation cannot overtake Stopped; seven TypeScript tests passed. PERSIST-17-1 shows Stopped can currently precede indefinite socket cleanup, so lifecycle acceptance remains blocked. |
| Terminal interrupt exits/releases port | Real POSIX SIGINT CLI-process tests ran and passed under normal readable sockets. PERSIST-17-1 applies to the shared exceptional stalled-client cleanup path. |
| Different model immediately reuses same port | Three real process launches for each termination mode passed and checked distinct cell lengths. Stalled-client variant must be added with fix. |
| Authenticated and same-origin | Shared token/Origin gate is unchanged; focused server tests exercised rejection and valid control. Real defect reproduction sends authenticated same-origin Stop. |
| Worker/active-playback/checkpoint/repeated cycles | 28 Python shutdown/server tests passed. Additional reviewer-owned `test_failure_recovery.py` passed: blocked os.replace fails during two concurrent close waiters; original checkpoint bytes remain valid, temporary file removed, close completes, worker threads gone, further Reset rejected. Its initial failed run used an overly specific expected error substring; corrected to the actual documented checkpoint error, then passed. |
| Windows forced-termination report reproduced or ruled out, terminal/launch recorded | Read Windows workflow/test launch code and actual downloaded XML plus command JSON. Archived Windows Server2025 build26100, PowerShell7.6.6, Python3.12.10 CPU run at `215e87d`: 28 shutdown tests, zero failure/skip; isolated CREATE_NEW_CONSOLE + GenerateConsoleCtrlEvent path. It establishes that controlled console/browser path, not every human terminal/keyboard configuration. No fresh Windows run performed in this lane. |
| Stop one session/start another documented | Protocol and viewer docs distinguish browser closure pause, Pause, Reset, Checkpoint, Stop, and terminal interrupt, and document same-port launches and manual Windows terminal procedure. |

PERSIST-17-1 blocks the reliable complete-cleanup criterion despite normal-path tests passing. The file/worker lifecycle itself handled an independently injected checkpoint replace failure correctly.

## Issue 24 / PR 37 acceptance map

| Original criterion | Independent verification and remaining boundary |
| --- | --- |
| Explicit equivalent introductory model per available backend | Exact marked CPU/Metal/CUDA trap snippets run by four local shells. Five selected tests passed (four shell cases + static audit). Local device enumeration confirms CPU and Apple M4 Max Metal available, CUDA unavailable. |
| Current flags and existing paths | Exact snippets execute real CLI; static flags/model-path/link audit passed. Shared command structure inspected against CLI and saved-parameter handling. |
| JSON string parameter | Both POSIX and PowerShell blocks preserve `scenario="basics"`; authenticated output-provenance assertions passed in sh/Bash/Zsh/pwsh. |
| Resume saved parameters/provenance | Exact resume commands passed; source digest and saved seed/parameters checked; resumed native/controller states equal uninterrupted run; new parameter/source changes rejected. |
| Quoting tested in claimed shells | Exact headless/continuation/JSON snippets ran locally in sh, Bash, Zsh and PowerShell7.6.4 with Standard native arguments. Downloaded Windows PowerShell7.6.6 report independently inspected. Windows PowerShell5.1/cmd are explicitly excluded for JSON examples. |
| Paths with spaces | Test fixture working directory, copied model, output and resume paths contain spaces; exact commands passed. |
| Unavailable backend explained, no fallback | CUDA branch exits2 with explicit unavailable-device error and no output; actual local run passed assertions. Windows report has expected Metal/CUDA exits2. Docs distinguish compilation, availability and scientific support. |
| Pause/Reset/closure/Stop distinctions | Read lifecycle table, getting-started and viewer changes; semantics match source for normal cleanup. #17 stalled-client blocker applies to implementation, not a new #24 defect. |
| Platform/backend execution recorded | Checked-in verification record separates macOS CPU/Metal commands, CPU-only JSON/resume/live checks, Windows CPU, and unavailable NVIDIA hardware. Actual archived reports agree with these bounds. Local review did not rerun Windows or claim NVIDIA runtime validation. |
| Links/examples consistency | Static audit passed >35 examples using current parser and existing paths/anchors. Every top-level tutorial links directly to shared backend/shell guidance. |

No confirmed #24 finding. Only selected headless and documentation tests were rerun by this lane; literal live-launch snippets were reviewed against downloaded Windows command reports and #17's freshly executed real CLI tests. This avoids unowned use of documented fixed port8765 during parallel review. Local #24 test harness uses test static assets and correctly does not claim renderer validation.

## Checks actually run

All paths below are relative to the named issue worktree except reviewer scripts.

| Worktree | Command/check | Result |
| --- | --- | --- |
| issue-18 | `.venv/bin/python -m pytest -q python/tests/test_channels.py python/tests/test_scene_channel_budget.py python/tests/test_checkpoint.py python/tests/test_sbml.py` | 52 passed, 0.37s |
| issue-18/viewer | `pnpm exec vitest run tests/channels.test.ts tests/scene.test.ts tests/color.test.ts` | 21 passed |
| issue-16 | `.venv/bin/python -m pytest -q python/tests/test_replay.py` | 7 passed |
| issue-16/viewer | `pnpm exec vitest run tests/replay.test.ts tests/replay-bundle.test.ts tests/presentation-state.test.ts` | 31 passed |
| issue-17 | `.venv/bin/python -m pytest -q python/tests/test_viewer_shutdown.py python/tests/test_viewer_server.py` | 28 passed, 1.23s |
| issue-17/viewer | `pnpm exec vitest run tests/live.test.ts` | 7 passed |
| issue-24 | `.venv/bin/python -m pytest -q python/tests/test_tutorial_commands.py -k 'headless or flags_paths'` | 5 passed, 5 deselected, 16.03s |
| issue-24 | `.venv/bin/python -m microsimulator devices --json` | CPU + Apple M4 Max Metal available; CUDA unavailable |
| issue-17 | reviewer `test_failure_recovery.py` via pytest | 1 passed, 0.08s after correcting probe error-message expectation |
| issue-17 + integration | reviewer `stalled_socket.py`, authenticated healthy-client Stop, 12s observation | PERSIST-17-1 reproduced on both pinned sources; owned sockets/server cleaned up |
| issue-16 | reviewer `make_named_replay.py` | five native named periodic checkpoints exported successfully; artifacts in this directory |
| integration5514d99 | reviewer `named_replay_browser.mjs`, launched by coordinator on4343 | Passed after probe-only camera float tolerance correction; all label/settings assertions retained |
| archived Windows evidence | parsed `tests.xml`, `tutorial-commands.xml`, platform and per-command JSON | verified reported 28 + 4 passing tests, zero skips/failures, CPU executed, GPU absence explicit |

## Browser probe status and coverage limits

`named_replay_browser.mjs` was independently authored here to check four native names and stable numerical-index display settings during forward/backward replay, including fixed ranges, composite state, device visibility, signal visibility, selected cell, camera and grid. Local direct Node launch failed in the macOS sandbox; its escalated retry remained pending and was aborted after 833.7 seconds, with no assertions run in that attempt. The coordinator ran the unchanged assertions through an approved Python browser runner on port4343. Its first run reached the camera assertion and exposed only 5e-15 OrbitControls rounding under exact equality. The agreed probe-only correction keeps the grid comparison exact and uses 1e-9 camera/target tolerance, matching the committed browser test. This is not a product finding. The corrected probe **passed** on integration `5514d99` through coordinator execution, with all original product assertions intact. `named-replay-browser.json` and `named-replay-preferences.png` contain the result and screenshot. It verified all four native labels, numerical species/signal selections, both fixed ranges, hidden device/signal preferences, stable cell ID, camera and exact reference-grid stability, plus composite order/enabled/range settings across five frames and backward seeks. This was an executed independent reviewer-authored probe; only browser process launch was delegated to the coordinator.

No fresh Windows terminal application/physical keyboard, CUDA device, WebKit/Firefox, application heap benchmark, or hardware conformance run was performed. The original Windows controlled-console evidence is an inspected prior artifact, not a new review execution. Browser lifecycle/topology assertions that this lane only read are explicitly described as such above, rather than promoted to independently executed checks.

This reviewer used no network posts, review submissions, branch changes, or production-file edits. Port4342 Vite was started for the browser probe; its owned process29612 was terminated and lsof confirmed no listener. All other probe servers bind ephemeral loopback ports and are cleaned up in finally paths.

## Repair follow-up

Independent re-review of the #17 repair is recorded in `shutdown-rereview.md`. Production repaira29ae2b and test follow-ups through clean2b03f16 were inspected; original real-TCP probe now completes, and37 focused tests passed (committed30 plus reviewer7). No new production finding. Windows real-backpressure fixture setup remains unresolved in that follow-up report; latest2b03f16 failed the backlog precondition before Stop.
