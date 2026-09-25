# Independent re-review of PERSIST-17-1 repair

This is an archived review/evidence record. The [round summary](../feedback-review.md) records final closure, exact heads and successful Windows validation. Statements that a check was pending describe the time of that review, not its final status. Temporary evidence paths identify local artifacts; they are not repository dependencies.

**Verdict: the reviewed repair addresses PERSIST-17-1 on the independently exercised macOS paths. No new production correctness finding. Windows real-backpressure fixture setup remains unresolved; the latest exact-head run failed before the shutdown action.**

Reviewed issue 17 / PR 34 production diff `b50c72d78e84555fa050849721113897f22ed255..a29ae2be39f34c6123b23b8639149b6636fa1db5`, then both test-only follow-ups through clean committed head **`2b03f169dfdd5470139f1765127639deda3b368e`** (`a29ae2b..4451295..2b03f16`). Production source is unchanged after `a29ae2b`. The reviewer did not author or edit production code or committed tests. Commands below ran against the clean issue 17 worktree at `2b03f16`; Git head and clean status were checked before/after the focused suite.

## Why the repair closes the reproduced failure

`_close_socket` bounds the entire aiohttp close operation, covering the writer drain that precedes aiohttp's own handshake timeout. If delivery/close cannot finish, it aborts the captured transport rather than asking it to flush indefinitely. Shutdown lifecycle sends and socket closes run concurrently, so independent stalled receivers do not add serial timeout delays. Responsive clients retain normal notifications and code1000 close behavior.

Initial frame delivery is also bounded. `_websocket` owns a final `_close_socket` call after disconnecting the socket and draining its canceled command consumer, so an upgraded handler blocked on its initial send has a bounded path to completion. Capturing the transport at connect makes it available even if request/socket state changes during cleanup. The event/operation-lock/worker shutdown path still waits for current model mutation or atomic file replacement; the new network deadline does not truncate that work.

The added `CancelledError` handling distinguishes a canceled shared aiohttp drain waiter from explicit task cancellation. When `asyncio.current_task().cancelling()` is nonzero, cancellation is re-raised. Internal waiter cancellation with an uncanceled surrounding task causes failed transport cleanup. This distinction was examined in source and independently exercised for `_close_socket`, `_broadcast`, and `broadcast_frame`.

## Executed independent evidence

1. Re-ran the original unchanged reviewer real-TCP reproduction:

```console
.venv/bin/python /private/tmp/microsimulator-swarm/review-round-1/persistence/stalled_socket.py
```

```text
server_write_buffer_before_stop 6563021
healthy_initial_cells 40000
stopped_with_receiver_paused True 0.001014291075989604
cleanup_completed True
```

The receiver stayed paused. On fixed code this original script's stalled initial delivery can time out before the second healthy client's scene has been decoded and Stop is sent. Therefore its sub-millisecond Stop result demonstrates elimination of the old indefinite wait but is **not** claimed to measure Stop overlapping an active blocked send. The committed regression below establishes that overlap explicitly.

2. Ran clean-head committed shutdown/server tests together with seven reviewer-owned focused cases:

```console
.venv/bin/python -m pytest -q /private/tmp/microsimulator-swarm/review-round-1/persistence/test_shutdown_cancellation.py /private/tmp/microsimulator-swarm/review-round-1/persistence/test_failure_recovery.py python/tests/test_viewer_shutdown.py python/tests/test_viewer_server.py
```

**37 passed in 9.89 seconds.** The first invocation ran in the restricted sandbox: all 19 network cases failed exclusively at socket bind with PermissionError; 18 nonnetwork cases passed. The same command was reissued with the existing approved escalation and all 37 passed. The sandbox bind failures are not product failures and are not counted as successful coverage.

The committed 30 cases include both actual backpressure termination paths, authenticated Stop and application-runner cleanup. They connect the healthy client first, pause an actual raw TCP receiver before its authenticated upgrade, observe an actual server transport backlog exceeding 1 MB, and then begin shutdown immediately. They assert readable-client lifecycle messages, normal close, stopped event, completed application cleanup, cleared transport buffer, absence of live worker threads, and a different one-cell session reusing the same port. Neither model callbacks nor send/close operations are mocked in this regression. They also retain the original current-step-only batch cancellation, repeated Stop, rejected commands, checkpoint completion, accidental disconnect/reconnect, POSIX SIGINT, and repeated CLI process restart tests.

The six reviewer cancellation cases independently cover each of three new catch sites with (a) canceled underlying drain future and (b) explicitly canceled outer task. Case(a) returns after aborting the failed transport; case (b) propagates CancelledError. These focused cases intentionally use controlled awaitables to isolate cancellation semantics; they supplement rather than replace the real TCP regression.

The reviewer checkpoint-failure case independently blocks then fails os.replace while two close waiters run. It verifies unchanged old checkpoint bytes, successful old-file load, temporary-file removal, completed shutdown, no live worker threads, and rejection of subsequent Reset. It passed before and after the network repair.

## Test-only portability changes reviewed

`a29ae2b..4451295` limits only the marked probe connection's kernel send buffer, through aiohttp's response-prepare hook, while preserving the default production event loop/timeouts and the measured backlog assertion. `4451295..2b03f16` pauses the receiver before sending the upgrade, then synchronizes on the server's authenticated status 101 response preparation instead of allowing the client to prefetch scene bytes while reading the handshake. It adds diagnostic actual buffer/transport state on a setup timeout. These changes improve fixture control; they do not weaken the shutdown outcome checks or substitute a fake transport.

The earlier two Windows attempts failed in the new fixture's backlog-establishment precondition, before sending Stop; the existing 28 tests passed there. The exact-head Windows run for `2b03f16` also failed the new backlog-establishment precondition: Proactor reported queued 0 despite reader_bytes 0 and actual send 16384 / receive 4096 buffers. The author is diagnosing the server await chain. This review does not call that run passed, nor use the older Windows 28-test artifact as evidence for the new backpressure repair. The platform-specific fixture limitation remains open; it is not yet evidence of a production repair failure because Stop was never reached in those cases. No new Windows terminal or NVIDIA hardware run was performed by this reviewer.

## Cleanup and ownership

The review-owned Vite process 29612 on port 4342 was terminated after browser verification; a subsequent lsof query found no listener. No other reviewer's server was stopped. All reproduction-owned ephemeral servers closed in finally paths. The earlier named-replay browser probe passed through coordinator launch, with results and the explicitly agreed camera-tolerance correction recorded in `report.md`.


## Windows fixture follow-up reviewed at e9e631f

Reviewed clean committed head **`e9e631fae71b5dfd1a38355bc8bb2a277ffcdb86`**, including diagnostics-only `2b03f16..c67ff0d` and the Windows fixture adjustment `c67ff0d..e9e631f`. `git diff a29ae2b e9e631f --stat` confirms that only `python/tests/test_viewer_shutdown.py` changed after the production repair. No new production correctness finding.

Windows diagnostics showed both authenticated handlers already waiting to receive commands, two registered sockets, no Python transport backlog, and no bytes prefetched into the paused client's StreamReader. Therefore the first scene had been accepted below Python's transport accounting. The final fixture sends at most eight real authenticated Frame requests on Windows while a single owned task continuously drains the healthy client. POSIX retains the original stalled-initial-delivery scenario. This adjustment does not change the default Windows event loop, production timeouts, real network operations, or the requirement to observe more than 1 MB in the actual server transport before Stop or runner cleanup begins.

The drain task owns the only concurrent read from the healthy WebSocket. It acknowledges frame arrivals separately from a queue containing every non-frame message; unexpected errors are not silently filtered from the required stopping/stopped sequence. Normal verification awaits pump failure, receiver completion, and close code 1000. The bounded request pump is canceled before the termination action; cancellation affects the test helper, while already-running server work remains subject to production shutdown.

During review, the initial draft's helper cleanup could re-raise an already-failed task before closing test sockets. The author corrected that before commit: finally cancels owned helpers and uses `gather(return_exceptions=True)` before aborting the paused transport and closing the server. Normal-path task failures still propagate in the verification path. This is a resolved test-only review correction, not an additional production finding.

Unchanged outcome assertions still require the stopped event, completed application cleanup, a closing transport with zero queued bytes, no live worker threads, and immediate reuse of the same port by a different one-cell model. The following independent rerun passed on clean `e9e631f`:

```console
.venv/bin/python -m pytest -q python/tests/test_viewer_shutdown.py -k real_tcp_backpressure
```

**2 passed, 19 deselected in 8.44 seconds.** This is macOS/POSIX execution. The unchanged six cancellation and checkpoint-failure probes were not redundantly rerun. Exact-head Windows runs 36084995343 and 36084991741 were still active when this addendum was written; Windows success is not inferred from local results.


## Bounded timer follow-up reviewed at 4677491

Inspected clean committed head **`46774910df12cc913ab3dc83812ddf8e42f9aa42`** and its exact diff from `e9e631f`. The only functional change replaces the Windows pump's inner five-second `wait_for(frames.get())` with `await frames.get()`. This acknowledgement includes native scene capture as well as network delivery, so the removed deadline did not isolate the failure under test.

The setup observer retains its overall 15-second timeout. On setup failure, the enclosing finally explicitly cancels the pump and sole healthy-reader task and gathers their outcomes before releasing network resources. On observed backpressure, the normal path also cancels/awaits the pump before initiating shutdown. Therefore removal of the inner timeout does not introduce an unowned infinite acknowledgement wait. The maximum eight requests, measured actual backlog greater than 1 MB, default Windows event loop, lifecycle assertions, close/cleanup deadlines, worker checks, and same-port restart remain unchanged.

**No new finding.** No production source changed and no broad tests were repeated for this one-line fixture adjustment. The previous `e9e631f` Windows attempt failed only its inner acknowledgement deadline while the server was still awaiting capture; exact-head `4677491` Windows runs remain pending and are not counted as passed here.
