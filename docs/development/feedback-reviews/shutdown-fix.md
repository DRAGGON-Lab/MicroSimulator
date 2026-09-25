# PERSIST-17-1 shutdown repair — verified

This is an archived review/evidence record. The [round summary](../feedback-review.md) records final closure, exact heads and successful Windows validation. Statements that a check was pending describe the time of that review, not its final status. Temporary evidence paths identify local artifacts; they are not repository dependencies.

- Final pushed head: `46774910df12cc913ab3dc83812ddf8e42f9aa42`, branch `marpaia/17`.
- Production repair: `a29ae2be39f34c6123b23b8639149b6636fa1db5`; subsequent commits change only regression setup/diagnostics.
- Worktree: `/private/tmp/microsimulator-swarm/issue-17`; clean. No master, integration, or frozen-base edits.

## Confirmed original failure

Ran the independent reviewer script `stalled_socket.py` under this worktree's own Python with a 50-second outer subprocess timeout before editing. A valid 40,000-cell scene left **6,563,551 bytes** in the real server write buffer while its TCP receiver was paused. A second authenticated receiver saw `stopping` and `stopped`, but `controller.stopped` was false after **12.000854708 seconds**.

The await chain was `LiveController._close -> WebSocketResponse.close -> StreamWriter.drain -> BaseProtocol._drain_helper`. Resuming the paused receiver allowed cleanup to finish: aiohttp's internal close timeout starts after its preceding drain.

## Repair

`viewer_server.py` now places a one-second outer deadline around the complete WebSocket close, including close-frame write and drain. Timeout, failed close, or dependent drain cancellation aborts the captured transport. A regular transport close can otherwise keep flushing indefinitely.

Initial frame writes have the same one-second delivery deadline as frame broadcasts. WebSocket request finalization also executes bounded socket close, so an initial-send request cannot hold application cleanup indefinitely. Lifecycle sends and socket closes run concurrently across receivers; playback stops attempting further frame delivery once shutdown begins.

The real regression exposed another race: aiohttp shares a drain waiter between writes, so timing out an initial write can cancel an independent close/write waiter. Dependent cancellation is treated as a failed network operation and aborts the connection; a task's own explicit cancellation still propagates, distinguished through `Task.cancelling()`.

Native work remains cooperative: the current step, reset, scene capture, or atomic checkpoint write finishes before worker shutdown. Authentication, closed command vocabulary, healthy `stopping`/`stopped` notifications and code-1000 closure are retained. The protocol document distinguishes network deadlines from cooperative model-operation completion.

## Actual backpressure regression

`test_real_tcp_backpressure_releases_connections_and_reuses_port` covers authenticated Stop and application-runner cleanup.

It verifies a healthy receiver's valid 40,000-cell scene, performs a second authenticated WebSocket upgrade over a real TCP socket, and keeps that receiver paused. The probe constrains kernel socket buffers, waits for the real validated status-101 server response, and **requires more than 1 MB actually queued in the server transport before shutdown**. No send/close mock, production timeout change, alternate Windows loop, or skipped backpressure assertion is used.

Platform setup differs explicitly:

- **POSIX/macOS:** the valid initial scene itself stalls delivery.
- **Windows:** the default Proactor loop can accept the entire first frame below Python transport accounting even though the client has not read it. At most eight real authenticated Frame requests establish sustained broadcast backpressure. A dedicated task drains the healthy receiver. The pump stops immediately after observed backlog; one 15-second setup budget covers native scene capture and receipt. Windows coverage is sustained broadcast backpressure, not a claim that its initial send stalled.

Both paths retain five-second post-request Stop/cleanup checks, healthy lifecycle notifications and normal close, zero queued transport bytes after abort, no simulation worker thread, and a second distinct model successfully bound/read on the exact same port before the paused client is released. Helper tasks are canceled and collected in finally even if a helper had already failed; errors remain visible on the normal verification path. Only legitimate frame completion is ignored while waiting for lifecycle notifications.

## Validation

- Full local live-server/shutdown set after production repair: **30 passed in 7.97 seconds**.
- Final affected local real-backpressure cases at `4677491`: **2 passed in 8.42 seconds**.
- Changed Python Ruff lint/format and Pyright: passed, **0 type errors / 0 warnings**. `git diff --check`: passed.
- Original independent reproduction after production repair completes with the receiver still paused; exact output: `shutdown-fix-reproduction.log`.
- Exact-head hosted Windows run [36085257990](https://github.com/DRAGGON-Lab/MicroSimulator/actions/runs/36085257990): **SUCCESS, 30 passed in 25.23 seconds**, Windows/Python 3.12.10/pytest 8.4.2.
- Exact-head paired hosted Windows run [36085262246](https://github.com/DRAGGON-Lab/MicroSimulator/actions/runs/36085262246): **SUCCESS**.
- The Windows log explicitly confirms both real-backpressure cases and both subprocess Stop/actual console Ctrl+C restart cases passed. Full successful log saved as `shutdown-fix-windows-36085257990.log`.
- Independent reviewer separately reports the unchanged production repair passes its original stalled-socket reproduction plus 37 local cases, including six cancellation distinctions and checkpoint-write failure. Its review document is the authority for those additional checks.

## Windows setup diagnosis and limits

Earlier Windows runs failed the regression precondition, with the other 28 tests passing. Bounded diagnostics proved two registered WebSocket handlers had already completed initial delivery and were waiting in `receive()`, while the paused Python reader and Proactor write buffer both held zero bytes. This ruled out claiming an initial-send stall on Windows. The first sustained-broadcast attempt then hit a redundant five-second acknowledgement timer while still capturing a native scene; removing that inner timer retained the single overall 15-second setup bound and let the real backlog prerequisite run. Final exact-head runs above passed all assertions.

The existing hosted CPU workflow was not weakened or switched to another event loop. No GPU coverage is claimed for this network-only repair, and no timeout is claimed for a model operation that never cooperatively returns.
