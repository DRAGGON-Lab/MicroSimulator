"""Shutdown regressions use real sockets and a controllably blocked worker."""

from __future__ import annotations

import asyncio
import signal
import socket
import sys
from pathlib import Path
from threading import Event
from threading import enumerate as threads
from typing import Any, cast
from urllib.parse import urlsplit

import pytest
from aiohttp import ClientSession, ClientWebSocketResponse, WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer
from microsimulator import BackendKind, CellInit, Simulation, load_checkpoint
from microsimulator import checkpoint as checkpoint_module
from microsimulator.checkpoint import JSONValue
from microsimulator.viewer_server import (
    LiveCommand,
    LiveController,
    LiveSession,
    LiveViewerError,
    create_live_app,
    parse_command,
)


def _factory() -> tuple[Simulation, dict[str, JSONValue]]:
    simulation = Simulation(BackendKind.CPU)
    simulation.add_cell(CellInit())
    return simulation, {}


def _dist(path: Path) -> Path:
    dist = path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>shutdown test</title>")
    return dist


class _BlockedModel:
    def __init__(self) -> None:
        self.simulation, _ = _factory()
        self.entered = Event()
        self.release = Event()

    def controller_state(self) -> JSONValue:
        return None

    def step(self, dt: float) -> None:
        self.entered.set()
        assert self.release.wait(10), "test did not release blocked step"
        self.simulation.step(dt)


async def _entered(event: Event) -> None:
    assert await asyncio.to_thread(event.wait, 5), "worker never reached blocking point"


@pytest.mark.parametrize("playing", [False, True])
def test_stop_preempts_same_socket_batch_and_rejects_new_commands(
    tmp_path: Path,
    playing: bool,
) -> None:
    async def exercise() -> None:
        model = _BlockedModel()
        session = LiveSession(lambda: (model, {}), dt=0.1)
        app, token = create_live_app(session, _dist(tmp_path), frame_steps=10_000)
        client = TestClient(TestServer(app))
        try:
            await client.start_server()
            origin = str(client.make_url("/")).rstrip("/")
            ws = await client.ws_connect(
                f"/api/v1/session?token={token}",
                headers={"Origin": origin},
            )
            await ws.receive_json(timeout=5)
            await ws.send_json({"type": "play"} if playing else {"type": "step", "steps": 10_000})
            await _entered(model.entered)
            await ws.send_json({"type": "stop"})
            await ws.send_json({"type": "stop"})
            # Read until admission is closed while the current step is blocked.
            while (await ws.receive_json(timeout=5)).get("type") != "session":
                pass
            for name in ("play", "step", "reset", "checkpoint"):
                await ws.send_json({"type": name})
                rejected = await ws.receive_json(timeout=5)
                assert rejected["type"] == "error"
                assert "stopping" in rejected["message"]
            assert session.completed_steps == 0
            model.release.set()
            while True:
                message = await ws.receive_json(timeout=5)
                if message == {"type": "session", "state": "stopped"}:
                    break
            assert session.completed_steps == 1
            assert (await ws.receive(timeout=5)).type == WSMsgType.CLOSE
            assert ws.close_code == 1000
        finally:
            model.release.set()
            await client.close()
        assert not any(t.name.startswith("microsimulator-live") for t in threads())

    asyncio.run(exercise())


def test_stop_waits_for_atomic_checkpoint_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        output = tmp_path / "run.json"
        session = LiveSession(_factory, dt=0.1, checkpoint_output=output)
        session.checkpoint()
        original = output.read_bytes()
        session.step()
        entered, release = Event(), Event()
        replace = checkpoint_module.os.replace

        def blocked_replace(source: str | Path, destination: str | Path) -> None:
            entered.set()
            assert release.wait(10), "test did not release checkpoint replace"
            replace(source, destination)

        monkeypatch.setattr(checkpoint_module.os, "replace", blocked_replace)
        controller = LiveController(session)
        save = asyncio.create_task(controller.command(LiveCommand("checkpoint")))
        try:
            await _entered(entered)
            await controller.command(LiveCommand("stop"))
            assert not controller.stopped.is_set()
            assert output.read_bytes() == original
            for name in ("play", "step", "reset", "checkpoint"):
                with pytest.raises(LiveViewerError, match="stopping"):
                    await controller.command(parse_command('{"type":"' + name + '"}'))
            release.set()
            assert await save == str(output.resolve())
            await asyncio.wait_for(controller.close(), 5)
            assert abs(load_checkpoint(output).time - 0.1) < 1e-7
            assert list(tmp_path.iterdir()) == [output]
        finally:
            release.set()
            await controller.close()

    asyncio.run(exercise())


def test_cancelled_waiter_does_not_release_worker_early() -> None:
    async def exercise() -> None:
        model = _BlockedModel()
        session = LiveSession(lambda: (model, {}), dt=0.1)
        controller = LiveController(session)
        operation = asyncio.create_task(controller.command(LiveCommand("step", 10_000)))
        try:
            await _entered(model.entered)
            operation.cancel()
            controller.request_stop()
            await asyncio.sleep(0)
            assert not operation.done()
            assert not controller.stopped.is_set()
            model.release.set()
            with pytest.raises(asyncio.CancelledError):
                await operation
            await asyncio.wait_for(controller.close(), 5)
            assert session.completed_steps == 1
        finally:
            model.release.set()
            await controller.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("_attempt", range(10))
def test_disconnect_pauses_without_stopping_and_close_is_idempotent(
    tmp_path: Path,
    _attempt: int,
) -> None:
    async def exercise() -> None:
        session = LiveSession(_factory, dt=0.1)
        app, token = create_live_app(session, _dist(tmp_path))
        client = TestClient(TestServer(app))
        await client.start_server()
        origin = str(client.make_url("/")).rstrip("/")
        try:
            ws = await client.ws_connect(
                f"/api/v1/session?token={token}",
                headers={"Origin": origin},
            )
            await ws.receive_json(timeout=5)
            await ws.send_json({"type": "play"})
            await ws.receive_json(timeout=5)
            await ws.close()
            ws2 = await client.ws_connect(
                f"/api/v1/session?token={token}",
                headers={"Origin": origin},
            )
            assert (await ws2.receive_json(timeout=5))["playing"] is False
            await ws2.send_json({"type": "stop"})
            await asyncio.gather(ws2.close(), client.close())
            await client.close()
        finally:
            await client.close()

    asyncio.run(exercise())


def test_stop_command_is_closed() -> None:
    assert parse_command('{"type":"stop"}') == LiveCommand("stop")
    with pytest.raises(LiveViewerError, match="unknown fields"):
        parse_command('{"type":"stop","force":true}')


def test_reconnect_observes_pause_before_disconnected_command_drains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        entered, release = Event(), Event()
        reconnect_started = asyncio.Event()
        frame_message = LiveSession.frame_message
        connect = LiveController.connect
        frames, connections = 0, 0

        def blocked_frame(session: LiveSession, *, playing: bool) -> dict[str, JSONValue]:
            nonlocal frames
            frames += 1
            if frames == 2:
                entered.set()
                assert release.wait(10), "test did not release frame capture"
            return frame_message(session, playing=playing)

        async def observe_connect(controller: LiveController, ws: web.WebSocketResponse) -> None:
            nonlocal connections
            connections += 1
            if connections == 2:
                reconnect_started.set()
            await connect(controller, ws)

        monkeypatch.setattr(LiveSession, "frame_message", blocked_frame)
        monkeypatch.setattr(LiveController, "connect", observe_connect)
        app, token = create_live_app(LiveSession(_factory, dt=0.1), _dist(tmp_path))
        client = TestClient(TestServer(app))
        try:
            await client.start_server()
            origin = str(client.make_url("/")).rstrip("/")

            async def open_socket() -> ClientWebSocketResponse:
                return await client.ws_connect(
                    f"/api/v1/session?token={token}",
                    headers={"Origin": origin},
                )

            ws = await open_socket()
            await ws.receive_json(timeout=5)
            await ws.send_json({"type": "play"})
            await _entered(entered)
            # close() finishes the network handshake while the server's canceled
            # frame capture is still holding its worker/serialization lock.
            await ws.close()
            reconnect = asyncio.create_task(open_socket())
            await asyncio.wait_for(reconnect_started.wait(), 5)
            release.set()
            ws2 = await reconnect
            assert (await ws2.receive_json(timeout=5))["playing"] is False
            await ws2.close()
        finally:
            release.set()
            await client.close()

    asyncio.run(exercise())


def test_stop_is_not_blocked_by_a_stalled_frame_send(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        entered = asyncio.Event()
        send = web.WebSocketResponse.send_str
        frame_count = 0

        async def stalled_send(
            ws: web.WebSocketResponse,
            data: str,
            compress: int | None = None,
        ) -> None:
            nonlocal frame_count
            if data.startswith('{"type":"frame"'):
                frame_count += 1
                if frame_count > 1:
                    entered.set()
                    await asyncio.Event().wait()
            await send(ws, data, compress=compress)

        monkeypatch.setattr(web.WebSocketResponse, "send_str", stalled_send)
        app, token = create_live_app(LiveSession(_factory, dt=0.1), _dist(tmp_path))
        client = TestClient(TestServer(app))
        try:
            await client.start_server()
            origin = str(client.make_url("/")).rstrip("/")
            ws = await client.ws_connect(
                f"/api/v1/session?token={token}",
                headers={"Origin": origin},
            )
            await ws.receive_json(timeout=5)
            await ws.send_json({"type": "play"})
            await asyncio.wait_for(entered.wait(), 5)
            await ws.send_json({"type": "stop"})
            assert await ws.receive_json(timeout=5) == {"type": "session", "state": "stopping"}
            assert await ws.receive_json(timeout=5) == {"type": "session", "state": "stopped"}
            await ws.close()
        finally:
            await client.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("termination", ["stop", "interrupt"])
def test_cli_process_exits_and_reuses_port_for_another_model(
    tmp_path: Path,
    termination: str,
) -> None:
    if termination == "interrupt" and sys.platform == "win32":
        pytest.skip("Windows Ctrl+C requires an attached console; use documented manual check")

    async def exercise() -> None:
        dist = _dist(tmp_path)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        for index in range(3):
            model = tmp_path / f"model {index}.py"
            model.write_text(
                "from microsimulator import CellInit\n"
                "def build(context):\n"
                "    simulation = context.simulation()\n"
                "    cell = CellInit()\n"
                f"    cell.length = {2 + index}\n"
                "    simulation.add_cell(cell)\n"
                "    return simulation\n"
            )
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "microsimulator",
                "view",
                "--model",
                str(model),
                "--backend",
                "cpu",
                "--dt",
                "0.1",
                "--viewer-dist",
                str(dist),
                "--port",
                str(port),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                assert process.stdout is not None
                line = (await asyncio.wait_for(process.stdout.readline(), 10)).decode().strip()
                if not line.startswith("MicroSimulator live viewer: "):
                    _, error = await asyncio.wait_for(process.communicate(), 5)
                    pytest.fail(f"viewer did not start: {line} {error.decode()}")
                url = urlsplit(line.split(": ", 1)[1])
                async with (
                    ClientSession() as client,
                    client.ws_connect(
                        f"http://{url.netloc}/api/v1/session?{url.query}",
                        headers={"Origin": f"http://{url.netloc}"},
                    ) as ws,
                ):
                    frame = cast(dict[str, Any], await ws.receive_json(timeout=5))
                    assert frame["scene"]["frame"]["cells"][0]["length"] == 2 + index
                    if termination == "interrupt":
                        process.send_signal(signal.SIGINT)
                    else:
                        await ws.send_json({"type": "stop"})
                    assert await ws.receive_json(timeout=5) == {
                        "type": "session",
                        "state": "stopping",
                    }
                    assert await ws.receive_json(timeout=5) == {
                        "type": "session",
                        "state": "stopped",
                    }
                    assert (await ws.receive(timeout=5)).type == WSMsgType.CLOSE
                _, error = await asyncio.wait_for(process.communicate(), 10)
                assert process.returncode == 0, error.decode()
                assert not error, error.decode()
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

    asyncio.run(exercise())
