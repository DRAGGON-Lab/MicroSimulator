from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
from aiohttp import WSServerHandshakeError
from aiohttp.test_utils import TestClient, TestServer
from microsimulator import (
    BackendKind,
    CellInit,
    Simulation,
    load_checkpoint,
    load_checkpoint_bundle,
    viewer_server,
)
from microsimulator.checkpoint import JSONValue
from microsimulator.cli import main
from microsimulator.viewer_server import (
    LiveSession,
    LiveViewerError,
    create_live_app,
    parse_command,
)


def _factory() -> tuple[Simulation, dict[str, JSONValue]]:
    simulation = Simulation(BackendKind.CPU)
    cell = CellInit()
    cell.length = 2.0
    cell.growth_rate = 0.5
    simulation.add_cell(cell)
    return simulation, {"model": {"name": "viewer-test"}}


class _TestController:
    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation
        self.completed_steps = 0

    def step(self, dt: float) -> None:
        self.simulation.step(dt)
        self.completed_steps += 1

    def controller_state(self) -> JSONValue:
        return {
            "kind": "viewer-test-controller",
            "version": 1,
            "completed_steps": self.completed_steps,
        }


def _controller_factory() -> tuple[_TestController, dict[str, JSONValue]]:
    simulation, provenance = _factory()
    return _TestController(simulation), provenance


def _dist(path: Path) -> Path:
    dist = path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")
    (assets / "app.js").write_text("", encoding="utf-8")
    return dist


def test_live_session_steps_resets_and_writes_only_configured_checkpoint(
    tmp_path: Path,
) -> None:
    output = tmp_path / "live.cm2.json"
    session = LiveSession(_factory, dt=0.25, checkpoint_output=output)
    initial = session.frame_message(playing=False)
    assert initial["revision"] == 0
    assert cast(dict[str, Any], initial["scene"])["frame"]["time"] == 0.0

    session.step(2)
    stepped = session.frame_message(playing=False)
    assert stepped["completed_steps"] == 2
    assert cast(dict[str, Any], stepped["scene"])["frame"]["time"] == 0.5
    assert session.checkpoint() == output.resolve()
    assert load_checkpoint(output).time == 0.5

    session.reset()
    reset = session.frame_message(playing=False)
    assert reset["revision"] == 2
    assert reset["completed_steps"] == 0
    assert cast(dict[str, Any], reset["scene"])["frame"]["time"] == 0.0

    disabled = LiveSession(_factory, dt=0.25)
    with pytest.raises(LiveViewerError, match="not configured"):
        disabled.checkpoint()


def test_live_session_preserves_native_controller_state(tmp_path: Path) -> None:
    output = tmp_path / "controller.cm2.json"
    session = LiveSession(_controller_factory, dt=0.2, checkpoint_output=output)
    session.step(3)
    session.checkpoint()

    bundle = load_checkpoint_bundle(output)
    assert bundle.controller == {
        "kind": "viewer-test-controller",
        "version": 1,
        "completed_steps": 3,
    }


@pytest.mark.parametrize(
    ("encoded", "message"),
    [
        ('{"type":"step","steps":0}', "step count"),
        ('{"type":"step","path":"elsewhere"}', "unknown fields"),
        ('{"type":"reset","extra":true}', "unknown fields"),
        ('{"type":"eval"}', "unknown command"),
        ('["step"]', "JSON object"),
    ],
)
def test_command_protocol_is_closed(encoded: str, message: str) -> None:
    with pytest.raises(LiveViewerError, match=message):
        parse_command(encoded)


def test_live_websocket_requires_same_origin_token_and_controls_session(tmp_path: Path) -> None:
    async def exercise() -> None:
        checkpoint = tmp_path / "socket.cm2.json"
        application, token = create_live_app(
            LiveSession(_factory, dt=0.1, checkpoint_output=checkpoint),
            _dist(tmp_path),
            token="a" * 32,
            fps=60.0,
        )
        client = TestClient(TestServer(application))
        await client.start_server()
        origin = str(client.make_url("/")).rstrip("/")
        try:
            with pytest.raises(WSServerHandshakeError) as wrong_token:
                await client.ws_connect(
                    "/api/v1/session?token=wrong",
                    headers={"Origin": origin},
                )
            assert wrong_token.value.status == 403
            with pytest.raises(WSServerHandshakeError) as wrong_origin:
                await client.ws_connect(
                    f"/api/v1/session?token={token}",
                    headers={"Origin": "https://attacker.invalid"},
                )
            assert wrong_origin.value.status == 403

            socket = await client.ws_connect(
                f"/api/v1/session?token={token}",
                headers={"Origin": origin},
            )
            initial = cast(dict[str, Any], await socket.receive_json())
            assert initial["type"] == "frame"
            assert initial["playing"] is False

            await socket.send_str(json.dumps({"type": "step", "steps": 2}))
            stepped = cast(dict[str, Any], await socket.receive_json())
            assert stepped["completed_steps"] == 2
            assert abs(cast(float, stepped["scene"]["frame"]["time"]) - 0.2) < 1.0e-7

            await socket.send_json({"type": "checkpoint"})
            saved = cast(dict[str, Any], await socket.receive_json())
            assert saved == {"type": "checkpoint", "path": str(checkpoint.resolve())}
            assert checkpoint.exists()

            await socket.send_json({"type": "reset"})
            reset = cast(dict[str, Any], await socket.receive_json())
            assert reset["completed_steps"] == 0
            assert reset["scene"]["frame"]["time"] == 0.0

            await socket.send_json({"type": "eval", "source": "malicious()"})
            error = cast(dict[str, Any], await socket.receive_json())
            assert error["type"] == "error"
            assert "unknown command" in error["message"]
            await socket.close()
        finally:
            await client.close()

    asyncio.run(exercise())


def test_cli_constructs_a_resettable_live_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.py"
    model.write_text(
        """from microsimulator import CellInit

def build(context):
    simulation = context.simulation()
    cell = CellInit()
    cell.length = float(context.parameters["length"])
    simulation.add_cell(cell)
    return simulation
""",
        encoding="utf-8",
    )
    dist = _dist(tmp_path)
    captured: dict[str, Any] = {}

    def fake_serve(
        session: LiveSession,
        viewer_dist: str | Path,
        **options: object,
    ) -> None:
        session.step()
        captured["stepped"] = session.frame_message(playing=False)
        session.reset()
        captured["reset"] = session.frame_message(playing=False)
        captured["dist"] = Path(viewer_dist)
        captured["options"] = options

    monkeypatch.setattr(viewer_server, "serve_live", fake_serve)
    status = main(
        [
            "view",
            "--model",
            str(model),
            "--parameter",
            "length=4.25",
            "--seed",
            "17",
            "--dt",
            "0.2",
            "--viewer-dist",
            str(dist),
            "--port",
            "9001",
            "--frame-steps",
            "3",
            "--fps",
            "24",
        ]
    )

    assert status == 0
    assert captured["dist"] == dist.resolve()
    assert cast(dict[str, object], captured["options"])["port"] == 9001
    stepped = cast(dict[str, Any], captured["stepped"])
    reset = cast(dict[str, Any], captured["reset"])
    assert stepped["scene"]["frame"]["cells"][0]["length"] > 4.25
    assert stepped["scene"]["frame"]["time"] > 0.0
    assert reset["scene"]["frame"]["time"] == 0.0
    assert reset["scene"]["frame"]["cells"][0]["length"] == 4.25
