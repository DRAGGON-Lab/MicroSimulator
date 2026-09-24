"""Authenticated loopback control protocol for the independent scene viewer."""

from __future__ import annotations

import asyncio
import json
import math
import secrets
import webbrowser
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlencode

from aiohttp import WSMsgType, web

from .checkpoint import JSONValue, save_checkpoint
from .runner import RunnableModel, controller_state, model_channel_metadata, native_simulation
from .scene import capture_scene, dumps_scene

MAX_COMMAND_BYTES = 4096
MAX_STEP_BATCH = 10_000

type ModelFactory = Callable[[], tuple[RunnableModel, Mapping[str, JSONValue]]]
type CommandName = Literal["frame", "step", "play", "pause", "reset", "checkpoint"]


class LiveViewerError(RuntimeError):
    """Raised when a live-viewer session or command is invalid."""


@dataclass(frozen=True, slots=True)
class LiveCommand:
    name: CommandName
    steps: int = 0


def parse_command(encoded: str) -> LiveCommand:
    """Parse the closed, data-only viewer command vocabulary."""

    if len(encoded.encode("utf-8")) > MAX_COMMAND_BYTES:
        raise LiveViewerError(f"command exceeds the {MAX_COMMAND_BYTES}-byte limit")
    try:
        value = cast(object, json.loads(encoded))
    except (json.JSONDecodeError, RecursionError) as error:
        raise LiveViewerError("command is not valid JSON") from error
    if not isinstance(value, dict):
        raise LiveViewerError("command must be a JSON object")
    command = cast(dict[object, object], value)
    name = command.get("type")
    if name == "step":
        if set(command) - {"type", "steps"}:
            raise LiveViewerError("step command has unknown fields")
        steps = command.get("steps", 1)
        if (
            not isinstance(steps, int)
            or isinstance(steps, bool)
            or steps < 1
            or steps > MAX_STEP_BATCH
        ):
            raise LiveViewerError(f"step count must be an integer in [1, {MAX_STEP_BATCH}]")
        return LiveCommand("step", steps)
    names: set[str] = {"frame", "play", "pause", "reset", "checkpoint"}
    if not isinstance(name, str) or name not in names:
        raise LiveViewerError("unknown command type")
    if set(command) != {"type"}:
        raise LiveViewerError(f"{name} command has unknown fields")
    return LiveCommand(cast(CommandName, name))


class LiveSession:
    """Own a deterministic runnable model and produce immutable scene documents."""

    def __init__(
        self,
        factory: ModelFactory,
        *,
        dt: float,
        checkpoint_output: str | Path | None = None,
    ) -> None:
        if not math.isfinite(dt) or dt <= 0.0:
            raise LiveViewerError("time step must be finite and positive")
        self._factory = factory
        self._dt = dt
        self._checkpoint_output = (
            Path(checkpoint_output).resolve() if checkpoint_output is not None else None
        )
        self._model, self._provenance = self._build()
        self._completed_steps = 0
        self._revision = 0

    @property
    def completed_steps(self) -> int:
        return self._completed_steps

    @property
    def checkpoint_enabled(self) -> bool:
        return self._checkpoint_output is not None

    def _build(self) -> tuple[RunnableModel, dict[str, JSONValue]]:
        model, provenance = self._factory()
        native_simulation(model).validate()
        model_channel_metadata(model)
        return model, dict(provenance)

    def step(self, steps: int = 1) -> None:
        if steps < 1 or steps > MAX_STEP_BATCH:
            raise LiveViewerError(f"step count must be in [1, {MAX_STEP_BATCH}]")
        completed = 0
        try:
            for _ in range(steps):
                self._model.step(self._dt)
                completed += 1
                self._completed_steps += 1
        finally:
            if completed > 0:
                self._revision += 1

    def reset(self) -> None:
        model, provenance = self._build()
        self._model = model
        self._provenance = provenance
        self._completed_steps = 0
        self._revision += 1

    def checkpoint(self) -> Path:
        destination = self._checkpoint_output
        if destination is None:
            raise LiveViewerError("checkpoint output is not configured")
        provenance = dict(self._provenance)
        provenance["live_session"] = {
            "completed_steps": self._completed_steps,
            "dt": self._dt,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            native_simulation(self._model),
            destination,
            provenance=provenance,
            controller=controller_state(self._model),
            channel_metadata=model_channel_metadata(self._model),
        )
        return destination

    def frame_message(self, *, playing: bool) -> dict[str, JSONValue]:
        native = native_simulation(self._model)
        scene = cast(
            dict[str, JSONValue],
            json.loads(
                dumps_scene(
                    capture_scene(native, channel_metadata=model_channel_metadata(self._model))
                )
            ),
        )
        return {
            "type": "frame",
            "revision": self._revision,
            "completed_steps": self._completed_steps,
            "playing": playing,
            "checkpoint_enabled": self.checkpoint_enabled,
            "scene": scene,
        }


class LiveController:
    """Serialize model work on one worker and broadcast frames to web clients."""

    def __init__(self, session: LiveSession, *, frame_steps: int = 1, fps: float = 30.0) -> None:
        if frame_steps < 1 or frame_steps > MAX_STEP_BATCH:
            raise LiveViewerError(f"frame step count must be in [1, {MAX_STEP_BATCH}]")
        if not math.isfinite(fps) or fps <= 0.0 or fps > 240.0:
            raise LiveViewerError("frame rate must be finite and in (0, 240]")
        self.session = session
        self.frame_steps = frame_steps
        self.frame_interval = 1.0 / fps
        self.playing = False
        self._sockets: set[web.WebSocketResponse] = set()
        self._play_task: asyncio.Task[None] | None = None
        self._play_wakeup = asyncio.Event()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="microsimulator-live")
        self._operation_lock = asyncio.Lock()

    async def _run(self, operation: Callable[..., Any], *arguments: object) -> Any:
        async with self._operation_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._worker, operation, *arguments)

    async def _message(self) -> dict[str, JSONValue]:
        return cast(
            dict[str, JSONValue],
            await self._run(partial(self.session.frame_message, playing=self.playing)),
        )

    async def _send_frame(self, socket: web.WebSocketResponse) -> None:
        await socket.send_str(json.dumps(await self._message(), separators=(",", ":")))

    async def broadcast_frame(self) -> None:
        if not self._sockets:
            return
        encoded = json.dumps(await self._message(), separators=(",", ":"))
        stale: list[web.WebSocketResponse] = []
        for socket in tuple(self._sockets):
            if socket.closed:
                stale.append(socket)
                continue
            try:
                await socket.send_str(encoded)
            except (ConnectionError, RuntimeError):
                stale.append(socket)
        self._sockets.difference_update(stale)

    async def _play(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while self.playing and self._sockets:
                started = loop.time()
                await self._run(self.session.step, self.frame_steps)
                await self.broadcast_frame()
                delay = self.frame_interval - (loop.time() - started)
                if delay > 0.0:
                    with suppress(TimeoutError):
                        await asyncio.wait_for(self._play_wakeup.wait(), timeout=delay)
                    self._play_wakeup.clear()
        finally:
            self.playing = False
            self._play_task = None

    async def play(self) -> None:
        if self.playing:
            return
        self.playing = True
        self._play_wakeup.clear()
        self._play_task = asyncio.create_task(self._play(), name="microsimulator-live-play")
        await self.broadcast_frame()

    async def pause(self, *, broadcast: bool = True) -> None:
        self.playing = False
        self._play_wakeup.set()
        task = self._play_task
        if task is not None and task is not asyncio.current_task():
            await task
        if broadcast:
            await self.broadcast_frame()

    async def command(self, command: LiveCommand) -> str | None:
        if command.name == "frame":
            await self.broadcast_frame()
        elif command.name == "step":
            await self.pause(broadcast=False)
            await self._run(self.session.step, command.steps)
            await self.broadcast_frame()
        elif command.name == "play":
            await self.play()
        elif command.name == "pause":
            await self.pause()
        elif command.name == "reset":
            await self.pause(broadcast=False)
            await self._run(self.session.reset)
            await self.broadcast_frame()
        else:
            destination = cast(Path, await self._run(self.session.checkpoint))
            return str(destination)
        return None

    async def connect(self, socket: web.WebSocketResponse) -> None:
        self._sockets.add(socket)
        await self._send_frame(socket)

    async def disconnect(self, socket: web.WebSocketResponse) -> None:
        self._sockets.discard(socket)
        if not self._sockets:
            await self.pause(broadcast=False)

    async def close(self) -> None:
        await self.pause(broadcast=False)
        sockets = tuple(self._sockets)
        self._sockets.clear()
        for socket in sockets:
            await socket.close(code=1001, message=b"server shutdown")
        self._worker.shutdown(wait=True, cancel_futures=True)


_CONTROLLER_KEY = web.AppKey("microsimulator.controller", LiveController)
_TOKEN_KEY = web.AppKey("microsimulator.token", str)


def _authorized(request: web.Request) -> bool:
    token = request.query.get("token", "")
    if not secrets.compare_digest(token, request.app[_TOKEN_KEY]):
        return False
    origin = request.headers.get("Origin")
    return origin == f"{request.scheme}://{request.host}"


async def _websocket(request: web.Request) -> web.StreamResponse:
    if not _authorized(request):
        raise web.HTTPForbidden(text="invalid live-viewer authority")
    socket = web.WebSocketResponse(max_msg_size=MAX_COMMAND_BYTES, heartbeat=20.0)
    await socket.prepare(request)
    controller = request.app[_CONTROLLER_KEY]
    await controller.connect(socket)
    try:
        async for message in socket:
            if message.type is not WSMsgType.TEXT:
                if message.type is WSMsgType.ERROR:
                    break
                await socket.send_json({"type": "error", "message": "text commands required"})
                continue
            try:
                result = await controller.command(parse_command(cast(str, message.data)))
                if result is not None:
                    await socket.send_json(
                        {"type": "checkpoint", "path": result},
                        dumps=lambda value: json.dumps(value, separators=(",", ":")),
                    )
            except Exception as error:
                await socket.send_json(
                    {"type": "error", "message": str(error)},
                    dumps=lambda value: json.dumps(value, separators=(",", ":")),
                )
    finally:
        await controller.disconnect(socket)
    return socket


def create_live_app(
    session: LiveSession,
    viewer_dist: str | Path,
    *,
    token: str | None = None,
    frame_steps: int = 1,
    fps: float = 30.0,
) -> tuple[web.Application, str]:
    """Create an aiohttp application and return it with its bearer token."""

    dist = Path(viewer_dist).resolve()
    index = dist / "index.html"
    assets = dist / "assets"
    if not index.is_file() or not assets.is_dir():
        raise LiveViewerError(f"viewer distribution is incomplete: {dist}")
    authority = token or secrets.token_urlsafe(32)
    if len(authority) < 32:
        raise LiveViewerError("live-viewer token must contain at least 32 characters")

    controller = LiveController(session, frame_steps=frame_steps, fps=fps)
    application = web.Application(client_max_size=MAX_COMMAND_BYTES)
    application[_CONTROLLER_KEY] = controller
    application[_TOKEN_KEY] = authority

    async def index_response(_: web.Request) -> web.FileResponse:
        response = web.FileResponse(index)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    async def cleanup(_: web.Application) -> None:
        await controller.close()

    application.router.add_get("/", index_response)
    application.router.add_get("/api/v1/session", _websocket)
    application.router.add_static("/assets", assets, show_index=False)
    application.on_cleanup.append(cleanup)
    return application, authority


def serve_live(
    session: LiveSession,
    viewer_dist: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    frame_steps: int = 1,
    fps: float = 30.0,
    open_browser: bool = False,
) -> None:
    """Serve one live session on loopback until interrupted."""

    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise LiveViewerError("live viewer host must be a loopback address")
    if port < 1 or port > 65_535:
        raise LiveViewerError("live viewer port must be in [1, 65535]")
    application, token = create_live_app(
        session,
        viewer_dist,
        frame_steps=frame_steps,
        fps=fps,
    )
    display_host = f"[{host}]" if ":" in host else host
    url = f"http://{display_host}:{port}/?{urlencode({'token': token})}"

    async def run() -> None:
        runner = web.AppRunner(application)
        await runner.setup()
        try:
            site = web.TCPSite(runner, host=host, port=port)
            try:
                await site.start()
            except OSError as error:
                raise LiveViewerError(
                    f"could not bind live viewer to {host}:{port}: {error}"
                ) from error
            print(f"MicroSimulator live viewer: {url}", flush=True)
            if open_browser:
                webbrowser.open(url)
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()

    with suppress(KeyboardInterrupt):
        asyncio.run(run())
