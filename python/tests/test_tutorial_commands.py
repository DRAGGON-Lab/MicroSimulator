"""Execute the published CLI examples through real shells, without rewriting them."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

import pytest
from aiohttp import ClientSession, WSMsgType
from microsimulator import load_checkpoint_bundle
from microsimulator.cli import _parser  # pyright: ignore[reportPrivateUsage]

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs/tutorials/commands.md"
BLOCKS = dict(
    re.findall(
        r"<!-- tutorial-command: ([\w-]+) -->\s*```\w+\n(.*?)\n```",
        GUIDE.read_text(encoding="utf-8"),
        re.DOTALL,
    )
)
SHELLS = ("pwsh",) if sys.platform == "win32" else ("sh", "bash", "zsh", "pwsh")


def _same_time(actual: float, expected: float) -> bool:
    # dt is stored in native float32; retain pytest.approx's intended tolerance.
    return math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8)


def _kill_tree(pid: int) -> None:
    # Only called for a subprocess created by this test. Killing just its shell
    # leaves uv/the live server holding the inherited stdout pipe and port.
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
            timeout=10,
        )
    else:
        with suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)


def _run_script(
    arguments: list[str],
    environment: dict[str, str],
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    with subprocess.Popen(
        arguments,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=sys.platform != "win32",
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=120)
        except subprocess.TimeoutExpired as error:
            _kill_tree(process.pid)
            stdout, stderr = process.communicate(timeout=10)
            raise AssertionError(f"command timed out: {stdout} {stderr}") from error
        return subprocess.CompletedProcess(arguments, process.returncode, stdout, stderr)


@dataclass
class CommandShell:
    name: str
    executable: str
    cwd: Path
    records: list[dict[str, Any]] = field(default_factory=lambda: list[dict[str, Any]]())

    def prepare(self, command: str) -> tuple[list[str], dict[str, str]]:
        environment = dict(os.environ)
        environment.pop("VIRTUAL_ENV", None)
        environment["UV_PROJECT_ENVIRONMENT"] = sys.prefix
        environment["PYTHONUNBUFFERED"] = "1"
        environment["UV_OFFLINE"] = "1"
        if self.name == "pwsh":
            script = self.cwd / "command.ps1"
            script.write_text(
                "$ErrorActionPreference = 'Stop'\n"
                "$global:LASTEXITCODE = 0\n"
                + BLOCKS["powershell-mode"]
                + "\n"
                + command
                + "\nexit $LASTEXITCODE\n",
                encoding="utf-8",
            )
            arguments = [self.executable, "-NoLogo", "-NoProfile", "-File", str(script)]
        else:
            script = self.cwd / "command.sh"
            script.write_text("set -e\n" + command + "\n", encoding="utf-8")
            arguments = [self.executable, str(script)]
        return arguments, environment

    def run(self, identifier: str, *, success: bool = True) -> subprocess.CompletedProcess[str]:
        command = BLOCKS[identifier]
        arguments, environment = self.prepare(command)
        print(f"Executing {self.name}: {identifier}", flush=True)
        result = _run_script(arguments, environment, self.cwd)
        self.records.append(
            {
                "id": identifier,
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        assert (result.returncode == 0) is success, result.stdout + result.stderr
        return result


@pytest.fixture(params=SHELLS)
def shell(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[CommandShell]:
    name = cast(str, request.param)
    executable = shutil.which(name)
    if executable is None:
        pytest.skip(f"{name} is not installed; no {name} coverage claimed")
    cwd = tmp_path / "repository with spaces"
    cwd.mkdir()
    shutil.copytree(ROOT / "examples", cwd / "examples")
    (cwd / "pyproject.toml").write_text(
        '[project]\nname="tutorial-command-test"\nversion="0.0.0"\n'
        'requires-python=">=3.12,<3.13"\n',
        encoding="utf-8",
    )
    # CLI lifecycle checks deliberately do not claim to test the browser bundle.
    (cwd / "viewer/dist/assets").mkdir(parents=True)
    (cwd / "viewer/dist/index.html").write_text("<!doctype html><title>CLI test</title>")
    instance = CommandShell(name, executable, cwd)
    version_args = (
        ["-NoLogo", "-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"]
        if name == "pwsh"
        else ["--version"]
        if name != "sh"
        else ["-c", "echo POSIX-sh"]
    )
    version = subprocess.run(
        [executable, *version_args], capture_output=True, text=True, timeout=60, check=True
    ).stdout.strip()
    if name == "pwsh":
        assert tuple(int(part) for part in version.split(".")[:2]) >= (7, 3)
    try:
        yield instance
    finally:
        if destination := os.environ.get("MICROSIMULATOR_COMMAND_REPORT"):
            directory = Path(destination)
            directory.mkdir(parents=True, exist_ok=True)
            node = cast(Any, request).node  # pytest's request.node is untyped.
            (directory / f"{node.name}.json").write_text(
                json.dumps(
                    {
                        "platform": platform.platform(),
                        "python": sys.version,
                        "shell": name,
                        "shell_version": version,
                        "cwd_contains_spaces": True,
                        "commands": instance.records,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )


def _document(path: Path) -> dict[str, Any]:
    # Authenticate checkpoints as well as inspecting their human-readable fields.
    load_checkpoint_bundle(path)
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_documented_headless_commands(shell: CommandShell) -> None:
    shell.run("devices")
    devices = cast(list[dict[str, Any]], json.loads(shell.run("devices-json").stdout))
    assert {record["backend"] for record in devices} == {"cpu", "metal", "cuda"}
    for record in devices:
        backend = record["backend"]
        result = shell.run(f"trap-{backend}", success=record["available"])
        output = shell.cwd / f"results/tutorial runs/trap-{backend}.json"
        if not record["available"]:
            assert f"backend {backend} device 0 is unavailable" in result.stderr
            assert not output.exists()
            continue
        document = _document(output)
        assert document["source_backend"]["kind"] == backend
        assert document["provenance"]["model"]["seed"] == 42
        assert document["provenance"]["model"]["parameters"] == {}
        assert document["provenance"]["run"]["completed_steps"] == 100
        assert _same_time(document["simulation"]["time"], 2.0)
        assert len(list(output.parent.glob(f"trap-{backend}.step-*.json"))) == 5
    shell.run("resume-trap")
    assert _same_time(
        _document(shell.cwd / "results/tutorial runs/trap-cpu-resumed.json")["simulation"]["time"],
        4.0,
    )

    suffix = "powershell" if shell.name == "pwsh" else "posix"
    shell.run(f"copy-{suffix}")
    shell.run(f"basics-{suffix}")
    shell.run("resume-basics")
    initial = _document(shell.cwd / "results/tutorial runs/basics.json")
    resumed = _document(shell.cwd / "results/tutorial runs/basics-resumed.json")
    for document in (initial, resumed):
        provenance = document["provenance"]["model"]
        assert provenance["seed"] == 42
        assert provenance["parameters"] == {"scenario": "basics"}
        assert (
            provenance["sha256"]
            == hashlib.sha256(
                (shell.cwd / "results/tutorial models/biophysics.py").read_bytes()
            ).hexdigest()
        )
    assert _same_time(initial["simulation"]["time"], 0.2)
    assert _same_time(resumed["simulation"]["time"], 0.4)
    assert (
        resumed["provenance"]["resume"]["sha256"]
        == hashlib.sha256(
            (shell.cwd / "results/tutorial runs/basics.json").read_bytes()
        ).hexdigest()
    )

    # Independently establish the claimed continuation semantics, using the same
    # installed CLI for a 20-step uninterrupted run.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "microsimulator",
            "run",
            "--model",
            str(shell.cwd / "results/tutorial models/biophysics.py"),
            "--parameter",
            'scenario="basics"',
            "--seed",
            "42",
            "--steps",
            "20",
            "--dt",
            "0.02",
            "--output",
            str(shell.cwd / "uninterrupted.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    uninterrupted = _document(shell.cwd / "uninterrupted.json")
    assert resumed["simulation"] == uninterrupted["simulation"]
    assert resumed["controller"] == uninterrupted["controller"]

    # The documented guard rejects overrides and edited source before executing
    # it; neither failed resume should create an output or run injected code.
    arguments, environment = shell.prepare(BLOCKS["resume-basics"] + " --parameter 'scenario=1'")
    rejected = subprocess.run(
        arguments,
        env=environment,
        cwd=shell.cwd,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert rejected.returncode != 0
    assert "do not pass --parameter" in rejected.stderr
    model = shell.cwd / "results/tutorial models/biophysics.py"
    model.write_text("raise RuntimeError('model executed before digest check')\n", encoding="utf-8")
    rejected = shell.run("resume-basics", success=False)
    assert "model digest does not match checkpoint" in rejected.stderr
    assert "model executed before digest check" not in rejected.stderr


def test_documented_live_commands(shell: CommandShell) -> None:
    suffix = "powershell" if shell.name == "pwsh" else "posix"
    shell.run(f"copy-{suffix}")
    shell.run(f"basics-{suffix}")

    async def exercise() -> None:
        for identifier in ("live-trap", "live-basics", "live-resume"):
            arguments, environment = shell.prepare(BLOCKS[identifier])
            process = await asyncio.create_subprocess_exec(
                *arguments,
                cwd=shell.cwd,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=sys.platform != "win32",
            )
            stopped = False
            try:
                assert process.stdout is not None
                line = (await asyncio.wait_for(process.stdout.readline(), 60)).decode().strip()
                assert line.startswith("MicroSimulator live viewer: "), line
                url = urlsplit(line.split(": ", 1)[1])
                assert url.port == 8765
                async with (
                    ClientSession() as client,
                    client.ws_connect(
                        f"http://{url.netloc}/api/v1/session?{url.query}",
                        headers={"Origin": f"http://{url.netloc}"},
                    ) as ws,
                ):
                    frame = await ws.receive_json(timeout=10)
                    assert frame["playing"] is False
                    assert _same_time(
                        frame["scene"]["frame"]["time"],
                        0.2 if identifier == "live-resume" else 0.0,
                    )
                    if identifier == "live-trap":
                        await ws.send_json({"type": "checkpoint"})
                        saved = await ws.receive_json(timeout=10)
                        assert saved["type"] == "checkpoint"
                        _document(shell.cwd / "results/tutorial runs/live-trap.json")
                    await ws.send_json({"type": "stop"})
                    while True:
                        message = await ws.receive(timeout=15)
                        if message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                            break
                        if message.type == WSMsgType.TEXT:
                            stopped |= message.json() == {"type": "session", "state": "stopped"}
                stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
                shell.records.append(
                    {
                        "id": identifier,
                        "command": BLOCKS[identifier],
                        "exit_code": process.returncode,
                        "stopped": stopped,
                        "stdout": stdout.decode(),
                        "stderr": stderr.decode(),
                    }
                )
                assert stopped
                assert process.returncode == 0, stderr.decode()
            finally:
                if process.returncode is None:
                    await asyncio.to_thread(_kill_tree, process.pid)
                    await asyncio.wait_for(process.communicate(), 10)

    asyncio.run(exercise())


def test_tutorial_flags_paths_and_links() -> None:
    documents = [
        *sorted((ROOT / "docs/tutorials").glob("*.md")),
        ROOT / "environments/metal/README.md",
        ROOT / "environments/cuda/README.md",
        ROOT / "viewer/README.md",
        ROOT / "docs/development/tutorial-command-validation.md",
    ]
    parser = _parser()
    checked = 0
    for path in documents:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if re.match(r"[a-z]+://", target):
                continue
            filename, _, anchor = target.partition("#")
            destination = (path.parent / filename).resolve() if filename else path
            assert destination.exists(), (path, target)
            if anchor and destination.suffix == ".md":
                headings = re.findall(r"^#+\s+(.+)$", destination.read_text(), re.MULTILINE)
                anchors = {re.sub(r"[^\w -]", "", h.lower()).replace(" ", "-") for h in headings}
                assert anchor in anchors, (path, target)
        if path.parent == ROOT / "docs/tutorials" and path != GUIDE:
            assert "commands.md#" in text, path
        for language, block in re.findall(r"```(console|sh)\n(.*?)\n```", text, re.DOTALL):
            del language
            for line in block.replace("\\\n", " ").splitlines():
                parts = shlex.split(line)
                if "microsimulator" not in parts or "--help" in parts:
                    continue
                # Exclude prose or output; only parse literal CLI invocations.
                if parts[:2] != ["uv", "run"]:
                    continue
                arguments = parser.parse_args(parts[parts.index("microsimulator") + 1 :])
                if (model := getattr(arguments, "model", None)) is not None:
                    model = cast(Path, model)
                    if model != Path("results/tutorial models/biophysics.py"):
                        assert (ROOT / model).is_file(), (path, model)
                checked += 1
    assert checked >= 35


def test_failed_command_cleanup_drains_descendant_pipes() -> None:
    # Regression for a failed live assertion hanging after only its shell died.
    code = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "print('ready', flush=True); time.sleep(60)"
    )
    with subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=sys.platform != "win32",
    ) as process:
        try:
            assert process.stdout is not None
            assert process.stdout.readline().strip() == "ready"
        finally:
            _kill_tree(process.pid)
        process.communicate(timeout=10)
        assert process.returncode is not None
