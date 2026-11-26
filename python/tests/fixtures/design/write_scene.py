"""Write the scene of each design beside this file, as the tests read them.

Run with an interpreter that has the `microfluidics` design tool installed:

    python python/tests/fixtures/design/write_scene.py

Each scene is written as the design tool gives it, less the fields that hold
paths into the source of the design and of the tool, which differ from one
machine to the next and which `microsimulator.design` does not read.

`mixer.py` is the mixer among the design tool's own example designs.
"""

import json
from pathlib import Path

from microfluidics.project import Project

HERE = Path(__file__).resolve().parent


def main() -> None:
    for design in ("trap_chamber", "mixer"):
        scene = Project(HERE / f"{design}.py").scene()
        del scene["origins"]
        for held in (*scene["instances"], *scene["findings"]):
            del held["origin"]
        text = json.dumps(scene, indent=2) + "\n"
        (HERE / f"{design}.scene.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
