"""Write the scene of the design beside this file, as the tests read it.

Run with an interpreter that has the `microfluidics` design tool installed:

    python python/tests/fixtures/design/write_scene.py

The scene is written as the design tool gives it, less the fields that hold
paths into the source of the design and of the tool, which differ from one
machine to the next and which `microsimulator.design` does not read.
"""

import json
from pathlib import Path

from microfluidics.project import Project

HERE = Path(__file__).resolve().parent


def main() -> None:
    scene = Project(HERE / "trap_chamber.py").scene()
    del scene["origins"]
    for held in (*scene["instances"], *scene["findings"]):
        del held["origin"]
    text = json.dumps(scene, indent=2) + "\n"
    (HERE / "trap_chamber.scene.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
