"""Record stage-local Z motion without changing the 3D solver or its random stream.

Run in a dedicated diagnostic process: native Python entry points are temporarily
instrumented and restored on exit. No controller implementation is duplicated.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

from microsimulator import (
    BackendKind,
    CellInit,
    GridBoundary,
    GridBoundaryKind,
    GridShape,
    MechanicsConfig,
    ModelContext,
    NativeController,
    PlaneConstraintInit,
    SignalGridSpec,
    SignalGridVelocityField,
    Simulation,
    StepPlan,
    UniformLengthDivision,
    Vec3,
    backend_available,
    build_model,
)
from microsimulator.scene import capture_scene, dumps_scene


def cells(simulation):
    return [
        dict(
            id=c.id,
            center=[c.position.x, c.position.y, c.position.z],
            direction=[c.direction.x, c.direction.y, c.direction.z],
            length=c.length,
            radius=c.radius,
        )
        for c in simulation.cells()
    ]


class Trace:
    def __init__(self, simulation, plane_z=0.0, tolerance=1e-6):
        self.simulation = simulation
        self.plane_z = plane_z
        self.tolerance = tolerance
        self.events = []
        self.first_event = None
        self.initial = json.loads(dumps_scene(capture_scene(simulation)))["frame"]
        self.record("initialization", [], cells(simulation))

    def record(self, stage, before, after, requested_z=None):
        entry = dict(
            stage=stage,
            time=self.simulation.time,
            cells=len(after),
            max_center_z=max((abs(c["center"][2]) for c in after), default=0.0),
            max_center_displacement_from_plane=max(
                (abs(c["center"][2] - self.plane_z) for c in after), default=0.0
            ),
            max_direction_z=max((abs(c["direction"][2]) for c in after), default=0.0),
        )
        previous = {c["id"]: c for c in before}
        entry["max_stage_center_change_z"] = max(
            (
                abs(c["center"][2] - previous[c["id"]]["center"][2])
                for c in after
                if c["id"] in previous
            ),
            default=0.0,
        )
        entry["max_stage_direction_change_z"] = max(
            (
                abs(c["direction"][2] - previous[c["id"]]["direction"][2])
                for c in after
                if c["id"] in previous
            ),
            default=0.0,
        )
        if requested_z is not None:
            entry["requested_direction_delta_z"] = requested_z
        self.events.append(entry)
        if (
            self.first_event is None
            and max(entry["max_center_displacement_from_plane"], entry["max_direction_z"])
            > self.tolerance
        ):
            self.first_event = dict(**entry, before=before, after=after)

    @contextmanager
    def instrument(self):
        names = {
            "divide": "division",
            "divide_equal": "division",
            "remove_cell": "removal",
            "step": "growth_and_chemistry",
            "apply_flow_drift": "flow_drift",
            "relax_cell_mechanics": "contact_and_constraint_relaxation",
            "set_cell_geometry": "geometry_edit",
        }
        originals = {}

        def wrapped(name, original):
            def call(simulation, *args, **kwargs):
                if simulation is not self.simulation:
                    return original(simulation, *args, **kwargs)
                before = cells(simulation)
                requested_z = None
                if name == "set_cell_geometry":
                    cell_id = args[0] if args else kwargs.get("cell_id")
                    direction = args[2] if len(args) >= 3 else kwargs.get("direction")
                    if cell_id is not None and direction is not None:
                        previous = simulation.cell(cell_id)
                        requested_z = direction.z - previous.direction.z
                result = original(simulation, *args, **kwargs)
                self.record(names[name], before, cells(simulation), requested_z)
                return result

            return call

        try:
            for name in names:
                originals[name] = getattr(Simulation, name)
                setattr(Simulation, name, wrapped(name, originals[name]))
            yield
        finally:
            for name, original in originals.items():
                setattr(Simulation, name, original)

    def report(self):
        return dict(
            plane_z=self.plane_z,
            tolerance=self.tolerance,
            initial_geometry=self.initial["cells"],
            constraints=self.initial["constraints"],
            backend=self.initial["backend"],
            first_out_of_plane=self.first_event,
            final_geometry=cells(self.simulation),
            stages=self.events,
        )


def add(simulation, position=(0, 0, 0), direction=(1, 0, 0), length=4.0, radius=0.5):
    cell = CellInit()
    cell.position, cell.direction = Vec3(*position), Vec3(*direction)
    cell.length, cell.radius, cell.growth_rate = length, radius, 0.0
    return simulation.add_cell(cell)


def fixtures(backend, seed, dt):
    results = {}
    for name in ("separated_planar", "crossing", "coincident_parallel"):
        simulation = Simulation(backend)
        add(simulation)
        add(
            simulation,
            position=(0, 3, 0) if name == "separated_planar" else (0, 0, 0),
            direction=(0, 1, 0) if name == "crossing" else (1, 0, 0),
        )
        normals = [
            [c.normal.x, c.normal.y, c.normal.z] for c in simulation.find_cell_contacts().contacts
        ]
        trace = Trace(simulation)
        with trace.instrument():
            simulation.relax_cell_mechanics(*MechanicsConfig().native_parameters())
        results[name] = dict(
            classification="expected 3D contact behavior",
            mechanics=MechanicsConfig().to_json(),
            contact_normals=normals,
            **trace.report(),
        )
    for name, direction in (("planar_division", (1, 0, 0)), ("inherited_tilt", (1, 0, 0.2))):
        simulation = Simulation(backend)
        founder = add(simulation, direction=direction)
        rng = random.Random(seed)
        policy = UniformLengthDivision(3.0, 3.0, jitter_z=False)
        state = {}
        policy.initialize(state, rng, (founder,))  # intentionally oversized diagnostic founder
        controller = NativeController(
            simulation,
            model_id="planarity-diagnostic",
            model_version=1,
            rng=rng,
            state=state,
            regulate=lambda step, policy=policy: StepPlan(divisions=policy.requests(step)),
            on_division=policy.on_division,
        )
        trace = Trace(simulation)
        with trace.instrument():
            controller.step(0.0)
        results[name] = dict(
            classification="expected inherited geometry and normalized XY jitter",
            controller_step_dt=0.0,
            division_target=3.0,
            jitter_z=False,
            mechanics=MechanicsConfig().to_json(),
            **trace.report(),
        )
    simulation = Simulation(backend)
    add(simulation, position=(0, 0, 0.8))
    for height, normal in ((-1.0, 1.0), (1.0, -1.0)):
        plane = PlaneConstraintInit()
        plane.point, plane.inward_normal = Vec3(0, 0, height), Vec3(0, 0, normal)
        simulation.add_plane_constraint(plane)
    trace = Trace(simulation, plane_z=0.8)

    def relax_to_tolerance(config):
        residuals = []
        for _ in range(20):
            residuals.append(
                max(
                    (
                        max(0.0, -contact.signed_separation)
                        for contact in simulation.find_external_contacts().contacts
                    ),
                    default=0.0,
                )
            )
            if residuals[-1] <= 1e-6 or (len(residuals) > 1 and residuals[-1] == residuals[-2]):
                break
            simulation.relax_cell_mechanics(*config.native_parameters())
        return residuals

    default_config = MechanicsConfig()
    tight_config = MechanicsConfig(residual_rms_tolerance=1e-8)
    with trace.instrument():
        default_residuals = relax_to_tolerance(default_config)
        default_geometry = cells(simulation)
        tight_residuals = relax_to_tolerance(tight_config)
    results["finite_height_constraints"] = dict(
        classification="expected 3D wall relaxation; finite height is not strict 2D",
        default_mechanics=default_config.to_json(),
        tight_mechanics=tight_config.to_json(),
        default_wall_penetration_by_pass=default_residuals,
        tight_wall_penetration_by_pass=tight_residuals,
        default_final_geometry=default_geometry,
        **trace.report(),
    )
    simulation = Simulation(backend)
    shape = GridShape()
    shape.x, shape.y, shape.z = 1, 1, 3
    spec = SignalGridSpec()
    spec.shape, spec.signal_count, spec.diffusion, spec.advection = shape, 1, [0.0], [Vec3()]
    boundary = GridBoundary()
    boundary.kind, boundary.values = GridBoundaryKind.FIXED, [0.0]
    spec.z_lower, spec.z_upper = boundary, boundary
    field = SignalGridVelocityField()
    field.x_faces, field.y_faces, field.z_faces = [0.0] * 6, [0.0] * 6, [0.2] * 4
    spec.velocity_field = field
    simulation.configure_signal_grid(spec)
    add(simulation, position=(0, 0, 1), length=1.0)
    trace = Trace(simulation, plane_z=1.0)
    with trace.instrument():
        simulation.apply_flow_drift(dt)
    results["vertical_flow"] = dict(
        classification="expected prescribed 3D advection",
        prescribed_velocity=[0.0, 0.0, 0.2],
        drift_dt=dt,
        **trace.report(),
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--scenario", help="Convenience alias for a JSON string scenario parameter")
    parser.add_argument("--parameter", action="append", default=[], metavar="NAME=JSON")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max-cells", type=int, default=128)
    parser.add_argument("--plane-z", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.dt) or args.dt <= 0 or args.steps < 1 or args.max_cells < 1:
        parser.error("dt, steps and max-cells must be positive and finite")
    if not math.isfinite(args.plane_z):
        parser.error("plane-z must be finite")
    if not args.model and (args.scenario is not None or args.parameter):
        parser.error("scenario and parameters require a model")
    backend = getattr(BackendKind, args.backend.upper())
    if not backend_available(backend):
        parser.error(f"{args.backend} backend unavailable; no fallback performed")
    result = dict(
        diagnostic_version=1,
        seed=args.seed,
        dt=args.dt,
        backend=args.backend,
        fixtures=fixtures(backend, args.seed, args.dt),
    )
    if args.model:
        parameters = {}
        for parameter in args.parameter:
            try:
                key, value = parameter.split("=", 1)
                if not key or key in parameters:
                    raise ValueError("parameter names must be nonempty and unique")
                parameters[key] = json.loads(value)
            except (ValueError, json.JSONDecodeError) as error:
                parser.error(f"invalid parameter {parameter!r}: {error}")
        if args.scenario is not None:
            if "scenario" in parameters:
                parser.error("provide scenario once, using --scenario or --parameter")
            parameters["scenario"] = args.scenario
        model, provenance = build_model(
            args.model,
            ModelContext(backend, 0, seed=args.seed, parameters=parameters),
        )
        trace = Trace(model.simulation, args.plane_z)
        completed = 0
        with trace.instrument():
            for _ in range(args.steps):
                if model.simulation.cell_count >= args.max_cells:
                    break
                model.step(args.dt)
                completed += 1
        result["tutorial"] = dict(
            model=str(args.model),
            parameters=parameters,
            model_state=model.controller_state().get("model"),
            controller_kind=model.controller_state().get("kind"),
            mechanics=model.controller_state().get("mechanics"),
            completed_steps=completed,
            stop_reason="max_cells" if model.simulation.cell_count >= args.max_cells else "steps",
            provenance=provenance,
            requested_steps=args.steps,
            max_cells=args.max_cells,
            **trace.report(),
        )
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
