"""Physical hydraulic units and port constraints, independent of signal species."""

import math

import microsimulator as ms
import numpy as np
import pytest


def duct(n: int = 4, length: int = 8) -> ms.FluidGridSpec:
    grid = ms.FluidGridSpec()
    grid.shape.x, grid.shape.y, grid.shape.z = n, length, n

    return grid


def pressure_ports() -> list[ms.FlowPort]:
    inlet, outlet = ms.FlowPort(), ms.FlowPort()
    inlet.name, inlet.value = "feed", 0.01
    outlet.name, outlet.upper = "drain", True

    return [inlet, outlet]


def test_physical_flow_python_api() -> None:
    grid = duct()
    result = ms.StokesFlowSolver().solve(grid, ms.FluidProperties(), pressure_ports())
    assert grid.site_count == 128
    assert len(result.pressure_pa) == grid.site_count
    assert len(result.field.y_faces) == 4 * 9 * 4
    assert result.ports[0].flow_rate_m3_s < 0 < result.ports[1].flow_rate_m3_s
    assert result.ports[0].pressure_pa == 0.01
    assert math.isclose(result.ports[0].area_m2, 16e-12)
    assert result.report.relative_residual <= 1e-5
    assert result.report.reynolds_number < 1


def test_unit_conversion_does_not_change_physics() -> None:
    solver, medium = ms.StokesFlowSolver(), ms.FluidProperties()
    grid = duct()
    a = solver.solve(grid, medium, pressure_ports())
    grid.spacing = 2
    grid.length_unit_m = 0.5e-6
    grid.time_unit_s = 3
    b = solver.solve(grid, medium, pressure_ports())
    np.testing.assert_allclose(b.pressure_pa, a.pressure_pa, atol=1e-8)
    np.testing.assert_allclose(b.field.y_faces, np.array(a.field.y_faces) * 6, rtol=2e-5)
    assert math.isclose(b.ports[0].flow_rate_m3_s, a.ports[0].flow_rate_m3_s, rel_tol=1e-5)


def test_disconnected_components_have_independent_pressure_gauges() -> None:
    grid = duct(5)
    grid.obstacles = [int(x == 2) for x in range(5) for _ in range(8 * 5)]
    ports: list[ms.FlowPort] = []

    for first_x, offset in [(0, 0), (3, 100)]:
        for upper in (False, True):
            p = ms.FlowPort()
            p.name = f"{first_x}-{upper}"
            p.upper = upper
            p.value = offset + (0 if upper else 0.01)
            p.sites = [
                x * 40 + (35 if upper else 0) + z
                for x in range(first_x, first_x + 2)
                for z in range(5)
            ]
            ports.append(p)

    parameters = ms.LinearSolveParameters()
    parameters.relative_tolerance, parameters.absolute_tolerance = 1e-10, 1e-12
    result = ms.StokesFlowSolver().solve(grid, ms.FluidProperties(), ports, parameters)
    assert math.isclose(
        result.ports[0].flow_rate_m3_s, result.ports[2].flow_rate_m3_s, rel_tol=1e-8
    )
    assert math.isclose(
        result.ports[1].flow_rate_m3_s, result.ports[3].flow_rate_m3_s, rel_tol=1e-8
    )

    with pytest.raises(ValueError, match="connected fluid component"):
        ms.StokesFlowSolver().solve(grid, ms.FluidProperties(), pressure_ports())


def test_square_duct_poiseuille_refinement() -> None:
    # Fully developed square-duct solution: Q = C * G * width**4 / mu.
    # C follows by integrating the sine-series solution of -laplacian(u)=G/mu.
    # Measure the central pressure gradient to exclude traction-port entrance effects.
    coefficient = (
        64
        / math.pi**6
        * sum(
            1 / (m * m * n * n * (m * m + n * n))
            for m in range(1, 100, 2)
            for n in range(1, 100, 2)
        )
    )
    errors: list[float] = []

    for n in (4, 8):
        grid = duct(n, 6 * n)
        grid.spacing = 4 / n
        parameters = ms.LinearSolveParameters()
        parameters.relative_tolerance, parameters.absolute_tolerance = 1e-8, 1e-10
        result = ms.StokesFlowSolver().solve(
            grid, ms.FluidProperties(), pressure_ports(), parameters
        )
        pressure = np.array(result.pressure_pa).reshape(n, 6 * n, n)
        gradient = float((pressure[:, 2 * n, :].mean() - pressure[:, 4 * n, :].mean()) / 8e-6)
        expected = coefficient * gradient * (4e-6) ** 4 / 1e-3
        errors.append(abs(result.ports[1].flow_rate_m3_s / expected - 1))

    assert errors[1] < 0.1
    assert errors[1] < errors[0] / 2.5


def test_failure_does_not_contaminate_next_solve() -> None:
    solver = ms.StokesFlowSolver()
    parameters = ms.LinearSolveParameters()
    parameters.max_iterations = 1

    with pytest.raises(RuntimeError, match="converge"):
        solver.solve(duct(), ms.FluidProperties(), pressure_ports(), parameters)

    assert solver.solve(duct(), ms.FluidProperties(), pressure_ports()).ports[1].flow_rate_m3_s > 0


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_physical_units(value: float) -> None:
    grid = duct()
    grid.length_unit_m = value

    with pytest.raises(ValueError, match="unit scales"):
        ms.StokesFlowSolver().solve(grid, ms.FluidProperties(), pressure_ports())


def test_body_proposals_preserve_inputs_and_unit_quaternions() -> None:
    grid = duct(12, 12)
    body = ms.CapsuleBody()
    body.position = (5.5, 5.5, 5.5)
    body.radius = 2
    body.torque_nm = (0, 0, 1e-20)
    solver = ms.StokesFlowSolver()
    step = solver.propose_body_step(grid, ms.FluidProperties(), [], [body], 0.1)
    assert body.orientation == (1, 0, 0, 0)
    assert step.accepted_dt == 0.1
    assert step.bodies[0].orientation[3] > 0
    assert math.isclose(sum(q * q for q in step.bodies[0].orientation), 1, abs_tol=1e-12)
    assert step.flow.bodies[0].marker_count > 0
    assert math.isclose(body.geometric_volume, 32 * math.pi / 3)
    limits = ms.FluidBodyStepParameters()
    limits.max_halvings = 0

    with pytest.raises(RuntimeError, match="substep halvings"):
        solver.propose_body_step(
            grid, ms.FluidProperties(), [], [body], 1e4, step_parameters=limits
        )

    assert body.position == (5.5, 5.5, 5.5)
    assert body.orientation == (1, 0, 0, 0)


def test_body_resolution_and_overlap_are_explicit_errors() -> None:
    grid = duct(12, 12)
    body = ms.CapsuleBody()
    body.position = (5.5, 5.5, 5.5)

    with pytest.raises(ValueError, match="two grid spacings"):
        ms.StokesFlowSolver().solve_bodies(grid, ms.FluidProperties(), [], [body])

    body.radius = 2
    other = ms.CapsuleBody()
    other.id = 2
    other.position = body.position
    other.radius = body.radius

    with pytest.raises(ValueError, match="overlap"):
        ms.StokesFlowSolver().solve_bodies(grid, ms.FluidProperties(), [], [body, other])
