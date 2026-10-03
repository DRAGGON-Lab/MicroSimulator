"""Solve pressure-driven medium flow without creating a biochemical signal grid."""

import argparse

import microsimulator as ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"), default="cpu")
    args = parser.parse_args()
    grid = ms.FluidGridSpec()
    grid.shape.x, grid.shape.y, grid.shape.z = 4, 16, 4
    grid.spacing = 1.0
    grid.length_unit_m = 1e-6
    grid.time_unit_s = 1.0
    feed, drain = ms.FlowPort(), ms.FlowPort()
    feed.name, feed.value = "feed", 0.01  # Pa
    drain.name, drain.upper = "drain", True
    backend = {
        "cpu": ms.BackendKind.CPU,
        "metal": ms.BackendKind.METAL,
        "cuda": ms.BackendKind.CUDA,
    }[args.backend]
    solver = ms.StokesFlowSolver(backend)
    result = solver.solve(grid, ms.FluidProperties(), [feed, drain])

    for port in result.ports:
        print(
            f"{port.name}: p={port.pressure_pa:.6g} Pa, Q={port.flow_rate_m3_s:.6g} m^3/s (outward)"
        )

    print(f"residual={result.report.relative_residual:.3g}, Re={result.report.reynolds_number:.3g}")


if __name__ == "__main__":
    main()
