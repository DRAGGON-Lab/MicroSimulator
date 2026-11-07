"""Run the seeded trap past the reported wall failure and check capsules."""

from __future__ import annotations

import argparse
from pathlib import Path

from microsimulator import BackendKind, ModelContext
from microsimulator.runner import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("cpu", "cuda"))
    parser.add_argument("--steps", type=int, default=12_310)
    args = parser.parse_args()
    backend = BackendKind.CPU if args.backend == "cpu" else BackendKind.CUDA
    model, _ = build_model(Path("examples/microfluidic_trap.py"), ModelContext(backend, 0, seed=42))
    simulation = model.simulation
    worst_gap = 0.0
    minimum_sample = float("inf")
    for step in range(1, args.steps + 1):
        try:
            model.step(0.001)
        except Exception as error:
            raise RuntimeError(f"step {step}: {error}") from error
        contacts = simulation.find_external_contacts()
        for contact in contacts.contacts:
            worst_gap = min(worst_gap, contact.signed_separation)
            if contact.signed_separation < -1.0e-4:
                raise AssertionError(
                    f"step {step}: cell {contact.cell_id} crosses wall "
                    f"{contact.constraint_id} by {-contact.signed_separation}"
                )
        for contact in simulation.find_cell_contacts().contacts:
            worst_gap = min(worst_gap, contact.signed_separation)
            if contact.signed_separation < -1.0e-4:
                raise AssertionError(
                    f"step {step}: cells {contact.first_id} and {contact.second_id} "
                    f"overlap by {-contact.signed_separation}"
                )
        for cell in simulation.cells():
            sample = simulation.sample_signals(cell.position)[0]
            minimum_sample = min(minimum_sample, sample)
        if step % 1000 == 0:
            print(
                f"{args.backend}: {step} steps, {simulation.cell_count} cells, "
                f"worst wall gap {worst_gap:.6g}, min sampled nutrient "
                f"{minimum_sample:.6g}",
                flush=True,
            )
    print(
        f"PASS {args.backend}: {args.steps} steps, worst gap {worst_gap:.6g}, "
        f"min nutrient {minimum_sample:.6g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
