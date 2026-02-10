#!/usr/bin/env python3
"""Generate LSWT fit snapshots for active-learning points.

This utility loads an active-learning `exploration_history.json`, re-runs the
LSWT screener on each point, and writes:
  - a CSV summary with raw parameters and R² metrics
  - a handful of PNG "snapshots" (exp vs LSWT dispersion) for selected points

Example:
  python util/classical_solvers/lswt_fit_snapshots.py \
    --history util/classical_solvers/test_variable_j1xy/exploration_history.json \
    --out util/classical_solvers/test_variable_j1xy/lswt_snapshots \
    --r2 0.7 --r2-lower 0.75 --n 6
"""

from __future__ import annotations

import argparse
import json
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from lswt_screener import LSWTScreener, LSWTConfig
from feature_extractor import NormalizedParameters


def _load_history(history_path: Path) -> Dict[str, Any]:
    with history_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _params_from_dict(d: Dict[str, Any]) -> NormalizedParameters:
    # History stores NormalizedParameters fields.
    return NormalizedParameters.from_dict(d)


def _format_J(J: List[float]) -> str:
    names = ["J1xy", "J1z", "D", "E", "F", "G", "J3xy", "J3z"]
    return ", ".join([f"{n}={v:.4f}" for n, v in zip(names, J)])


def _snapshot_plot(
    out_png: Path,
    screener: LSWTScreener,
    J: List[float],
    point_id: int,
    phase: str,
    r2_total: float,
    r2_lower: float,
    thresholds: Tuple[float, float],
):
    params = screener.base_params.copy()
    J1xy, J1z, D, E, F, G, J3xy, J3z = J
    params.update({
        "J1": J1xy,
        "J1z": J1z,
        "J3": J3xy,
        "J3z": J3z,
        "D": D,
        "E": E,
        "F": F,
        "G": G,
    })

    energies_lower, energies_upper = screener._compute_dispersion(
        params, screener.config.MAGNETIC_FIELD, screener.config.DIRECTION
    )

    exp_lower = screener.data_lower[2]
    exp_upper = screener.data_upper[2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax = axes[0]
    ax.plot(exp_lower, "k.", ms=4, label="Exp (lower)")
    ax.plot(energies_lower, "r-", lw=1.2, label="LSWT (lower)")
    ax.set_title("Lower band")
    ax.set_xlabel("Data index")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(exp_upper, "k.", ms=4, label="Exp (upper)")
    ax.plot(energies_upper, "b-", lw=1.2, label="LSWT (upper)")
    ax.set_title("Upper band")
    ax.set_xlabel("Data index")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    thr_total, thr_lower = thresholds
    fig.suptitle(
        f"Point {point_id} | {phase} | R²={r2_total:.3f} (thr {thr_total}) | "
        f"R²_lower={r2_lower:.3f} (thr {thr_lower})\n{_format_J(J)}",
        fontsize=10,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=Path, required=True, help="Path to exploration_history.json")
    ap.add_argument("--out", type=Path, required=True, help="Output directory")
    ap.add_argument("--r2", type=float, default=0.7, help="R² total threshold")
    ap.add_argument("--r2-lower", type=float, default=0.75, help="R² lower threshold")
    ap.add_argument("--n", type=int, default=6, help="Total number of snapshots to write")
    args = ap.parse_args()

    history = _load_history(args.history)
    points = history.get("points", [])
    if not points:
        raise SystemExit(f"No points found in {args.history}")

    config = LSWTConfig(R2_THRESHOLD=args.r2, R2_LOWER_THRESHOLD=args.r2_lower)
    screener = LSWTScreener(config=config, verbose=False)

    rows: List[Dict[str, Any]] = []
    counts = {
        "passed": 0,
        "failed": 0,
        "numerical": 0,
    }

    for p in points:
        point_id = int(p.get("point_id"))
        phase = p.get("phase", "")
        params = _params_from_dict(p.get("params", {}))
        J = params.to_raw()

        result = screener.screen(params)

        rows.append({
            "point_id": point_id,
            "phase_recorded": phase,
            "lswt_passed": bool(result.passed),
            "r2_total": float(result.r2_total),
            "r2_lower": float(result.r2_lower),
            "lswt_reason": result.reason,
            "J1xy": J[0],
            "J1z": J[1],
            "D": J[2],
            "E": J[3],
            "F": J[4],
            "G": J[5],
            "J3xy": J[6],
            "J3z": J[7],
            "params_norm": json.dumps(asdict(params)),
        })

        if result.passed:
            counts["passed"] += 1
        else:
            counts["failed"] += 1
        if result.numerical_issue:
            counts["numerical"] += 1

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "lswt_screening_summary.csv"

    # Sort: passed first, then highest R²
    rows_sorted = sorted(
        rows,
        key=lambda r: (bool(r["lswt_passed"]), float(r["r2_total"])),
        reverse=True,
    )

    # Write CSV without pandas
    if rows_sorted:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
            writer.writeheader()
            writer.writerows(rows_sorted)

    # Pick snapshots: some passed (best) and some failed (borderline + worst)
    n = max(args.n, 1)
    n_pass = max(1, n // 3)
    n_fail = max(1, n - n_pass)

    pass_rows = [r for r in rows_sorted if r["lswt_passed"]]
    fail_rows = [r for r in rows_sorted if not r["lswt_passed"]]

    snap_rows: List[Dict[str, Any]] = []
    snap_rows.extend(pass_rows[:n_pass])

    # Fail snapshots: highest R² failures first (borderline), then lowest R² (worst)
    n_fail_hi = max(1, n_fail // 2)
    snap_rows.extend(fail_rows[:n_fail_hi])
    remaining = n_fail - n_fail_hi
    if remaining > 0 and len(fail_rows) > n_fail_hi:
        snap_rows.extend(fail_rows[-remaining:])

    snaps_dir = args.out / "snapshots"
    thresholds = (args.r2, args.r2_lower)

    for row in snap_rows[:n]:
        point_id = int(row["point_id"])
        phase = str(row["phase_recorded"])
        J = [
            float(row["J1xy"]), float(row["J1z"]), float(row["D"]), float(row["E"]),
            float(row["F"]), float(row["G"]), float(row["J3xy"]), float(row["J3z"]),
        ]
        out_png = snaps_dir / f"point_{point_id:04d}_lswt.png"
        _snapshot_plot(
            out_png=out_png,
            screener=screener,
            J=J,
            point_id=point_id,
            phase=phase,
            r2_total=float(row["r2_total"]),
            r2_lower=float(row["r2_lower"]),
            thresholds=thresholds,
        )

    # Write a short README
    readme = args.out / "README.txt"
    readme.write_text(
        "LSWT screening report\n"
        f"History: {args.history}\n"
        f"Thresholds: R2_total >= {args.r2}, R2_lower >= {args.r2_lower}\n"
        f"Total points: {len(rows_sorted)}\n"
        f"Passed: {counts['passed']}\n"
        f"Failed: {counts['failed']}\n"
        f"Numerical issues: {counts['numerical']}\n\n"
        "Outputs:\n"
        f"- {csv_path.name}: per-point params and R²\n"
        "- snapshots/: example plots\n",
        encoding="utf-8",
    )

    print(str(readme))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
