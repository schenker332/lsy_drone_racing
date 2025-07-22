#!/usr/bin/env python3
"""
sweep_mpc.py

Schreibt unter sweeps/<Param1>_<Param2>/sweep.csv:
  | Param1 | Param2 | pass_rate | avg_full_pass_time | total_gates_passed |

Beispiel:
  4.0, 0.10, 3/5, 2.31, 13
"""
import os
import itertools
import csv
from pathlib import Path

import numpy as np
import fire

from scripts.sim import simulate


def sweep(
    param_ranges: list[str] = [
        "t_scaling:4.3:5.3:5",  # t_scaling from 4.0 to 6.0 in 5 steps
        "alpha_curv_speed:0.05:0.15:5",  #

    ],
    runs_per_val: int      = 50,
    config: str            = "level2.toml",
    controller: str | None = None,
    gui: bool | None       = False,
):
    # parse param_ranges → dict[name] = np.linspace(...)
    ranges: dict[str, np.ndarray] = {}
    for pr in param_ranges:
        name, start, stop, steps = pr.split(":")
        ranges[name] = np.linspace(float(start), float(stop), int(steps))

    keys   = list(ranges.keys())
    combos = itertools.product(*(ranges[k] for k in keys))

    # prepare output folder & csv
    sweep_dir = Path("sweeps") / "_".join(keys)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    out_csv   = sweep_dir / "sweep.csv"

    # header
    header = keys + ["pass_rate", "avg_full_pass_time", "total_gates_passed"]

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        f.flush(); os.fsync(f.fileno())

        # sweep loop
        for combo in combos:
            overrides = {keys[i]: float(combo[i]) for i in range(len(keys))}
            print(f"→ Sweeping with {overrides}")

            # simulate returns List[{"run", "gates_passed", "time", ...}, ...]
            all_stats = simulate(
                config            = config,
                controller        = controller,
                n_runs            = runs_per_val,
                gui               = gui,
                param_overrides   = overrides,
                return_stats      = True,
            )
            # take only the last runs_per_val entries
            run_stats = all_stats[-runs_per_val:]

            # compute metrics
            total_gates_passed = sum(r["gates_passed"] for r in run_stats)
            max_g = max(r["gates_passed"] for r in run_stats)
            full_runs = [r for r in run_stats if r["gates_passed"] == max_g]
            num_full = len(full_runs)
            pass_rate = f"{num_full}/{runs_per_val}"
            if num_full:
                avg_full = float(f"{np.mean([r['time'] for r in full_runs]):.4f}")
            else:
                avg_full = float("nan")

            # build row
            row = [overrides[k] for k in keys] + [pass_rate, avg_full, total_gates_passed]

            # write & flush immediately
            writer.writerow(row)
            f.flush(); os.fsync(f.fileno())
            print(f"   ✅ Wrote: {row}")

    print(f"\n✅ Sweep done → {out_csv}")


if __name__ == "__main__":
    fire.Fire(sweep)
