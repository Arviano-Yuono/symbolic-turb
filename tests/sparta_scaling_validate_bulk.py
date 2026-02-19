#!/usr/bin/env python3
"""
Lightweight validation of bulk-scaling diagnostics for a single case.

This script assumes you already generated:
1) Python stats CSV from tests/sparta_scaling_python_stats.py
2) OpenFOAM stats CSV from tests/sparta_scaling_openfoam_stats.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def _get_feature_avg(df: pd.DataFrame, name: str) -> float:
    rows = df[df["feature"] == name]
    if rows.empty:
        raise KeyError(f"Missing feature '{name}'")
    return float(rows.iloc[0]["avg"])


def _safe_rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate U_b and normalized-feature consistency for one case."
    )
    parser.add_argument("--python-stats-csv", type=Path, required=True)
    parser.add_argument("--openfoam-stats-csv", type=Path, required=True)
    parser.add_argument(
        "--ub-rel-tol",
        type=float,
        default=1e-3,
        help="Relative tolerance for U_b agreement.",
    )
    args = parser.parse_args()

    py_df = pd.read_csv(args.python_stats_csv)
    of_df = pd.read_csv(args.openfoam_stats_csv)

    py_ub = _get_feature_avg(py_df, "U_b")
    of_ub = _get_feature_avg(of_df, "U_b")
    ub_rel_diff = _safe_rel_diff(py_ub, of_ub)
    ub_ok = ub_rel_diff <= args.ub_rel_tol

    print(
        "Bulk velocity check: "
        f"U_b_python={py_ub:.6e}, U_b_openfoam={of_ub:.6e}, "
        f"relative_diff={ub_rel_diff:.6e}, tol={args.ub_rel_tol:.6e}, "
        f"status={'PASS' if ub_ok else 'FAIL'}"
    )

    for feature in ("I1_norm", "I2_norm", "T1_norm:gradU_norm"):
        try:
            py_val = _get_feature_avg(py_df, feature)
            of_val = _get_feature_avg(of_df, feature)
        except KeyError as exc:
            print(f"Qualitative check skipped: {exc}")
            continue

        if abs(py_val) < 1e-30:
            ratio = math.nan
        else:
            ratio = of_val / py_val

        print(
            "Qualitative normalized feature check: "
            f"{feature} python_avg={py_val:.6e}, openfoam_avg={of_val:.6e}, "
            f"ratio={ratio:.6e}"
        )

    if not ub_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
