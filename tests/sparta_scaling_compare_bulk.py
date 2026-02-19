#!/usr/bin/env python3
"""
Compare bulk-normalized SpaRTA scaling between Python (DNS) and OpenFOAM runtime.

Expected CSV schema for both inputs:
    feature,min,max,avg

OpenFOAM input can also be provided as a solver log. In that case this script
parses lines of the form:
    SpaRTA scaling: name=<feature> min=<...> max=<...> avg=<...>
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


SCALING_LINE_PATTERN = re.compile(
    r"SpaRTA scaling:\s+name=(?P<name>\S+)\s+"
    r"min=(?P<min>[-+0-9.eE]+)\s+"
    r"max=(?P<max>[-+0-9.eE]+)\s+"
    r"avg=(?P<avg>[-+0-9.eE]+)"
)

DEFAULT_PRIORITY_FEATURES = [
    "U_b",
    "I1_norm",
    "I2_norm",
    "T1_norm:gradU_norm",
    "T2_norm:gradU_norm",
    "T3_norm:gradU_norm",
]


def _parse_openfoam_log(log_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, start=1):
            match = SCALING_LINE_PATTERN.search(line)
            if not match:
                continue
            rows.append(
                {
                    "feature": match.group("name").strip('"'),
                    "min": float(match.group("min")),
                    "max": float(match.group("max")),
                    "avg": float(match.group("avg")),
                    "line_no": line_no,
                }
            )

    if not rows:
        raise ValueError(f"No SpaRTA scaling diagnostics found in log: {log_path}")

    df = pd.DataFrame(rows)
    return (
        df.sort_values("line_no")
        .drop_duplicates(subset=["feature"], keep="last")
        .drop(columns=["line_no"])
        .sort_values("feature")
    )


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return math.nan
    return numerator / denominator


def _scale(min_val: float, max_val: float) -> float:
    return max(abs(min_val), abs(max_val))


def _feature_warning(
    ratio_scale_abs: float,
    warn_ratio: float,
    severe_ratio: float,
    oom_low: float,
    oom_high: float,
) -> str:
    if math.isnan(ratio_scale_abs):
        return "NA"
    if ratio_scale_abs >= severe_ratio:
        return f"SEVERE(>={severe_ratio}x)"
    if ratio_scale_abs >= warn_ratio:
        return f"WARN(>={warn_ratio}x)"
    if ratio_scale_abs <= 1.0 / severe_ratio:
        return f"SEVERE(<={1.0/severe_ratio:.3g}x)"
    if ratio_scale_abs <= 1.0 / warn_ratio:
        return f"WARN(<={1.0/warn_ratio:.3g}x)"
    if ratio_scale_abs >= oom_high or ratio_scale_abs <= oom_low:
        return f"OOM({oom_low:g}-{oom_high:g})"
    return "OK"


def compare_stats(
    dns_df: pd.DataFrame,
    openfoam_df: pd.DataFrame,
    features: Optional[List[str]],
    warn_ratio: float,
    severe_ratio: float,
    oom_low: float,
    oom_high: float,
) -> pd.DataFrame:
    dns_idx = dns_df.set_index("feature")
    of_idx = openfoam_df.set_index("feature")

    common_features = sorted(set(dns_idx.index).intersection(of_idx.index))
    if features:
        selected = [name for name in features if name in common_features]
    else:
        selected = common_features
    if not selected:
        raise ValueError("No overlapping features found for comparison.")

    rows = []
    for feature in selected:
        dns_min = float(dns_idx.loc[feature, "min"])
        dns_max = float(dns_idx.loc[feature, "max"])
        dns_avg = float(dns_idx.loc[feature, "avg"])
        of_min = float(of_idx.loc[feature, "min"])
        of_max = float(of_idx.loc[feature, "max"])
        of_avg = float(of_idx.loc[feature, "avg"])

        dns_scale = _scale(dns_min, dns_max)
        of_scale = _scale(of_min, of_max)

        ratio_max = _safe_div(of_max, dns_max)
        ratio_scale = _safe_div(of_scale, dns_scale)
        ratio_scale_abs = abs(ratio_scale) if not math.isnan(ratio_scale) else math.nan

        warning = _feature_warning(
            ratio_scale_abs=ratio_scale_abs,
            warn_ratio=warn_ratio,
            severe_ratio=severe_ratio,
            oom_low=oom_low,
            oom_high=oom_high,
        )

        rows.append(
            {
                "feature": feature,
                "dns_min": dns_min,
                "dns_max": dns_max,
                "dns_avg": dns_avg,
                "openfoam_min": of_min,
                "openfoam_max": of_max,
                "openfoam_avg": of_avg,
                "ratio_max": ratio_max,
                "dns_scale": dns_scale,
                "openfoam_scale": of_scale,
                "ratio_scale": ratio_scale,
                "warning": warning,
            }
        )

    return pd.DataFrame(rows)


def print_summary_report(compare_df: pd.DataFrame) -> None:
    print("SpaRTA bulk-scaling comparison summary")
    ub_rows = compare_df[compare_df["feature"] == "U_b"]
    if not ub_rows.empty:
        ub = ub_rows.iloc[0]
        ub_abs_diff = abs(float(ub["openfoam_avg"]) - float(ub["dns_avg"]))
        ub_rel_diff = ub_abs_diff / max(abs(float(ub["dns_avg"])), 1e-30)
        print(
            "U_b comparison: "
            f"DNS={float(ub['dns_avg']):.6e}, "
            f"OpenFOAM={float(ub['openfoam_avg']):.6e}, "
            f"abs_diff={ub_abs_diff:.6e}, rel_diff={ub_rel_diff:.6e}"
        )

    for _, row in compare_df.iterrows():
        print(
            f"Feature {row['feature']} in Python is O({row['dns_scale']:.3e}) "
            f"but in OpenFOAM is O({row['openfoam_scale']:.3e}); "
            f"ratio = {row['ratio_scale']:.3e} [{row['warning']}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DNS vs OpenFOAM bulk-normalized SpaRTA scaling."
    )
    parser.add_argument(
        "--python-stats-csv",
        type=Path,
        required=True,
        help="CSV from tests/sparta_scaling_python_stats.py",
    )
    parser.add_argument(
        "--openfoam-stats-csv",
        type=Path,
        default=None,
        help="CSV from tests/sparta_scaling_openfoam_stats.py",
    )
    parser.add_argument(
        "--openfoam-log",
        type=Path,
        default=None,
        help="OpenFOAM solver log containing SpaRTA scaling lines.",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional explicit feature list. If omitted, all common features are compared.",
    )
    parser.add_argument(
        "--warn-ratio",
        type=float,
        default=5.0,
        help="Warn when scale ratio magnitude is above this multiplier (or below reciprocal).",
    )
    parser.add_argument(
        "--severe-ratio",
        type=float,
        default=10.0,
        help="Severe warning when scale ratio magnitude is above this multiplier (or below reciprocal).",
    )
    parser.add_argument(
        "--oom-low",
        type=float,
        default=1e-3,
        help="Lower order-of-magnitude threshold.",
    )
    parser.add_argument(
        "--oom-high",
        type=float,
        default=1e3,
        help="Upper order-of-magnitude threshold.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("output/sparta_scaling_compare_bulk.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if args.openfoam_stats_csv is None and args.openfoam_log is None:
        raise ValueError("Provide either --openfoam-stats-csv or --openfoam-log.")

    dns_df = pd.read_csv(args.python_stats_csv)
    if args.openfoam_stats_csv is not None:
        of_df = pd.read_csv(args.openfoam_stats_csv)
    else:
        assert args.openfoam_log is not None
        of_df = _parse_openfoam_log(args.openfoam_log)

    requested_features = args.features
    if requested_features is None:
        requested_features = DEFAULT_PRIORITY_FEATURES.copy()

    compare_df = compare_stats(
        dns_df=dns_df,
        openfoam_df=of_df,
        features=requested_features,
        warn_ratio=args.warn_ratio,
        severe_ratio=args.severe_ratio,
        oom_low=args.oom_low,
        oom_high=args.oom_high,
    )
    print_summary_report(compare_df=compare_df)

    output_csv = args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    compare_df.to_csv(output_csv, index=False)
    print(f"Wrote bulk scaling comparison CSV: {output_csv}")


if __name__ == "__main__":
    main()
