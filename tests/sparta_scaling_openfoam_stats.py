#!/usr/bin/env python3
"""
Parse SpaRTA scaling diagnostics from an OpenFOAM log.

Expected line format (emitted by kOmegaSSTA when printSpaRTAScaling=true):
    SpaRTA scaling: name=<feature> min=<value> max=<value> avg=<value>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


SCALING_LINE_PATTERN = re.compile(
    r"SpaRTA scaling:\s+name=(?P<name>\S+)\s+"
    r"min=(?P<min>[-+0-9.eE]+)\s+"
    r"max=(?P<max>[-+0-9.eE]+)\s+"
    r"avg=(?P<avg>[-+0-9.eE]+)"
)


def parse_openfoam_scaling_log(log_path: Path) -> List[Dict[str, float]]:
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
                    "line_no": float(line_no),
                }
            )
    return rows


def select_occurrence(
    rows: List[Dict[str, float]],
    mode: str,
) -> pd.DataFrame:
    if not rows:
        raise ValueError("No SpaRTA scaling diagnostics found in the provided log.")

    df = pd.DataFrame(rows)
    if mode == "all":
        return df

    ascending = mode == "first"
    reduced = (
        df.sort_values("line_no", ascending=ascending)
        .drop_duplicates(subset=["feature"], keep="first")
        .sort_values("feature")
    )
    return reduced


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse SpaRTA scaling diagnostics from OpenFOAM logs."
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        required=True,
        help="Path to OpenFOAM solver log.",
    )
    parser.add_argument(
        "--occurrence",
        choices=("latest", "first", "all"),
        default="latest",
        help="Select which occurrence to keep for each feature.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("output/sparta_scaling_openfoam.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    rows = parse_openfoam_scaling_log(log_path=args.log_path)
    df = select_occurrence(rows=rows, mode=args.occurrence)

    output_csv = args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Parsed {len(rows)} scaling lines from log: {args.log_path}")
    print(f"Selected {len(df)} rows ({args.occurrence})")
    print(f"Wrote OpenFOAM scaling CSV: {output_csv}")

    for _, row in df.iterrows():
        print(
            "SpaRTA openfoam scaling: "
            f"name={row['feature']} min={row['min']:.6e} "
            f"max={row['max']:.6e} avg={row['avg']:.6e}"
        )


if __name__ == "__main__":
    main()
