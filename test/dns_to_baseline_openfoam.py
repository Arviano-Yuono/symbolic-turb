import argparse
import os
import shutil
from typing import List, Optional, Tuple

import numpy as np

from symbolic_turb.core import FlowData
from symbolic_turb.data import (
    KTHLoader,
    build_reduced_target_from_reference,
    expand_fields_by_inverse_map,
    interpolate_fields_between_flows,
    read_flow_data_from_openfoam,
    write_flow_data_to_openfoam,
)


SUPPORTED_FIELDS = {"U", "k", "omega", "Rij", "tau_ij"}


def _parse_fields(raw: str) -> List[str]:
    fields = [f.strip() for f in raw.split(",") if f.strip()]
    if not fields:
        raise ValueError("No fields provided")

    unknown = [f for f in fields if f not in SUPPORTED_FIELDS]
    if unknown:
        raise ValueError(
            f"Unsupported fields: {unknown}. Supported fields: {sorted(SUPPORTED_FIELDS)}"
        )

    return fields


def _resolve_requested_fields(fields: List[str]) -> Tuple[List[str], bool, bool]:
    """
    Resolve field requests with aliases.

    Returns:
        interpolation_fields: fields needed from source interpolation (FlowData names)
        write_rij: whether to write standard OpenFOAM Rij field (default name mapping)
        write_tau_ij: whether to additionally write OpenFOAM field named tau_ij
    """
    interpolation_fields: List[str] = []
    write_rij = False
    write_tau_ij = False

    for field_name in fields:
        if field_name == "tau_ij":
            write_tau_ij = True
            if "Rij" not in interpolation_fields:
                interpolation_fields.append("Rij")
            continue

        if field_name == "Rij":
            write_rij = True
            if "Rij" not in interpolation_fields:
                interpolation_fields.append("Rij")
            continue

        interpolation_fields.append(field_name)

    return interpolation_fields, write_rij, write_tau_ij


def run_pipeline(
    dns_path: str,
    baseline_case_path: str,
    output_case_path: Optional[str],
    time: str,
    fields: List[str],
    interpolation_method: str,
    atol: float,
    create_if_missing: bool,
    template_time: str,
) -> None:
    interpolation_fields, write_rij, write_tau_ij = _resolve_requested_fields(fields)

    target_case_path = baseline_case_path
    if output_case_path:
        target_case_path = output_case_path
        base_abs = os.path.abspath(baseline_case_path)
        target_abs = os.path.abspath(target_case_path)
        if target_abs != base_abs and not os.path.exists(target_case_path):
            parent = os.path.dirname(target_case_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            shutil.copytree(baseline_case_path, target_case_path)
            print(f"Prepared output case copy: {baseline_case_path} -> {target_case_path}")

    print(f"[1/4] Loading DNS data from: {dns_path}")
    dns_flow = KTHLoader(data_path=dns_path, flow_data=FlowData()).load()

    print(f"[2/4] Reading baseline OpenFOAM case from: {target_case_path} (time={time})")
    baseline_3d = read_flow_data_from_openfoam(
        case_path=target_case_path,
        time=time,
        fields=[],  # only coordinates are needed to build interpolation target
    )

    x_ref = float(np.mean(baseline_3d.coords[:, 0]))
    target_2d, inverse_3d_to_2d = build_reduced_target_from_reference(
        reference_flow=baseline_3d,
        keep_dims=(1, 2),
        fill_dim_values={0: x_ref},
        atol=atol,
    )

    print(
        f"      baseline points: 3D={baseline_3d.coords.shape[0]}, "
        f"streamwise-avg 2D={target_2d.coords.shape[0]}"
    )

    print(f"[3/4] Interpolating source -> baseline 2D fields: {interpolation_fields}")
    mapped_2d = interpolate_fields_between_flows(
        source_flow=dns_flow,
        target_flow=target_2d,
        field_names=interpolation_fields,
        source_dims=(1, 2),
        target_dims=(1, 2),
        method=interpolation_method,
        fallback_method="nearest",
    )

    print("[4/4] Expanding mapped 2D fields back to 3D and writing to OpenFOAM")
    mapped_3d = expand_fields_by_inverse_map(
        reduced_flow=mapped_2d,
        full_reference_flow=baseline_3d,
        inverse_map=inverse_3d_to_2d,
        field_names=interpolation_fields,
    )

    default_write_fields = [f for f in interpolation_fields if f != "Rij"]
    if write_rij:
        default_write_fields.append("Rij")

    written_time = time
    if default_write_fields:
        written_time = write_flow_data_to_openfoam(
            case_path=target_case_path,
            flow_data=mapped_3d,
            time=time,
            fields=default_write_fields,
            atol=atol,
            create_if_missing=create_if_missing,
            template_time=template_time,
        )

    if write_tau_ij:
        written_time = write_flow_data_to_openfoam(
            case_path=target_case_path,
            flow_data=mapped_3d,
            time=time,
            fields=["Rij"],
            field_map={"Rij": "tau_ij"},
            atol=atol,
            create_if_missing=create_if_missing,
            template_time=template_time,
        )

    print(f"Done. Wrote fields {fields} to case='{target_case_path}', time='{written_time}'")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Map/interpolate source flow fields to streamwise-averaged baseline OpenFOAM mesh "
            "and write interpolated fields back to OpenFOAM."
        )
    )
    parser.add_argument(
        "--dns-path",
        default="dataset/reference/AR_1_180",
        help="Path to source dataset folder compatible with KTHLoader",
    )
    parser.add_argument(
        "--baseline-case-path",
        default="dataset/baseline/AR_1_180",
        help="Path to baseline OpenFOAM case directory",
    )
    parser.add_argument(
        "--output-case-path",
        default=None,
        help=(
            "Optional output case path. If set and it does not exist, the baseline case "
            "is copied there first, then fields are written to that output case."
        ),
    )
    parser.add_argument(
        "--time",
        default="3000",
        help="OpenFOAM time directory to read/write (e.g. 3000 or latest)",
    )
    parser.add_argument(
        "--fields",
        default="U,k",
        help="Comma-separated fields to interpolate/write. Supported: U,k,omega,Rij,tau_ij",
    )
    parser.add_argument(
        "--interpolation-method",
        default="linear",
        choices=["linear", "cubic", "nearest"],
        help="Interpolation method for scipy.griddata",
    )
    parser.add_argument(
        "--atol",
        default=1e-10,
        type=float,
        help="Coordinate tolerance for reduction and OpenFOAM coordinate mapping",
    )
    parser.add_argument(
        "--create-if-missing",
        action="store_true",
        help="Create missing target OpenFOAM field files using template fields",
    )
    parser.add_argument(
        "--template-time",
        default="0",
        help="Template time directory used when creating missing target field files",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    fields = _parse_fields(args.fields)

    if not os.path.isdir(args.dns_path):
        raise FileNotFoundError(f"Source dataset path does not exist: {args.dns_path}")
    if not os.path.isdir(args.baseline_case_path):
        raise FileNotFoundError(
            f"Baseline OpenFOAM case path does not exist: {args.baseline_case_path}"
        )
    if args.output_case_path and os.path.exists(args.output_case_path) and not os.path.isdir(
        args.output_case_path
    ):
        raise FileExistsError(
            f"Output case path exists but is not a directory: {args.output_case_path}"
        )

    run_pipeline(
        dns_path=args.dns_path,
        baseline_case_path=args.baseline_case_path,
        output_case_path=args.output_case_path,
        time=args.time,
        fields=fields,
        interpolation_method=args.interpolation_method,
        atol=args.atol,
        create_if_missing=args.create_if_missing,
        template_time=args.template_time,
    )


if __name__ == "__main__":
    main()
