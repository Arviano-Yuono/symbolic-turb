#!/usr/bin/env python3
"""
Compute SpaRTA feature scaling statistics from Python-side data.

Supported inputs:
1) serialized FlowData snapshot (.pkl)
2) OpenFOAM case (through FOAMLoader)
3) DNS reference case (through KTHLoader)

Output CSV schema:
    feature,min,max,avg

Formula notes (matched to OpenFOAM diagnostics):
    S = sqrt(2 * symm(gradU):symm(gradU))
    tau = 1 / max(S/0.31 + omega_floor, omega + omega_floor)
    Sij = tau * dev(symm(gradU))
    Wij = tau * skew(gradU)
    I1 = tr(Sij & Sij), I2 = tr(Wij & Wij)
    U_b = <U_x>_A  (cross-section bulk velocity)
    gradU_norm = gradU / U_b
    S_norm = sqrt(2 * symm(gradU_norm):symm(gradU_norm))
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from symbolic_turb.core import FlowData, compute_basis_tensor, compute_invariants
from symbolic_turb.data import FOAMLoader, KTHLoader, Preprocessor


def _load_flow_data_from_pkl(path: Path) -> FlowData:
    flow_data = FlowData()
    return flow_data.load(str(path))


def _load_flow_data_from_foam(case_path: Path, time: str | None) -> FlowData:
    loader = FOAMLoader(
        data_path=str(case_path),
        flow_data=FlowData(),
        time=time,
        fields=["U", "omega"],
        sample_location="point",
        streamwise_average=False,
    )
    return loader.load()


def _load_flow_data_from_dns(case_path: Path) -> FlowData:
    return KTHLoader(
        data_path=str(case_path),
        flow_data=FlowData(),
    ).load()


def _ensure_grad_u(flow_data: FlowData) -> FlowData:
    if flow_data.gradU.size != 0:
        return flow_data
    return Preprocessor.compute_gradU(flow_data=flow_data)


def _stats(values: np.ndarray) -> Dict[str, float]:
    values_1d = np.asarray(values, dtype=float).reshape(-1)
    finite_mask = np.isfinite(values_1d)
    if not np.any(finite_mask):
        return {"min": float("nan"), "max": float("nan"), "avg": float("nan")}

    finite = values_1d[finite_mask]
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "avg": float(np.mean(finite)),
    }


def _constant_feature(value: float) -> np.ndarray:
    return np.array([value], dtype=float)


def _hydraulic_diameter_from_coords(coords: np.ndarray) -> float:
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        return float("nan")

    y_min = float(np.min(coords[:, 1]))
    y_max = float(np.max(coords[:, 1]))
    z_min = float(np.min(coords[:, 2]))
    z_max = float(np.max(coords[:, 2]))
    ly = y_max - y_min
    lz = z_max - z_min
    if ly <= 0.0 or lz <= 0.0:
        return float("nan")

    area = ly * lz
    perimeter = 2.0 * (ly + lz)
    if perimeter <= 0.0:
        return float("nan")
    return 4.0 * area / perimeter


def _resolve_nu(flow_data: FlowData, kinematic_viscosity: Optional[float]) -> float:
    if kinematic_viscosity is not None:
        return float(kinematic_viscosity)

    nu = np.asarray(flow_data.nu, dtype=float).reshape(-1)
    if nu.size == 0:
        return float("nan")

    finite_positive = nu[np.isfinite(nu) & (nu > 0.0)]
    if finite_positive.size == 0:
        return float("nan")

    return float(np.mean(finite_positive))


def _compute_re_bulk(
    flow_data: FlowData,
    ub: float,
    kinematic_viscosity: Optional[float],
) -> float:
    nu = _resolve_nu(flow_data=flow_data, kinematic_viscosity=kinematic_viscosity)
    dh = _hydraulic_diameter_from_coords(np.asarray(flow_data.coords, dtype=float))
    if not np.isfinite(nu) or not np.isfinite(dh) or nu <= 0.0:
        return float("nan")
    return float(ub * dh / nu)


def compute_scaling_features(
    flow_data: FlowData,
    omega_floor: float,
    yz_atol: float,
    ub_epsilon: float,
    kinematic_viscosity: Optional[float],
) -> Dict[str, np.ndarray]:
    grad_u = np.asarray(flow_data.gradU, dtype=float)
    if grad_u.ndim != 3 or grad_u.shape[1:] != (3, 3):
        raise ValueError(f"Expected gradU shape (N,3,3), got {grad_u.shape}")

    n_points = grad_u.shape[0]
    if n_points == 0:
        raise ValueError("gradU is empty")

    if flow_data.omega.size == 0:
        omega = np.ones((n_points,), dtype=float)
    else:
        omega = np.asarray(flow_data.omega, dtype=float).reshape(-1)
        if omega.shape[0] != n_points:
            raise ValueError(
                f"omega length ({omega.shape[0]}) does not match gradU N ({n_points})"
            )

    symm_grad_u = 0.5 * (grad_u + np.transpose(grad_u, (0, 2, 1)))
    skew_grad_u = 0.5 * (grad_u - np.transpose(grad_u, (0, 2, 1)))

    # OpenFOAM-style strain-rate magnitude:
    # S = sqrt(2 * symm(gradU):symm(gradU))
    s_mag = np.sqrt(2.0 * np.einsum("nij,nij->n", symm_grad_u, symm_grad_u))
    tau = 1.0 / np.maximum((s_mag / 0.31) + omega_floor, omega + omega_floor)

    tr_symm_grad_u = np.trace(symm_grad_u, axis1=1, axis2=2)
    dev_symm_grad_u = symm_grad_u - (
        (tr_symm_grad_u / 3.0)[:, None, None] * np.eye(3)[None, :, :]
    )

    # SpaRTA training-side tensors:
    # Sij = tau * dev(symm(gradU)), Wij = tau * skew(gradU)
    sij = tau[:, None, None] * dev_symm_grad_u
    wij = tau[:, None, None] * skew_grad_u

    i1, i2 = compute_invariants(Sij=sij, Wij=wij)
    t1, t2, t3 = compute_basis_tensor(Sij=sij, Wij=wij)

    ub = Preprocessor.compute_bulk_velocity(flow_data=flow_data, component=0, yz_atol=yz_atol)
    re_bulk = _compute_re_bulk(
        flow_data=flow_data,
        ub=ub,
        kinematic_viscosity=kinematic_viscosity,
    )

    grad_u_mag = np.sqrt(np.einsum("nij,nij->n", grad_u, grad_u))

    features: Dict[str, np.ndarray] = {
        "U_b": _constant_feature(ub),
        "Re_bulk": _constant_feature(re_bulk),
        "omega": omega,
        "Ux": np.asarray(flow_data.U[:, 0], dtype=float).reshape(-1),
        "Uy": np.asarray(flow_data.U[:, 1], dtype=float).reshape(-1),
        "Uz": np.asarray(flow_data.U[:, 2], dtype=float).reshape(-1),
        "gradU_mag": grad_u_mag,
        "S": s_mag,
        "tau": tau,
        "I1": np.asarray(i1, dtype=float).reshape(-1),
        "I2": np.asarray(i2, dtype=float).reshape(-1),
        "T1:T1": np.einsum("nij,nij->n", t1, t1),
        "T2:T2": np.einsum("nij,nij->n", t2, t2),
        "T3:T3": np.einsum("nij,nij->n", t3, t3),
        "T1:gradU": np.einsum("nij,nij->n", t1, grad_u),
        "T2:gradU": np.einsum("nij,nij->n", t2, grad_u),
        "T3:gradU": np.einsum("nij,nij->n", t3, grad_u),
    }

    if abs(ub) <= ub_epsilon:
        print(
            "SpaRTA python scaling warning: "
            f"|U_b|={abs(ub):.6e} <= ub_epsilon={ub_epsilon:.6e}; "
            "skipping U/U_b and gradU/U_b diagnostics."
        )
        return features

    _, u_norm, grad_u_norm = Preprocessor.compute_bulk_normalized_velocity_fields(
        flow_data=flow_data,
        ub=ub,
        ub_epsilon=ub_epsilon,
        yz_atol=yz_atol,
    )
    symm_grad_u_norm = 0.5 * (grad_u_norm + np.transpose(grad_u_norm, (0, 2, 1)))
    skew_grad_u_norm = 0.5 * (grad_u_norm - np.transpose(grad_u_norm, (0, 2, 1)))
    s_norm = np.sqrt(2.0 * np.einsum("nij,nij->n", symm_grad_u_norm, symm_grad_u_norm))

    i1_norm, i2_norm = compute_invariants(Sij=symm_grad_u_norm, Wij=skew_grad_u_norm)
    t1_norm, t2_norm, t3_norm = compute_basis_tensor(
        Sij=symm_grad_u_norm,
        Wij=skew_grad_u_norm,
    )

    features.update(
        {
            "Ux/U_b": u_norm[:, 0],
            "Uy/U_b": u_norm[:, 1],
            "Uz/U_b": u_norm[:, 2],
            "gradU_mag/U_b": grad_u_mag / abs(ub),
            "S_norm": s_norm,
            "I1_norm": np.asarray(i1_norm, dtype=float).reshape(-1),
            "I2_norm": np.asarray(i2_norm, dtype=float).reshape(-1),
            "T1_norm:T1_norm": np.einsum("nij,nij->n", t1_norm, t1_norm),
            "T2_norm:T2_norm": np.einsum("nij,nij->n", t2_norm, t2_norm),
            "T3_norm:T3_norm": np.einsum("nij,nij->n", t3_norm, t3_norm),
            "T1_norm:gradU_norm": np.einsum("nij,nij->n", t1_norm, grad_u_norm),
            "T2_norm:gradU_norm": np.einsum("nij,nij->n", t2_norm, grad_u_norm),
            "T3_norm:gradU_norm": np.einsum("nij,nij->n", t3_norm, grad_u_norm),
        }
    )
    return features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute SpaRTA scaling stats from Python-side data."
    )
    parser.add_argument(
        "--source",
        choices=("pkl", "foam", "dns"),
        default="pkl",
        help="Input source type.",
    )
    parser.add_argument(
        "--flow-pkl",
        type=Path,
        default=Path("tests/sparta.pkl"),
        help="Path to FlowData .pkl (used when --source pkl).",
    )
    parser.add_argument(
        "--case-path",
        type=Path,
        default=Path("dataset/test-baseline/AR_1_180_kOmegaSSTA"),
        help="OpenFOAM case path (used when --source foam).",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="0",
        help="OpenFOAM time directory to load (used when --source foam).",
    )
    parser.add_argument(
        "--dns-case-path",
        type=Path,
        default=Path("dataset/reference/AR_1_180"),
        help="DNS case path for KTHLoader (used when --source dns).",
    )
    parser.add_argument(
        "--omega-floor",
        type=float,
        default=1e-30,
        help="Lower bound used in tau = 1/max(omega, omega_floor).",
    )
    parser.add_argument(
        "--ub-epsilon",
        type=float,
        default=1e-12,
        help="Minimum |U_b| required for U/U_b and gradU/U_b diagnostics.",
    )
    parser.add_argument(
        "--yz-atol",
        type=float,
        default=1e-10,
        help="Tolerance used while grouping YZ points for U_b computation.",
    )
    parser.add_argument(
        "--kinematic-viscosity",
        type=float,
        default=None,
        help="Optional nu override for Re_bulk = U_b*D_h/nu.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("output/sparta_scaling_python.csv"),
        help="CSV path for computed statistics.",
    )

    args = parser.parse_args()

    print(
        "SpaRTA python scaling formulas: "
        "tau=1/max(S/0.31+omega_floor,omega+omega_floor), "
        "Sij=tau*dev(symm(gradU)), Wij=tau*skew(gradU), "
        "I1=tr(Sij&Sij), I2=tr(Wij&Wij), "
        "U_b=<U_x>_A, gradU_norm=gradU/U_b."
    )

    if args.source == "pkl":
        flow_data = _load_flow_data_from_pkl(args.flow_pkl)
    elif args.source == "foam":
        flow_data = _load_flow_data_from_foam(case_path=args.case_path, time=args.time)
    else:
        flow_data = _load_flow_data_from_dns(case_path=args.dns_case_path)

    flow_data = _ensure_grad_u(flow_data)
    features = compute_scaling_features(
        flow_data=flow_data,
        omega_floor=args.omega_floor,
        yz_atol=args.yz_atol,
        ub_epsilon=args.ub_epsilon,
        kinematic_viscosity=args.kinematic_viscosity,
    )

    rows = []
    for feature_name, values in features.items():
        stat = _stats(values)
        rows.append(
            {
                "feature": feature_name,
                "min": stat["min"],
                "max": stat["max"],
                "avg": stat["avg"],
            }
        )
        print(
            "SpaRTA python scaling: "
            f"name={feature_name} min={stat['min']:.6e} "
            f"max={stat['max']:.6e} avg={stat['avg']:.6e}"
        )

    output_csv = args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Wrote Python scaling CSV: {output_csv}")


if __name__ == "__main__":
    main()
