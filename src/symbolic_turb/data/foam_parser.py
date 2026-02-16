"""
Utilities to read/write OpenFOAM fields using foamlib and FlowData.

This module is intentionally independent from loaders so it can be used as a
standalone bridge:
    - OpenFOAM -> FlowData
    - FlowData -> OpenFOAM
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from symbolic_turb.core import FlowData
from .coord_mapping import build_index_map_by_coords


DEFAULT_FLOW_TO_FOAM_FIELD_MAP = {
    "U": "U",
    "p": "p",
    "T": "T",
    "rho": "rho",
    "nu": "nu",
    "k": "k",
    "epsilon": "epsilon",
    "omega": "omega",
    "nut": "nut",
    "Rij": "R",
}

DEFAULT_RIJ_READ_CANDIDATES = ["R", "Rij", "ReynoldsStress", "tauRANS", "tau_ij"]
DEFAULT_TEMPLATE_FIELDS = ["U", "p", "k", "omega", "nut"]
DEFAULT_FIELD_CLASS = {
    "U": "volVectorField",
    "Rij": "volSymmTensorField",
}


def _get_case(case_path: str) -> Any:
    try:
        import foamlib  # type: ignore
    except Exception as e:
        raise ImportError(
            "foamlib is not installed/importable in this environment. Install it first."
        ) from e

    return foamlib.FoamCase(case_path)


def _resolve_time_name(case: Any, case_path: str, time: Optional[str]) -> str:
    if time and str(time).lower() not in ("latest", "end", "final"):
        return str(time)

    time_names = [str(td.name) for td in case]
    if not time_names:
        dirs = [
            d
            for d in os.listdir(case_path)
            if os.path.isdir(os.path.join(case_path, d))
        ]
        time_names = dirs

    def _is_float(value: str) -> bool:
        try:
            float(value)
            return True
        except Exception:
            return False

    numeric_names = [t for t in time_names if _is_float(t)]
    if numeric_names:
        return sorted(numeric_names, key=lambda x: float(x))[-1]

    if time_names:
        return sorted(time_names)[-1]

    raise RuntimeError("No OpenFOAM time directories found")


def _read_cell_centers(case: Any, time_name: str) -> np.ndarray:
    time_dir = case[time_name]
    C_file = time_dir.cell_centers()
    C = np.asarray(C_file["internalField"], dtype=float)
    if C.ndim != 2 or C.shape[1] != 3:
        raise RuntimeError(f"cell centres have unexpected shape: {C.shape}")
    return C


def _extract_internal_field_array(field_file: Any, n_points: Optional[int]) -> np.ndarray:
    internal = field_file["internalField"]

    if isinstance(internal, np.ndarray):
        return np.asarray(internal, dtype=float)

    # OpenFOAM uniform fields may come as ("uniform", value_or_$name)
    if (
        isinstance(internal, tuple)
        and len(internal) == 2
        and str(internal[0]).lower() == "uniform"
    ):
        if n_points is None:
            raise ValueError(
                "Cannot expand uniform internalField without known point count"
            )

        value = internal[1]
        if isinstance(value, str) and value.startswith("$"):
            ref_name = value[1:]
            value = field_file[ref_name]

        value_np = np.asarray(value, dtype=float)
        if value_np.ndim == 0:
            return np.full((n_points,), float(value_np), dtype=float)
        return np.tile(value_np.reshape(1, -1), (n_points, 1))

    return np.asarray(internal, dtype=float)


def _read_openfoam_field(
    case: Any,
    time_name: str,
    foam_field_name: str,
    n_points: Optional[int],
) -> np.ndarray:
    field_file = case.file(f"{time_name}/{foam_field_name}")
    return _extract_internal_field_array(field_file=field_file, n_points=n_points)


def _to_rij_tensor(field_values: np.ndarray) -> np.ndarray:
    if field_values.ndim == 2 and field_values.shape[1] == 6:
        xx, yy, zz, xy, yz, xz = [field_values[:, j] for j in range(6)]
        out = np.zeros((field_values.shape[0], 3, 3), dtype=float)
        out[:, 0, 0] = xx
        out[:, 1, 1] = yy
        out[:, 2, 2] = zz
        out[:, 0, 1] = out[:, 1, 0] = xy
        out[:, 1, 2] = out[:, 2, 1] = yz
        out[:, 0, 2] = out[:, 2, 0] = xz
        return out

    if field_values.ndim == 2 and field_values.shape[1] == 9:
        out = field_values.reshape(-1, 3, 3)
        out = 0.5 * (out + np.transpose(out, (0, 2, 1)))
        return out

    if field_values.ndim == 3 and field_values.shape[1:] == (3, 3):
        out = 0.5 * (field_values + np.transpose(field_values, (0, 2, 1)))
        return out

    raise ValueError(
        "Unsupported Rij field shape for conversion. "
        f"Expected (N,6), (N,9), or (N,3,3), got {field_values.shape}"
    )


def _to_openfoam_rij_symm6(rij: np.ndarray) -> np.ndarray:
    if rij.ndim == 2 and rij.shape[1] == 6:
        return np.asarray(rij, dtype=float)

    if rij.ndim != 3 or rij.shape[1:] != (3, 3):
        raise ValueError(
            f"Rij must be shape (N,3,3) or (N,6), got {rij.shape}"
        )

    out = np.zeros((rij.shape[0], 6), dtype=float)
    out[:, 0] = rij[:, 0, 0]  # xx
    out[:, 1] = rij[:, 1, 1]  # yy
    out[:, 2] = rij[:, 2, 2]  # zz
    out[:, 3] = 0.5 * (rij[:, 0, 1] + rij[:, 1, 0])  # xy
    out[:, 4] = 0.5 * (rij[:, 1, 2] + rij[:, 2, 1])  # yz
    out[:, 5] = 0.5 * (rij[:, 0, 2] + rij[:, 2, 0])  # xz
    return out


def _build_index_map_by_coords(
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    atol: float,
) -> np.ndarray:
    return build_index_map_by_coords(
        source_coords=source_coords,
        target_coords=target_coords,
        atol=atol,
    )


def _prepare_internal_field_for_write(flow_field_name: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if flow_field_name == "U":
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"U must have shape (N,3), got {values.shape}")
        return values

    if flow_field_name == "Rij":
        return _to_openfoam_rij_symm6(values)

    if values.ndim == 1:
        return values

    raise ValueError(
        f"Field '{flow_field_name}' must be 1D scalar, (N,3) vector, or Rij tensor; got {values.shape}"
    )


def _find_template_file(
    case: Any,
    time_name: str,
    template_time: Optional[str],
    foam_field_name: str,
) -> Optional[Any]:
    candidate_paths = []

    if template_time:
        candidate_paths.append(f"{template_time}/{foam_field_name}")
        candidate_paths.extend(
            [f"{template_time}/{name}" for name in DEFAULT_TEMPLATE_FIELDS]
        )

    candidate_paths.append(f"{time_name}/{foam_field_name}")
    candidate_paths.extend([f"{time_name}/{name}" for name in DEFAULT_TEMPLATE_FIELDS])

    case_root = str(case.path)
    for rel_path in candidate_paths:
        abs_path = os.path.join(case_root, rel_path)
        if os.path.exists(abs_path):
            return case.file(rel_path)

    return None


def _initialize_missing_field_file(
    field_file: Any,
    template_file: Any,
    n_points: int,
    flow_field_name: str,
) -> None:
    for key in template_file.keys():
        if key != "internalField":
            field_file[key] = template_file[key]

    field_file["internalField"] = np.zeros((n_points,), dtype=float)

    expected_class = DEFAULT_FIELD_CLASS.get(flow_field_name)
    if expected_class is not None:
        field_file["FoamFile"] = {"class": expected_class}


def read_flow_data_from_openfoam(
    case_path: str,
    flow_data: Optional[FlowData] = None,
    time: Optional[str] = None,
    fields: Optional[List[str]] = None,
    field_map: Optional[Dict[str, str]] = None,
) -> FlowData:
    """
    Read OpenFOAM fields into FlowData.

    Args:
        case_path: OpenFOAM case directory path.
        flow_data: Optional FlowData to populate. If None, a new instance is created.
        time: Time directory to read (e.g. "3000"). If None/"latest", uses latest.
        fields: FlowData field names to load. If None, uses supported defaults.
        field_map: Optional map {flow_field_name: openfoam_field_name}.

    Returns:
        FlowData populated with coordinates and requested fields.
    """
    flow_data = FlowData() if flow_data is None else flow_data
    field_map = {} if field_map is None else dict(field_map)

    case = _get_case(case_path=case_path)
    time_name = _resolve_time_name(case=case, case_path=case_path, time=time)

    coords = _read_cell_centers(case=case, time_name=time_name)
    n_points = coords.shape[0]

    flow_data.coords = coords
    flow_data.x_vec = np.unique(coords[:, 0])
    flow_data.y_vec = np.unique(coords[:, 1])
    flow_data.z_vec = np.unique(coords[:, 2])

    requested_fields = (
        fields
        if fields is not None
        else [
            "U",
            "p",
            "T",
            "rho",
            "nu",
            "k",
            "epsilon",
            "omega",
            "nut",
            "Rij",
        ]
    )

    for flow_field_name in requested_fields:
        if flow_field_name not in flow_data.get_field_names():
            raise ValueError(f"Unknown FlowData field: '{flow_field_name}'")

        try:
            if flow_field_name == "Rij" and "Rij" not in field_map:
                rij_values = None
                for candidate in DEFAULT_RIJ_READ_CANDIDATES:
                    try:
                        arr = _read_openfoam_field(
                            case=case,
                            time_name=time_name,
                            foam_field_name=candidate,
                            n_points=n_points,
                        )
                        rij_values = _to_rij_tensor(arr)
                        break
                    except Exception:
                        continue
                if rij_values is None:
                    continue
                setattr(flow_data, flow_field_name, rij_values)
                continue

            foam_field_name = field_map.get(
                flow_field_name, DEFAULT_FLOW_TO_FOAM_FIELD_MAP.get(flow_field_name)
            )
            if foam_field_name is None:
                continue

            raw_values = _read_openfoam_field(
                case=case,
                time_name=time_name,
                foam_field_name=foam_field_name,
                n_points=n_points,
            )

            if flow_field_name == "U":
                if raw_values.ndim != 2 or raw_values.shape[1] != 3:
                    raise ValueError(
                        f"OpenFOAM field '{foam_field_name}' expected (N,3), got {raw_values.shape}"
                    )
                setattr(flow_data, flow_field_name, raw_values)
            elif flow_field_name == "Rij":
                setattr(flow_data, flow_field_name, _to_rij_tensor(raw_values))
            else:
                setattr(flow_data, flow_field_name, raw_values.reshape(-1))
        except FileNotFoundError:
            continue

    flow_data.simulation_config = Path(case_path.rstrip(os.sep)).name
    flow_data.n_points = n_points
    flow_data.grid_shape = np.array(
        [
            flow_data.x_vec.shape[0],
            flow_data.y_vec.shape[0],
            flow_data.z_vec.shape[0],
        ]
    )
    flow_data.is_loaded = True

    return flow_data


def write_flow_data_to_openfoam(
    case_path: str,
    flow_data: FlowData,
    time: Optional[str] = None,
    fields: Optional[List[str]] = None,
    field_map: Optional[Dict[str, str]] = None,
    atol: float = 1e-10,
    create_if_missing: bool = False,
    template_time: Optional[str] = "0",
) -> str:
    """
    Write FlowData fields into OpenFOAM field files.

    Mapping is done by coordinates, not by index order, so this is robust when
    FlowData point order differs from OpenFOAM internal cell order.

    Args:
        case_path: OpenFOAM case directory path.
        flow_data: Source FlowData.
        time: Target time directory to write (e.g. "3000"). If None/"latest", uses latest.
        fields: FlowData fields to write. If None, writes known supported fields when available.
        field_map: Optional map {flow_field_name: openfoam_field_name}.
        atol: Coordinate matching tolerance.
        create_if_missing: If True, create missing target field files from templates.
        template_time: Preferred time directory for template headers/boundary when creating fields.

    Returns:
        The resolved time directory name written to.
    """
    if flow_data.coords.shape[0] == 0:
        raise ValueError("flow_data.coords is empty")

    field_map = {} if field_map is None else dict(field_map)
    requested_fields = (
        fields
        if fields is not None
        else [
            "U",
            "p",
            "T",
            "rho",
            "nu",
            "k",
            "epsilon",
            "omega",
            "nut",
            "Rij",
        ]
    )

    case = _get_case(case_path=case_path)
    time_name = _resolve_time_name(case=case, case_path=case_path, time=time)
    foam_coords = _read_cell_centers(case=case, time_name=time_name)

    source_to_target_idx = _build_index_map_by_coords(
        source_coords=flow_data.coords,
        target_coords=foam_coords,
        atol=atol,
    )

    n_points = foam_coords.shape[0]
    for flow_field_name in requested_fields:
        if flow_field_name not in flow_data.get_field_names():
            raise ValueError(f"Unknown FlowData field: '{flow_field_name}'")

        values = np.asarray(getattr(flow_data, flow_field_name))
        if values.size == 0:
            continue
        if values.shape[0] != flow_data.coords.shape[0]:
            raise ValueError(
                f"Field '{flow_field_name}' first axis must match flow_data.coords count "
                f"({flow_data.coords.shape[0]}), got {values.shape}"
            )

        foam_field_name = field_map.get(
            flow_field_name, DEFAULT_FLOW_TO_FOAM_FIELD_MAP.get(flow_field_name)
        )
        if foam_field_name is None:
            continue

        field_values = _prepare_internal_field_for_write(
            flow_field_name=flow_field_name,
            values=values,
        )
        field_values = field_values[source_to_target_idx]

        rel_path = f"{time_name}/{foam_field_name}"
        abs_path = os.path.join(str(case.path), rel_path)
        target_file = case.file(rel_path)

        if not os.path.exists(abs_path):
            if not create_if_missing:
                raise FileNotFoundError(
                    f"Target field file does not exist: {abs_path}. "
                    "Set create_if_missing=True to create it."
                )

            template = _find_template_file(
                case=case,
                time_name=time_name,
                template_time=template_time,
                foam_field_name=foam_field_name,
            )
            if template is None:
                raise FileNotFoundError(
                    "Cannot create missing field file because no template field was found. "
                    f"Target={rel_path}, template_time={template_time}"
                )
            _initialize_missing_field_file(
                field_file=target_file,
                template_file=template,
                n_points=n_points,
                flow_field_name=flow_field_name,
            )

        target_file["internalField"] = field_values

    return time_name
