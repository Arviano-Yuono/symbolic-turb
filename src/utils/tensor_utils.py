import numpy as np
from typing import Optional

def build_tensor_field(
    txx: np.ndarray, txy: np.ndarray, txz: np.ndarray,
    tyx: np.ndarray, tyy: np.ndarray, tyz: np.ndarray,
    tzx: np.ndarray, tzy: np.ndarray, tzz: np.ndarray,
) -> np.ndarray:
    """
    Assemble a 2D tensor field into a full tensor array.

    Output shape:
        T[y, z, i, j] where i,j in {0,1,2} correspond to x,y,z.

    All inputs must have identical shape (Ny, Nz).
    """
    # Basic shape checks
    shp = txx.shape
    for name, arr in {
        "txy": txy, "txz": txz, "tyx": tyx, "tyy": tyy, "tyz": tyz,
        "tzx": tzx, "tzy": tzy, "tzz": tzz
    }.items():
        if arr.shape != shp:
            raise ValueError(f"Shape mismatch: txx {shp} vs {name} {arr.shape}")

    Ny, Nz = shp
    T = np.empty((Ny, Nz, 3, 3), dtype=np.result_type(
        txx, txy, txz, tyx, tyy, tyz, tzx, tzy, tzz
    ))

    T[..., 0, 0] = txx
    T[..., 0, 1] = txy
    T[..., 0, 2] = txz

    T[..., 1, 0] = tyx
    T[..., 1, 1] = tyy
    T[..., 1, 2] = tyz

    T[..., 2, 0] = tzx
    T[..., 2, 1] = tzy
    T[..., 2, 2] = tzz

    return T

def build_symmetric_tensor_field(
    txx: np.ndarray, txy: np.ndarray, txz: np.ndarray,
    tyy: np.ndarray, tyz: np.ndarray,
    tzz: np.ndarray,
) -> np.ndarray:
    """
    Assemble a symmetric 2D tensor field into a full tensor array.

    Output:
        T[y, z, 3, 3] with T_ij = T_ji

    Inputs must share shape (Ny, Nz).
    """
    shp = txx.shape
    for name, arr in {"txy": txy, "txz": txz, "tyy": tyy, "tyz": tyz, "tzz": tzz}.items():
        if arr.shape != shp:
            raise ValueError(f"Shape mismatch: txx {shp} vs {name} {arr.shape}")

    Ny, Nz = shp
    T = np.empty((Ny, Nz, 3, 3), dtype=np.result_type(txx, txy, txz, tyy, tyz, tzz))

    T[..., 0, 0] = txx
    T[..., 1, 1] = tyy
    T[..., 2, 2] = tzz

    T[..., 0, 1] = T[..., 1, 0] = txy
    T[..., 0, 2] = T[..., 2, 0] = txz
    T[..., 1, 2] = T[..., 2, 1] = tyz

    return T

def flatten_symmetric_tensor(field: np.ndarray) -> np.ndarray:
    """
    Flatten symmetric tensor field, with the format: xx,yy,zz,xy,xz,yz
    """
    flattened_tensor = np.zeros((field.shape[0], 6))
    flattened_tensor[:, 0] = field[:, 0,0]
    flattened_tensor[:, 1] = field[:, 1,1]
    flattened_tensor[:, 2] = field[:, 2,2]
    flattened_tensor[:, 3] = field[:, 0,1]
    flattened_tensor[:, 4] = field[:, 0,2]
    flattened_tensor[:, 5] = field[:, 1,2]
    return flattened_tensor