from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from symbolic_turb.core import (
    FlowData,
    compute_anisotropy,
    compute_basis_tensor,
    compute_invariants,
    compute_k_field,
    compute_rotation_rate,
    compute_strain_rate,
)

from ..base_loader import BaseLoader
from ..foam_parser import read_flow_data_from_openfoam
from ..preprocess import Preprocessor


class FOAMLoader(BaseLoader):
    """
    Load OpenFOAM data through foam_parser and populate FlowData.

    OpenFOAM I/O is delegated to `read_flow_data_from_openfoam` so this loader
    focuses only on metadata and derived physics fields.
    """

    def __init__(
        self,
        data_path: str,
        flow_data: FlowData,
        time: Optional[str] = None,
        region: str = "region0",
    ) -> None:
        super().__init__(data_path, flow_data)

        self.time = time  # e.g. "0", "1000", or None -> latest
        self.region = region  # reserved for future multi-region support
        self.simulation_config = Path(self.data_path).name

        self._raw_flow_data: Optional[FlowData] = None

    def _read_raw(self) -> FlowData:
        if self._raw_flow_data is None:
            self._raw_flow_data = read_flow_data_from_openfoam(
                case_path=self.data_path,
                time=self.time,
                fields=["U", "k", "omega", "Rij"],
            )
        return self._raw_flow_data

    def load(self) -> FlowData:
        # required fields
        (
            self.flow_data.x_vec,
            self.flow_data.y_vec,
            self.flow_data.z_vec,
            self.flow_data.coords,
        ) = self._load_coords()
        self.flow_data.U = self._load_mean_velocities()

        # metadata needed before gradU
        self.flow_data = self.set_metadata()

        # compute gradU
        self.flow_data = Preprocessor().compute_gradU(flow_data=self.flow_data)

        raw = self._read_raw()
        k = np.asarray(raw.k, dtype=float).reshape(-1) if raw.k.size != 0 else None
        omega = (
            np.asarray(raw.omega, dtype=float).reshape(-1) if raw.omega.size != 0 else None
        )
        Rij = np.asarray(raw.Rij, dtype=float) if raw.Rij.size != 0 else None

        # physics
        self.flow_data.Sij = compute_strain_rate(gradU=self.flow_data.gradU)
        self.flow_data.Wij = compute_rotation_rate(gradU=self.flow_data.gradU)

        # Prefer k if present, otherwise infer from Rij if present
        if k is not None:
            self.flow_data.k = k
        elif Rij is not None:
            self.flow_data.Rij = Rij
            self.flow_data.k = compute_k_field(Rij=Rij)
        else:
            raise RuntimeError(
                "FOAMLoader: couldn't find 'k' field and couldn't find any Rij field "
                "(e.g., R/Rij/ReynoldsStress/tau_ij). Please write 'k' or export Rij."
            )

        # omega (if not present fallback like DNS case)
        if omega is not None:
            self.flow_data.omega = omega
        else:
            self.flow_data.omega = np.ones_like(self.flow_data.k)

        # anisotropy only if Rij exists
        if Rij is not None:
            self.flow_data.Rij = Rij
            self.flow_data.anisotropy = compute_anisotropy(
                Rij=Rij, k=self.flow_data.k
            )

        self.flow_data.T1, self.flow_data.T2, self.flow_data.T3 = compute_basis_tensor(
            Sij=self.flow_data.Sij,
            Wij=self.flow_data.Wij,
            omega=self.flow_data.omega,
        )
        self.flow_data.I1, self.flow_data.I2 = compute_invariants(
            Sij=self.flow_data.Sij,
            Wij=self.flow_data.Wij,
            omega=self.flow_data.omega,
        )

        return self.flow_data

    def _load_coords(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw = self._read_raw()
        coords = np.asarray(raw.coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise RuntimeError(f"FOAMLoader: cell centres have unexpected shape: {coords.shape}")
        return raw.x_vec.copy(), raw.y_vec.copy(), raw.z_vec.copy(), coords.copy()

    def _load_mean_velocities(self) -> np.ndarray:
        raw = self._read_raw()
        U = np.asarray(raw.U, dtype=float)
        if U.ndim != 2 or U.shape[1] != 3:
            raise RuntimeError(f"FOAMLoader: U has unexpected shape: {U.shape}")
        return U.copy()
