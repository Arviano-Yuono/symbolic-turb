import os.path as osp
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from symbolic_turb.core import FlowData

from ..base_loader import BaseLoader
from ..preprocess import Preprocessor

"""
CSV file columns:
    "I1","I2","N1","N2","N3","Rall:0","Rall:1","Rall:2","Rall:3","Rall:4","Rall:5", \
    "T1:0","T1:1","T1:2","T1:3","T1:4","T1:5","T2:0","T2:1","T2:2","T2:3","T2:4", \
    "T2:5","T3:0","T3:1","T3:2","T3:3","T3:4","T3:5","U:0","U:1","U:2","k","nonlinearStress:0", \
    "nonlinearStress:1","nonlinearStress:2","nonlinearStress:3","nonlinearStress:4","nonlinearStress:5", \
    "nut","omega","p","Points:0","Points:1","Points:2"

stress format:
    0: xx
    1: yy
    2: zz
    3: xy
    4: xz
    5: yz
"""


class RANSLoader(BaseLoader):
    """
    RANSLoader is a class inheriting from Formatter that formats Baseline k-omega SST RANS data
    obtained using openFOAM simulation into a FlowData object.

    Note:
        The Baseline k-omega SST RANS data from openFOAM simulation must be formatted to csv first
        using the script provided in the "symbolic_turb/utils/load_sst.py"

    Args:
        data_path (str): Path to the DNS data directory.
        flow_data (FlowData): FlowData object to be populated with Baseline SST data.

    Method:
        format(): Load and format the Baseline SST data for the FlowData object.
                returns FlowData object populated with DNS data.

    """

    def __init__(self, data_path: str, flow_data: FlowData) -> None:
        super().__init__(data_path, flow_data)

        self.simulation_config = Path(self.data_path).name.split("AR_")[
            -1
        ]  # takin the sim config, ex: "1_80" from "AR_1_180"

    def format(self) -> FlowData:
        """Format the RANS DNS data for the FlowData object"""
        self.df = pd.read_csv(osp.join(self.data_path, "surface.csv"))

        (
            self.flow_data.x_vec,
            self.flow_data.y_vec,
            self.flow_data.z_vec,
            self.flow_data.coords,
        ) = self._load_coords()
        self.flow_data.U = self._load_mean_velocities()
        self.flow_data.k = self._load_k()
        self.flow_data = Preprocessor().compute_gradU(self.flow_data)
        self.flow_data.omega = self._load_omega()

        # metadata
        self.flow_data = self._set_metadata()
        return self.flow_data

    def _set_metadata(self) -> FlowData:
        return super().set_metadata()

    def _load_coords(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Load coordinates data based on the baseline dataset file structure
        x_flat = np.array(self.df["Points:0"])
        y_flat = np.array(self.df["Points:1"])
        z_flat = np.array(self.df["Points:2"])
        _coords = np.stack([x_flat, y_flat, z_flat], axis=1)

        x_vec = np.unique(x_flat)
        y_vec = np.unique(y_flat)
        z_vec = np.unique(z_flat)
        return (x_vec, y_vec, z_vec, _coords)

    def _load_mean_velocities(self) -> np.ndarray:
        # Load Mean Velocity Data
        u_vec = np.array(self.df["U:0"])
        v_vec = np.array(self.df["U:1"])
        w_vec = np.array(self.df["U:2"])
        return np.stack([u_vec, v_vec, w_vec], axis=1)

    def _load_k(self) -> np.ndarray:
        # Load baseline k Data
        k_vec = np.array(self.df["k"])
        return k_vec

    def _load_omega(self) -> np.ndarray:
        # Load baseline omega (w) Data
        omega_vec = np.array(self.df["omega"])
        return omega_vec
