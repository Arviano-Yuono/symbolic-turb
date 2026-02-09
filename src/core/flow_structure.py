"""
This module consists of a standardized data structure for flow data.

"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FlowData:
    """
    Canonical data class for flow data.

    Args:
        coords: Coordinates of the flow data points.
        U: Mean flow velocity at each point.
        p: Pressure at each point.
        T: Temperature at each point.
        rho: Density at each point.
        nu: Kinematic viscosity at each point.
        k: Turbulent kinetic energy at each point.
        epsilon: Turbulent dissipation rate at each point.
        omega: Turbulent frequency at each point.
        nut: Turbulent viscosity at each point.
        Rij: Reynolds stress tensor at each point.
        Sij: Strain rate tensor at each point.
        Wij: Rotation rate tensor at each point.

    Methods:
        validate(): Validate the data in the FlowData object.
                returns True if valid, False otherwise.
        __save__(): Save the FlowData object to .pkl at save_path.
    """

    # Metadata
    simulation_config: Optional[str] = None
    n_points: int = 0
    grid_shape: np.ndarray = field(
        default_factory=lambda: np.empty(
            3,
        )
    )  # (3,)

    # Coordinates and mean flow
    x_vec: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (nx,)
    y_vec: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (ny,)
    z_vec: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (nz,)
    coords: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))  # (N,3)
    U: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))  # (N,3)

    # Scalar fields
    p: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (N,)
    T: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (N,)
    rho: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (N,)
    nu: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (N,)

    # Turbulence scalars
    k: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (N,)
    epsilon: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (N,)
    omega: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (N,)
    nut: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (N,)

    # Tensors (tensor format: (N, i, j))
    Rij: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))  # (N,3,3)
    bij: np.ndarray = field(
        default_factory=lambda: np.empty((0, 6))
    )  # (N,6) because anisotropy is symmetric

    # Derived tensors (optional, computed later)
    gradU: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))  # (N,3,3)
    anisotropy: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3, 3))
    )  # (N,3,3)
    Sij: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))  # (N,3,3)
    Wij: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))  # (N,3,3)

    # Tensor basis / Invariants
    T1: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))  # (N,3,3)
    T2: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))  # (N,3,3)
    T3: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))  # (N,3,3)

    I1: np.ndarray = field(default_factory=lambda: np.empty((0, 1, 1)))  # (N,1,1)
    I2: np.ndarray = field(default_factory=lambda: np.empty((0, 1, 1)))  # (N,1,1)

    # Bool
    is_loaded: bool = False
    is_preprocessed: bool = False

    def get_field_names(self):
        "return a list of class properties dynamically"
        return [field.name for field in self.__dataclass_fields__.values()]

    def validate(self):
        N = self.coords.shape[0]
        assert self.U.shape == (N, 3), (
            f"Invalid shape for U {self.U.shape} with {N} points"
        )

    def save(self, save_path: str):
        "Save FlowData object to a .pkl file"
        return NotImplementedError
        import pickle

        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    def load(self, load_path: str):
        "Load FlowData object from a .pkl file"
        return NotImplementedError
        import pickle

        with open(load_path, "rb") as f:
            self = pickle.load(f)
        self.validate()
        self.is_loaded = True

    def __repr__(self):
        repr_str = "FlowData(\n"
        for field_name in self.get_field_names():
            field_value = getattr(self, field_name)
            if type(field_value) is np.ndarray:
                repr_str += f"    {field_name}: {field_value.shape}\n"
            else:
                repr_str += f"    {field_name}: {field_value}\n"
        repr_str += ")"
        return repr_str
