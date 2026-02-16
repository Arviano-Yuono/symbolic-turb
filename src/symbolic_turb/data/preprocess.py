from typing import List

import numpy as np
from scipy.interpolate import griddata

from symbolic_turb.core import FlowData
from .coord_mapping import build_index_map_by_coords


class Preprocessor:
    """
    Preprocesses the flow data by calculating the derivated fields of the FlowData object.
    Static Methods:
        compute_gradU: Calculates the velocity gradient tensor adaptively.
        interpolate_scalar_field: Interpolates a scalar field to a new grid.
    """

    @staticmethod
    def compute_gradU(flow_data: FlowData) -> FlowData:
        nx, ny, nz = (
            flow_data.grid_shape[0],
            flow_data.grid_shape[1],
            flow_data.grid_shape[2],
        )
        N = nx * ny * nz
        edge = (
            2 if min([n for n in (nx, ny, nz) if n > 1]) >= 3 else 1
        )  # for safety in calculating gradient
        assert flow_data.U.shape == (N, 3), f"Expected U (N,3), got {flow_data.U.shape}"

        U = flow_data.U.reshape(nx, ny, nz, 3)
        grad = np.zeros((nx, ny, nz, 3, 3), dtype=flow_data.U.dtype)

        def grad_scalar(F):
            dFx = np.zeros_like(F)
            dFy = np.zeros_like(F)
            dFz = np.zeros_like(F)

            if nx > 1 and ny > 1 and nz > 1:
                dFx, dFy, dFz = np.gradient(
                    F,
                    flow_data.x_vec,
                    flow_data.y_vec,
                    flow_data.z_vec,
                    edge_order=edge,
                )
                return dFx, dFy, dFz

            if nz == 1:  # (x,y)
                F2 = F[:, :, 0]
                if nx > 1 and ny > 1:
                    gx, gy = np.gradient(
                        F2, flow_data.x_vec, flow_data.y_vec, edge_order=edge
                    )
                    dFx[:, :, 0], dFy[:, :, 0] = gx, gy
                elif nx > 1:
                    (gx,) = np.gradient(F2[:, 0], flow_data.x_vec, edge_order=edge)
                    dFx[:, 0, 0] = gx
                elif ny > 1:
                    (gy,) = np.gradient(F2[0, :], flow_data.y_vec, edge_order=edge)
                    dFy[0, :, 0] = gy
                return dFx, dFy, dFz

            if ny == 1:  # (x,z)
                F2 = F[:, 0, :]
                if nx > 1 and nz > 1:
                    gx, gz = np.gradient(
                        F2, flow_data.x_vec, flow_data.z_vec, edge_order=edge
                    )
                    dFx[:, 0, :], dFz[:, 0, :] = gx, gz
                elif nx > 1:
                    (gx,) = np.gradient(F2[:, 0], flow_data.x_vec, edge_order=edge)
                    dFx[:, 0, 0] = gx
                elif nz > 1:
                    (gz,) = np.gradient(F2[0, :], flow_data.z_vec, edge_order=edge)
                    dFz[0, 0, :] = gz
                return dFx, dFy, dFz

            if nx == 1:  # (y,z)
                F2 = F[0, :, :]
                if ny > 1 and nz > 1:
                    gy, gz = np.gradient(
                        F2, flow_data.y_vec, flow_data.z_vec, edge_order=edge
                    )
                    dFy[0, :, :], dFz[0, :, :] = gy, gz
                elif ny > 1:
                    (gy,) = np.gradient(F2[:, 0], flow_data.y_vec, edge_order=edge)
                    dFy[0, :, 0] = gy
                elif nz > 1:
                    (gz,) = np.gradient(F2[0, :], flow_data.z_vec, edge_order=edge)
                    dFz[0, 0, :] = gz
                return dFx, dFy, dFz

            return dFx, dFy, dFz

        for comp in range(3):
            dFx, dFy, dFz = grad_scalar(U[..., comp])
            grad[..., comp, 0] = dFx
            grad[..., comp, 1] = dFy
            grad[..., comp, 2] = dFz

        flow_data.gradU = grad.reshape(N, 3, 3)
        return flow_data

    @staticmethod
    def _build_index_map_by_coords(
        source_coords: np.ndarray,
        target_coords: np.ndarray,
        atol: float = 1e-10,
    ) -> np.ndarray:
        return build_index_map_by_coords(
            source_coords=source_coords,
            target_coords=target_coords,
            atol=atol,
        )

    @staticmethod
    def map_fields_by_coords(
        source_data: FlowData,
        target_data: FlowData,
        field_names: List[str],
        atol: float = 1e-10,
    ) -> FlowData:
        """
        Map fields from source_data to target_data by coordinates.

        Useful when both datasets represent the same points but not in the same order,
        e.g. OpenFOAM internal field order vs another FlowData point ordering.
        """
        assert source_data.coords.shape[0] != 0, "source_data has no coordinates"
        assert target_data.coords.shape[0] != 0, "target_data has no coordinates"

        source_to_target_idx = Preprocessor._build_index_map_by_coords(
            source_coords=source_data.coords,
            target_coords=target_data.coords,
            atol=atol,
        )

        n_source = source_data.coords.shape[0]
        for field_name in field_names:
            assert field_name in source_data.get_field_names(), (
                f"Field '{field_name}' is not in source_data. "
                f"Available fields: {source_data.get_field_names()}"
            )
            assert field_name in target_data.get_field_names(), (
                f"Field '{field_name}' is not in target_data. "
                f"Available fields: {target_data.get_field_names()}"
            )

            source_field = np.asarray(getattr(source_data, field_name))
            assert source_field.shape[0] == n_source, (
                f"Field '{field_name}' first axis must match source point count "
                f"({n_source}), got {source_field.shape}"
            )

            mapped_field = source_field[source_to_target_idx]
            setattr(target_data, field_name, mapped_field.copy())

        return target_data

    @staticmethod
    def interpolate_tensor_field(
        source_data: FlowData,
        target_data: FlowData,
        method: str = "cubic",
    ):
        """
        Interpolate tensor field from source_data grid to target_data with different resolutions.

        Args:
            source_data (FlowData): source data that will be interpolated.
            target_data (FlowData): target data.

        Returns:
            target_data (FlowData): interpolated data.
        """
        # assert field_name in source_data.get_field_names(), (
        #     f"Field {field_name} not found in source data field name list:\n{source_data.get_field_names()}."
        # )
        assert method in ["linear", "cubic"], f"Invalid interpolation method: {method}"
        assert source_data.coords.shape[0] != 0, "Source data has no points"
        assert target_data.coords.shape[0] != 0, "Target data has no points"

        dns_points = source_data.coords[:, 1:3]
        rans_points = target_data.coords[:, 1:3]

        for i in range(3):
            target_data.U[:, i] = griddata(
                dns_points, source_data.U[:, i], rans_points, method="cubic"
            )

        target_data.k = griddata(dns_points, source_data.k, rans_points, method="cubic")

        _Rij_holder = np.zeros((target_data.k.shape[0], 3, 3))
        for i in range(3):
            for j in range(i, 3):
                mapped_comp = griddata(
                    dns_points, source_data.Rij[:, i, j], rans_points, method="cubic"
                )
                _Rij_holder[:, i, j] = mapped_comp
                _Rij_holder[:, j, i] = mapped_comp  # symmetry

        target_data.Rij = _Rij_holder
        target_data.is_loaded = True

        return target_data
