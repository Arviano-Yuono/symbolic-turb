def load_sst_to_csv(foam_path: str, csv_path: str) -> None:
    """load RANS simulation result from openFOAM to a csv file"""
    import pyvista as pv
    import pandas as pd

    case = pv.POpenFOAMReader(foam_path)
    case.set_active_time_value(3000)
    mesh = case.read()[0]  # internal mesh

    slice_plane = mesh.slice(
        normal='x',
        origin=[1.0, 0, 0],
        generate_triangles=False
    )

    slice_point_data = slice_plane.cell_data_to_point_data()

    points = slice_point_data.points  # (Npoints, 3)

    data = {
        'Points:0': points[:, 0],  # x
        'Points:1': points[:, 1],  # y
        'Points:2': points[:, 2],  # z
        'U:0': slice_point_data['U'][:, 0],
        'U:1': slice_point_data['U'][:, 1],
        'U:2': slice_point_data['U'][:, 2],
        'k': slice_point_data['k'],
        'omega': slice_point_data['omega'],
        'nut': slice_point_data['nut'],
    }

    df_points = pd.DataFrame(data)
    df_points.to_csv(
        csv_path,
        index=False
    )