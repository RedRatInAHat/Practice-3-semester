import open3d
import numpy as np


def download_ply(path):
    """Downloading .ply file

    Args:
        path (str): Path to the .ply file
    Returns:
        numpy.array1: points of loaded file
        numpy.array2: colors of loaded file
    """
    pcd = open3d.io.read_point_cloud(path)
    return np.asarray(pcd.points), np.asarray(pcd.colors)


if __name__ == "__main__":
    from points_object import PointsObject

    points, color = download_ply("models/fragment.ply")
    cube = PointsObject()
    cube.set_points(points, color)
    print(cube.get_points())