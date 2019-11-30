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


def download_to_object(path, number_of_points=None):
    """Downloading .ply file and making an instance of PointsObject

    Args:
        path (str): Path to the .ply file
        number_of_points (int): number of active points which will be added to object
    Returns:
        object (instance of PointsObject)

    """
    from points_object import PointsObject

    pcd = open3d.io.read_point_cloud(path)
    object = PointsObject()
    object.set_points(np.asarray(pcd.points), np.asarray(pcd.colors), number=number_of_points)
    return object


if __name__ == "__main__":
    points, color = download_ply("models/violet thor.ply")
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(color)
    open3d.visualization.draw_geometries([pcd])
