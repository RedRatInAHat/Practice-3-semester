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
    points, color = download_ply("models/violet thor.ply")
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(color)
    open3d.visualization.draw_geometries([pcd])
