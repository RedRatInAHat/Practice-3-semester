import open3d

def visualize(points, colors):
    """Visualization of Points Cloud

    Visualising a points cloud with a coordinate system.

    Arguments:
        points (numpy.array): points of cloud
        colors (numpy.array): colors of cloud
    """

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors)

    points_axis = [[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]]
    lines_axis = [[0, 1], [0, 2], [0, 3]]
    colors_axis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    line_set_a = open3d.geometry.LineSet()
    line_set_a.points = open3d.utility.Vector3dVector(points_axis)
    line_set_a.lines = open3d.utility.Vector2iVector(lines_axis)
    line_set_a.colors = open3d.utility.Vector3dVector(colors_axis)

    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(line_set_a)
    vis.run()


if __name__ == "__main__":
    visualize([], [])