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


def visualize_object(objects):
    """Vizualaize instances of PointsObjects

    Arguments:
        objects (PointsObject): list of instances of PointsObject
    """
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    pcds = []

    for i in range(len(objects)):
        if objects[i].visible:
            pcds.append(open3d.geometry.PointCloud())
            pcds[len(pcds) - 1].points = open3d.utility.Vector3dVector(objects[i].get_points()[0])
            pcds[len(pcds) - 1].colors = open3d.utility.Vector3dVector(objects[i].get_points()[1])
            vis.add_geometry(pcds[len(pcds) - 1])

    points_axis = [[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]]
    lines_axis = [[0, 1], [0, 2], [0, 3]]
    colors_axis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    line_set_a = open3d.geometry.LineSet()
    line_set_a.points = open3d.utility.Vector3dVector(points_axis)
    line_set_a.lines = open3d.utility.Vector2iVector(lines_axis)
    line_set_a.colors = open3d.utility.Vector3dVector(colors_axis)

    vis.add_geometry(line_set_a)
    vis.run()


if __name__ == "__main__":
    visualize([], [])
