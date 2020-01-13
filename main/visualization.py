import open3d


def visualize_points(points, colors):
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
            vis.add_geometry(pcds[- 1])

    points_axis = [[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]]
    lines_axis = [[0, 1], [0, 2], [0, 3]]
    colors_axis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    line_set_a = open3d.geometry.LineSet()
    line_set_a.points = open3d.utility.Vector3dVector(points_axis)
    line_set_a.lines = open3d.utility.Vector2iVector(lines_axis)
    line_set_a.colors = open3d.utility.Vector3dVector(colors_axis)

    vis.add_geometry(line_set_a)
    vis.run()


def visualize(objects=None, points=None, points_color=None, lines=None, lines_points=None, lines_color=None):
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    if objects is not None:
        for obj in objects:
            if obj.visible:
                pcds = open3d.geometry.PointCloud()
                pcds.points = open3d.utility.Vector3dVector(obj.get_points()[0])
                pcds.colors = open3d.utility.Vector3dVector(obj.get_points()[1])
                vis.add_geometry(pcds)

    if points is not None:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        pcd.colors = open3d.utility.Vector3dVector(points_color)
        vis.add_geometry(pcd)

    if lines is not None:
        for i, _ in enumerate(lines):
            lines_set = open3d.geometry.LineSet()
            lines_set.points = open3d.utility.Vector3dVector(lines_points[i])
            lines_set.lines = open3d.utility.Vector2iVector(lines[i])
            lines_set.colors = open3d.utility.Vector3dVector(lines_color[i])
            vis.add_geometry(lines_set)

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
