import open3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import cv2

import moving_prediction
from moving_prediction import get_future_points


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


def get_histogram_of_probabilities(points, probabilities, step=0.1, y_start=0, y_stop=1.5):
    points_there = np.bitwise_and(points[:, 1] >= y_start, points[:, 1] <= y_stop)
    points = points[points_there][:, [0, 2]]
    probabilities = probabilities[points_there]
    dict = moving_prediction.sum_dif_probabilities_of_same_points({}, points, probabilities)
    probabilities = np.fromiter(dict.values(), dtype=float)
    points = moving_prediction.tuple_to_array(np.fromiter(dict.keys(), dtype=np.dtype('float, float')))
    x_min, x_max = np.min(points[:, 0]) - 2 * step, np.max(points[:, 0]) + 2 * step
    y_min, y_max = np.min(points[:, 1]) - 2 * step, np.max(points[:, 1]) + 2 * step
    x = np.arange(x_min, x_max, step)
    y = np.arange(y_min, y_max, step)
    z = np.zeros((x.shape[0], y.shape[0]))
    for i_prob, c in enumerate(points):
        i, j = (np.round([(-x_min + c[0]) / step, (-y_min + c[1]) / step])).astype(int)
        z[i, j] = probabilities[i_prob]
    plt.imshow(np.transpose(z), extent=[y_min, y_max, x_min, x_max])
    plt.xlabel("x, m")
    plt.ylabel("z, m")
    plt.colorbar()
    plt.show()


def show_found_functions(found_functions, t, points, tt, real_y, x_label='', y_label='', title=''):
    trajectory = get_future_points(found_functions, tt)
    legend = ["given points", 'ground truth']
    plt.plot(t, points, 'o', tt, real_y, '-')
    for func, y_ in zip(found_functions, trajectory):
        plt.plot(tt, y_, '--')
        legend.append(func)
    plt.legend(legend)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def show_points_with_obstacles(function, tt, obstacles_level):
    trajectory = get_future_points(function, tt)
    for i in obstacles_level:
        plt.axhline(y=i, color='cyan', linestyle='--')
    for func, y_ in zip(function, trajectory):
        plt.plot(tt, y_, '--')
    plt.show()

def show_image(image):
    cv2.imshow("", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
