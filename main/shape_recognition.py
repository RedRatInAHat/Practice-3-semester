import numpy as np
import random
import math


def RANSAC(xyz, xyz_normals, point_to_model_accuracy=0.05, normal_to_normal_accuracy=0.01,
           number_of_points_threshold=500,
           number_of_iterations=10, min_pc_number=100, number_of_subsets=10):
    """RANSAC method for finding parameters of point cloud and it's primitive shape(s)

    Args:
        xyz (np.ndarray): points of point cloud in xyz format
        xyz_normals (np.ndarray): normals of corresponding points
        point_to_model_accuracy (float): distance between model and point to apply the point as inliner
        normal_to_normal_accuracy (float): angle between model normals and point normal to apply the point as inliner
        number_of_points_threshold (int): number of inliners to apply model as successful
        number_of_iterations (int): number of iterations before the best models will be applied as successful
        min_pc_number (int): number of points recognized too small to search for the model
        number_of_subsets (int): number of subsets for detecting shape model
    """
    found_shapes = []
    itt = 0
    while itt < number_of_iterations and xyz.shape[0] > min_pc_number:
        itt += 1

        plane_normal, plane_ro, plane_inliners = get_best_plane_model(xyz, xyz_normals, point_to_model_accuracy,
                                                                      normal_to_normal_accuracy, number_of_subsets)

        if np.sum(plane_inliners) > number_of_points_threshold:
            print(np.sum(plane_inliners), np.sum(plane_inliners) > number_of_points_threshold)
            xyz_in = xyz[plane_inliners]
            # found_shapes.append(plane_points(plane_normal, plane_ro, np.min(xyz_in[:, 0]), np.max(xyz_in[:, 0]),
            #                                  np.min(xyz_in[:, 1]), np.max(xyz_in[:, 1]), np.min(xyz_in[:, 2]),
            #                                  np.max(xyz_in[:, 2])))
            # found_shapes.append(plane_points_free_shape(plane_normal, plane_ro, xyz_in))
            found_shapes.append(plane_points_long_one(plane_normal, plane_ro, xyz_in))
            # delete found points
            xyz = xyz[np.bitwise_not(plane_inliners)]
            xyz_normals = xyz_normals[np.bitwise_not(plane_inliners)]

    return found_shapes


def get_best_plane_model(xyz, xyz_normals, point_to_model_accuracy, normal_to_normal_accuracy, number_of_subsets):
    best_score = 0
    # plane fitting
    for _ in range(number_of_subsets):
        normal, ro = plane_fitting_one_point(xyz, xyz_normals)
        # finding plane inliners
        p_inliners = plane_inliners(xyz, xyz_normals, normal, ro, point_to_model_accuracy, normal_to_normal_accuracy)

        if np.sum(p_inliners) > best_score:
            # print(np.sum(abs(np.sum(xyz * normal, axis=1) - ro)[p_inliners])/np.sum(p_inliners))
            best_normal = normal
            best_ro = ro

    return best_normal, best_ro, p_inliners


def plane_fitting_one_point(points, normals):
    """ Finding the parameters of plane

    From equation n1(x-x0) + n2(y-y0) + n3(z-z0) = 0 find four parameters (n1 n2 n3 ro=n1x + n2y + n3z); xyz from point,
    n1 n2 n3 from normal. Need only one point

    Args:
        points (np.ndarray): points of fitting object
        normals (np.ndarray): normals of fitting object

    Returns:
        normal (np.ndarray): normal of found plane
        ro (float): offset parameter of found plane
    """

    # choose one point
    i = random.randint(0, points.shape[0] - 1)
    # finding parameters
    normal = normals[i]
    ro = np.sum(points[i] * normals[i])
    return normal, ro


def plane_fitting_three_points(points):
    p = points[np.random.randint(points.shape[0], size=3), :]
    p0, p1, p2 = p

    v1 = p2 - p0
    v2 = p1 - p0

    cp = np.cross(v1, v2)
    cp = cp / np.linalg.norm(cp)
    d = np.dot(cp, p2)

    return cp, d


def plane_inliners(points, normals, plane_normal, plane_ro, d_accuracy, a_accuracy):
    """ Finding inliners for created model

    Inserting point into equation and looking for the difference between recieved ro and model ro.
    Finding angle between normals.
    If in for both cases difference and angle are below the threshold, it sets as inliner.

    Args:
        points (np.ndarray): points of fitting object
        normals (np.ndarray): normals of fitting object
        plane_normal (np.ndarray): normal of plane model
        plane_ro (float): offset parameter of plane model
        d_accuracy (float): threshold difference between model and point
        a_accuracy (float): threshold angle between normals of model and point

    Returns:
        arg1 (np.ndarray): map, showing whether point belongs to model
    """
    # finding angle between point normal and plane model normal
    angles = angle_between_normals(normals, plane_normal)
    # threshold check
    angle_truth = np.logical_or(np.where(abs(angles) < a_accuracy, True, False),
                                np.where(abs(angles) - math.pi < a_accuracy, True, False))
    distance_truth = np.where(abs(np.sum(points * plane_normal, axis=1) - plane_ro) < d_accuracy, True, False)
    return np.logical_and(angle_truth, distance_truth)


def angle_between_normals(n1, n2):
    """ Returns the angle in radians between vectors 'n1' and 'n2'"""
    cosang = np.dot(n1, n2)
    sinang = np.linalg.norm(np.cross(n1, n2))
    return np.arctan2(sinang, cosang)


def plane_points_long_one(normal, ro, points, step=0.01):
    around_x = np.around(points[:, 0], decimals=get_count(step))
    x = np.unique(around_x)
    xyz = np.empty((0,3))
    for i, x_value in enumerate(x):
        min_y, max_y = np.min(points[around_x == x_value, 1]), np.max(points[around_x == x_value, 1])
        y = np.arange(min_y, max_y + step, step)
        xy = np.ones([y.shape[0], 3])
        xy[:, 0] *= x_value
        xy[:, 1] *= y
        xyz = np.vstack((xyz, xy))
    xyz[:, 2] = (ro - normal[0] * xyz[:, 0] - normal[1] * xyz[:, 1]) / normal[2]
    return xyz


def plane_points_free_shape(normal, ro, points, step=0.01):
    x = np.arange(np.min(points[:, 0]), np.max(points[:, 0]) + step, step)
    y = np.arange(np.min(points[:, 1]), np.max(points[:, 1]) + step, step)

    x = np.tile(x, (y.shape[0], 1))
    y = np.tile(np.array([y]).transpose(), (1, x.shape[1]))

    points_around = np.around(points[:, :2], decimals=get_count(step))
    found_points = np.c_[x.flatten(), y.flatten()]
    found_points_around = np.around(found_points, decimals=get_count(step))
    print(found_points_around.shape)

    # code from here https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    # i'm not cool enough to understand why does it work. @TODO understand
    nrows, ncols = points_around.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [points_around.dtype]}

    C = np.intersect1d(points_around.view(dtype), found_points_around.view(dtype))
    xy_points = C.view(points_around.dtype).reshape(-1, ncols)

    z = (ro - normal[0] * xy_points[:, 0] - normal[1] * xy_points[:, 1]) / normal[2]
    return np.c_[xy_points, z]


def plane_points(normal, ro, x_min, x_max, y_min, y_max, z_min, z_max, step=0.01):
    """Creating the plane between minimum and maximum points"""
    x = np.arange(x_min, x_max + step, step)
    y = np.arange(y_min, y_max + step, step)

    x = np.tile(x, (y.shape[0], 1))
    y = np.tile(np.array([y]).transpose(), (1, x.shape[1]))

    z = (ro - normal[0] * x - normal[1] * y) / normal[2]
    z_condition = np.where((z.flatten() >= z_min) & (z.flatten() <= z_max), True, False)

    return np.c_[np.c_[x.flatten()[z_condition], y.flatten()[z_condition]], z.flatten()[z_condition]]


def get_count(number):
    s = str(number)
    if '.' in s:
        return abs(s.find('.') - len(s)) - 1
    else:
        return 0
# @TODO

#
# def fitting(self, points, normals):
#     return shape_points
#
#
# def shere_fitting(self, points, normals):
#     return sphere_points
#
# def cube_fitting(self, points, normals):
#     return cube_points
#
# def conus_fitting(self, points, normals):
#     return conus_fitting
#
# def cylinder_fitting(self, points, normals):
#     return cylinder_fitting
#
# def model_fit(self, points, model_points):
#     return True or False
