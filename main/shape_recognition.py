import numpy as np
import random
import math


def RANSAC(xyz, xyz_normals, point_to_model_accuracy=0.1, normal_to_normal_accuracy=0.01, number_of_points_threshold=40,
           number_of_iterations=10, min_pc_number=100):
    """RANSAC method for finding parameters of point cloud and it's primitive shape(s)

    Args:
        xyz (np.ndarray): points of point cloud in xyz format
        xyz_normals (np.ndarray): normals of corresponding points
        point_to_model_accuracy (float): distance between model and point to apply the point as inliner
        normal_to_normal_accuracy (float): angle between model normals and point normal to apply the point as inliner
        number_of_points_threshold (int): number of inliners to apply model as successful
        number_of_iterations (int): number of iterations before the best models will be applied as successful
        min_pc_number (int): number of points recognized too small to search for the model
    """
    found_shapes = []
    itt = 0
    while itt < number_of_iterations and xyz.shape[0] > min_pc_number:
        itt += 1

        # plane fitting
        normal, ro = plane_fitting(xyz, xyz_normals)
        # finding plane inliners
        p_inliners = plane_inliners(xyz, xyz_normals, normal, ro, point_to_model_accuracy, normal_to_normal_accuracy)

        # if number of inliners more than threshold - create the model
        if np.sum(p_inliners) > number_of_points_threshold:
            xyz_in = xyz[p_inliners]
            # create full model and add it to list of models
            found_shapes.append(
                plane_points(normal, ro, np.min(xyz_in[:, 0]), np.max(xyz_in[:, 0]), np.min(xyz_in[:, 1]),
                             np.max(xyz_in[:, 1]), np.min(xyz_in[:, 2]), np.max(xyz_in[:, 2])))
            # delete found points
            xyz = xyz[np.bitwise_not(p_inliners)]
            xyz_normals = xyz_normals[np.bitwise_not(p_inliners)]
    return found_shapes


def plane_fitting(points, normals):
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
    distance_truth = np.where(abs(np.sum(points * normals, axis=1) - plane_ro) < d_accuracy, True, False)
    return np.logical_and(angle_truth, distance_truth)


def angle_between_normals(n1, n2):
    """ Returns the angle in radians between vectors 'n1' and 'n2'"""
    cosang = np.dot(n1, n2)
    sinang = np.linalg.norm(np.cross(n1, n2))
    return np.arctan2(sinang, cosang)


def plane_points(normal, ro, x_min, x_max, y_min, y_max, z_min, z_max, step=0.01):
    """Creating the plane between minimum and maximum points"""
    x = np.arange(x_min, x_max + step, step)
    y = np.arange(y_min, y_max + step, step)

    x = np.tile(x, (y.shape[0], 1))
    y = np.tile(np.array([y]).transpose(), (1, x.shape[1]))

    z = (ro - normal[0] * x - normal[1] * y) / normal[2]
    z_condition = np.where((z.flatten() >= z_min) & (z.flatten() <= z_max), True, False)

    return np.c_[np.c_[x.flatten()[z_condition], y.flatten()[z_condition]], z.flatten()[z_condition]]

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
