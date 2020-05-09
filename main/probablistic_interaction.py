import numpy as np
from scipy.spatial import distance
import time
import random

import shape_recognition
import visualization
from points_object import PointsObject


def create_new_probabilistic_position(moving_object_points, probability_of_points, environment_object):
    environment_points = environment_object.get_points()[0]
    environment_normals = environment_object.get_normals()
    # crutch
    environment_normals[:, 1] = np.abs(environment_normals[:, 1])
    #
    interaction_points_global, interaction_normals_global = find_interactions_global(moving_object_points,
                                                                                     environment_points,
                                                                                     environment_normals)
    if interaction_points_global.shape[0] == 0:
        print("no points found")
        return 0
    active_points, normals = approximate_environment(interaction_points_global)
    distances_between_points, interacting_object, interacting_environment = find_interaction_precise(
        moving_object_points, active_points)
    drowned_points, not_drowned_points = get_drowned_points(moving_object_points, interaction_points_global, interaction_normals_global)
    temp = PointsObject()
    temp.add_points(drowned_points,
                    np.asarray([[random.random(), random.random(), random.random()]] * drowned_points.shape[0]))
    temp2 = PointsObject()
    temp2.add_points(not_drowned_points, np.asarray([[random.random(), random.random(), random.random()]] * not_drowned_points.shape[0]))
    visualization.visualize([temp, temp2, environment_object])


def find_interactions_global(moving_object_points, environment_points, environment_normals, number_of_dimensions=3):
    for i in range(number_of_dimensions):
        p_min, p_max = np.min(moving_object_points[:, i]), np.max(moving_object_points[:, i])
        in_this_interval = np.logical_and(environment_points[:, i] >= p_min, environment_points[:, i] <= p_max)
        environment_points = environment_points[in_this_interval]
        environment_normals = environment_normals[in_this_interval]
    return environment_points, environment_normals


def approximate_environment(environment_points, min_number_of_approximated_points=30):
    temp = PointsObject()
    temp.add_points(environment_points)
    normals = temp.get_normals()
    found_shapes = shape_recognition.RANSAC(environment_points, normals,
                                            number_of_points_threshold=environment_points.shape[0] * 0.1,
                                            number_of_iterations=10, min_pc_number=environment_points.shape[0] * 0.3,
                                            number_of_subsets=10,
                                            use_planes=True, use_box=False,
                                            use_sphere=False, use_cylinder=True, use_cone=False)
    points = np.empty((0, 3))
    normals = np.empty((0, 3))
    for s in found_shapes:
        temp = PointsObject()
        temp.add_points(s)
        points = np.append(points, s, axis=0)
        normals = np.append(normals, temp.get_normals(), axis=0)
    return points, normals


def find_interaction_precise(moving_object_points, environment_points, influence_distance=0.05):
    distances_between_points = distance.cdist(moving_object_points, environment_points, 'euclidean')
    distances_between_points = np.where(distances_between_points > influence_distance, 0, distances_between_points)

    interacting_environment = np.sum(distances_between_points, axis=0) > 0
    interacting_object = np.sum(distances_between_points, axis=1) > 0
    return distances_between_points, interacting_object, interacting_environment


def get_drowned_points(moving_object_points, environment_points, environment_normals):

    distances_between_points = distance.cdist(moving_object_points, environment_points, 'euclidean')
    closest_point_ind = np.argmin(distances_between_points, axis=1)
    point_point_vector = moving_object_points - environment_points[closest_point_ind]

    scalar_projection = np.diag(np.dot(point_point_vector, environment_normals[closest_point_ind].T) / np.linalg.norm(
        environment_normals[closest_point_ind]))
    drowned_points = np.asarray(scalar_projection == np.abs(scalar_projection))
    print(scalar_projection.shape[0], np.sum(drowned_points))

    

    return moving_object_points[drowned_points], moving_object_points[np.invert(drowned_points)]

    # y = np.asarray([[1, -1, 0]])
    # x = np.asarray([[0, -1, 0]])
    # scalar_projeciton = np.dot(x, y.T) / np.linalg.norm(y)
    # print(scalar_projeciton)
    # x = np.asarray([[0, 1, 0]])
    # scalar_projeciton = np.dot(x, y.T) / np.linalg.norm(y)
    # print(scalar_projeciton)


def calculate_probabilistic_correction():
    pass
