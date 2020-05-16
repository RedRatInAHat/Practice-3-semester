import numpy as np
import pandas as pd
from scipy.spatial import distance
import sklearn.preprocessing
from scipy.spatial.transform import Rotation as R
import time

import moving_prediction
import shape_recognition
from points_object import PointsObject


def create_new_probabilistic_position(moving_object_points, probability_of_points, environment_object, d_x=0.1,
                                      d_angle=5.):
    environment_points = environment_object.get_points()[0]
    environment_points = np.round(environment_points / d_x) * d_x
    environment_object.set_points(environment_points)
    environment_normals = environment_object.get_normals()
    # crutch
    environment_normals = np.zeros_like(environment_normals)
    environment_normals[:, 1] = 1
    #
    interaction_points_global, interaction_normals_global = find_interactions_global(moving_object_points,
                                                                                     environment_points,
                                                                                     environment_normals)
    if interaction_points_global.shape[0] == 0:
        print("no points found")
        return 0

    # not sure if it has to be used
    # active_points, normals = approximate_environment(interaction_points_global)
    # distances_between_points, interacting_object, interacting_environment = find_interaction_precise(
    #     moving_object_points, active_points, normals)

    drowned_points_idx, drowned_points_distance_to_surface, drowned_points_normals = get_drowned_points_ind_v1(
        moving_object_points, interaction_points_global, interaction_normals_global)
    # drowned_points_idx, drowned_points_vector = get_drowned_points_ind_v2(moving_object_points,
    #                                                                       interaction_points_global,
    #                                                                       interaction_normals_global)

    correction, correction_probability = calculate_probabilistic_correction_v1(probability_of_points[
                                                                                   drowned_points_idx],
                                                                               drowned_points_distance_to_surface,
                                                                               drowned_points_normals, d_x)

    # correction, correction_probability = calculate_probabilistic_correction_v2(probability_of_points[
    #                                                                                drowned_points_idx],
    #                                                                            drowned_points_vector, d_x)
    center = expected_center_of_mass(moving_object_points, probability_of_points)
    rotations, rotations_probabilities, shifts, shift_probabilities = get_rotations(center, moving_object_points[
        drowned_points_idx], probability_of_points[drowned_points_idx], np.repeat(drowned_points_distance_to_surface,
                                                                                  3).reshape(
        [drowned_points_distance_to_surface.shape[0], 3]) * drowned_points_normals, d_x, d_angle)

    new_points, new_probability = rotate_and_shift_points(rotations, rotations_probabilities, shifts,
                                                          shift_probabilities, moving_object_points,
                                                          probability_of_points, center, d_x)

    new_points, new_probability = correct_points(new_points, new_probability, correction,
                                                 correction_probability)
    drowned_points_idx, drowned_points_distance_to_surface, drowned_points_normals = get_drowned_points_ind_v1(
        new_points, interaction_points_global, interaction_normals_global)

    return new_points[np.invert(drowned_points_idx)], new_probability[np.invert(drowned_points_idx)]


def find_interactions_global(moving_object_points, environment_points, environment_normals, number_of_dimensions=3):
    for i in range(number_of_dimensions):
        p_min, p_max = np.min(moving_object_points[:, i]), np.max(moving_object_points[:, i])
        in_this_interval = np.logical_and(environment_points[:, i] >= p_min, environment_points[:, i] <= p_max)
        environment_points = environment_points[in_this_interval]
        environment_normals = environment_normals[in_this_interval]
    return environment_points, environment_normals


def approximate_environment(environment_points, step=0.1, min_number_of_approximated_points=30):
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
        s = np.round(s / step) * step
        temp = PointsObject()
        temp.add_points(s)
        points = np.append(points, s, axis=0)
        normals = np.append(normals, temp.get_normals(), axis=0)
    return points, normals


def find_interaction_precise(moving_object_points, environment_points, environment_normals, influence_distance=0.05):
    distances_between_points = distance.cdist(moving_object_points, environment_points, 'euclidean')
    distances_between_points = np.where(distances_between_points > influence_distance, 0, distances_between_points)

    interacting_environment = np.sum(distances_between_points, axis=0) > 0
    interacting_object = np.sum(distances_between_points, axis=1) > 0
    return distances_between_points, interacting_object, interacting_environment


def get_drowned_points_ind_v1(moving_object_points, environment_points, environment_normals):
    distances_between_points = distance.cdist(moving_object_points, environment_points, 'euclidean')
    closest_point_ind = np.argmin(distances_between_points, axis=1)
    point_point_vector = moving_object_points - environment_points[closest_point_ind]

    scalar_projection = np.einsum('ij,ij->i', point_point_vector,
                                  environment_normals[closest_point_ind]) / np.linalg.norm(
        environment_normals[closest_point_ind])
    not_drowned_points = np.asarray(scalar_projection == np.abs(scalar_projection))

    # crutch to find what is wrong
    # pos_moving_object_points_ind = moving_object_points[:, 1] > 0
    # neg_moving_object_points_ind = moving_object_points[:, 1] < 0
    #
    # false_negative_idx = np.logical_and(pos_moving_object_points_ind, np.invert(not_drowned_points))
    # false_positive_idx = np.logical_and(neg_moving_object_points_ind, not_drowned_points)
    #
    # doesnt_work = closest_point_ind[false_negative_idx]
    # y = np.bincount(doesnt_work)
    # ii = np.nonzero(y)[0]
    # print("false negative")
    # print(np.vstack((ii, y[ii])).T)
    # print(scalar_projection[false_negative_idx])
    # print(environment_normals[ii])
    #
    # doesnt_work = closest_point_ind[false_positive_idx]
    # y = np.bincount(doesnt_work)
    # ii = np.nonzero(y)[0]
    # print("false positive")
    # print(np.vstack((ii, y[ii])).T)

    return np.invert(not_drowned_points), np.amin(distances_between_points, axis=1)[np.invert(not_drowned_points)], \
           environment_normals[closest_point_ind][np.invert(not_drowned_points)]


def get_drowned_points_ind_v2(moving_object_points, environment_points, environment_normals):
    distances_between_points = distance.cdist(moving_object_points, environment_points, 'euclidean')
    closest_point_ind = np.argmin(distances_between_points, axis=1)
    point_point_vector = moving_object_points - environment_points[closest_point_ind]

    scalar_projection = np.einsum('ij,ij->i', point_point_vector,
                                  environment_normals[closest_point_ind]) / np.linalg.norm(
        environment_normals[closest_point_ind])
    not_drowned_points = np.asarray(scalar_projection == np.abs(scalar_projection))
    drowned_points = np.invert(not_drowned_points)

    return drowned_points, -point_point_vector[drowned_points]


def calculate_probabilistic_correction_v1(probabilities, distances, normals, d_x):
    ddistance = np.repeat(distances, 3).reshape([distances.shape[0], 3])
    correction_with_every_point = normals * ddistance
    correction, correction_probability = unique_probabilistic_correction(correction_with_every_point, probabilities,
                                                                         d_x)
    return correction, correction_probability


def calculate_probabilistic_correction_v2(probabilities, vectors, d_x):
    correction, correction_probability = unique_probabilistic_correction(vectors, probabilities, d_x)
    return correction, correction_probability


def unique_probabilistic_correction(corrections, probabilities, d_x):
    df = pd.DataFrame(data=np.column_stack((np.round(corrections / d_x) * d_x, probabilities)),
                      columns=['x', 'y', 'z', 'p'])
    xyzp = df.round(2).groupby(['x', 'y', 'z']).max().reset_index().to_numpy()
    return xyzp[:, :3], xyzp[:, 3]


def correct_points(points, probabilities, correction, correction_probability):
    df = pd.DataFrame(columns=['x', 'y', 'z', 'p'])

    for c, cor in enumerate(correction):
        temp = pd.DataFrame(
            data=np.column_stack((points + cor, probabilities * correction_probability[c])),
            columns=['x', 'y', 'z', 'p'])
        df = df.append(temp, ignore_index=True)
    xyzp = df.round(2).groupby(['x', 'y', 'z']).max().reset_index().to_numpy()
    return xyzp[:, :3], xyzp[:, 3]


def expected_center_of_mass(points, probabilities):
    center = np.zeros([3])
    for i, _ in enumerate(center):
        df = pd.DataFrame(data=np.column_stack((points[:, i], probabilities)), columns=['x', 'p'])
        xp = df.round(2).groupby(['x']).max().reset_index().to_numpy()
        probability = xp[:, 1] / np.sum(xp[:, 1])
        center[i] = np.sum(xp[:, 0] * probability)
    return center


def get_rotations(center, points, probabilities, vectors, d_x, d_angle):
    point_center_vector = center - points
    point_new_center_vector = point_center_vector - vectors

    point_center_vector_length = np.linalg.norm(point_center_vector, axis=1)

    point_center_vector = sklearn.preprocessing.normalize(point_center_vector)
    point_new_center_vector = sklearn.preprocessing.normalize(point_new_center_vector)

    center_to_new_centers_vector = point_new_center_vector * point_center_vector_length[:, np.newaxis] + points - center
    shifts, shift_probabilities = unique_probabilistic_correction(center_to_new_centers_vector, probabilities,
                                                                  d_x)

    v_s = np.cross(point_center_vector, point_new_center_vector)
    s = np.linalg.norm(v_s, axis=1)
    c = np.einsum('ij,ij->i', point_center_vector, point_new_center_vector)
    v_x = np.zeros((points.shape[0], 3, 3))
    v_x[:, 0, 1] = -v_s[:, 2]
    v_x[:, 0, 2] = v_s[:, 1]
    v_x[:, 1, 0] = v_s[:, 2]
    v_x[:, 1, 2] = -v_s[:, 0]
    v_x[:, 2, 0] = -v_s[:, 1]
    v_x[:, 2, 1] = v_s[:, 0]
    temp = (1 - c) / (s * s)

    rotation_matrix = np.eye(3) + v_x + np.matmul(v_x, v_x) * temp[:, np.newaxis, np.newaxis]

    r = R.from_matrix(rotation_matrix)

    rotations, rotations_probabilities = unique_probabilistic_correction(r.as_euler('xyz', degrees=True), probabilities,
                                                                         d_angle)

    return rotations, rotations_probabilities, shifts, shift_probabilities


def rotate_and_shift_points(rotations, rotations_probabilities, shifts, shifts_probabilities, points,
                            points_probabilities, center, d_x):
    df = pd.DataFrame(columns=['x', 'y', 'z', 'p'])

    points_at_center = points - center

    for i, rot in enumerate(rotations):
        r = R.from_euler('xyz', rot, degrees=True)

        new_points = r.apply(points_at_center) + center
        new_points = np.round(new_points / d_x) * d_x
        temp = pd.DataFrame(
            data=np.column_stack((new_points, points_probabilities * rotations_probabilities[i])),
            columns=['x', 'y', 'z', 'p'])
        df = df.append(temp, ignore_index=True)
    xyzp = df.round(2).groupby(['x', 'y', 'z']).max().reset_index().to_numpy()
    rotated_points, rotated_points_probabilities = xyzp[:, :3], xyzp[:, 3]

    df = pd.DataFrame(columns=['x', 'y', 'z', 'p'])

    for s, sif in enumerate(shifts):
        temp = pd.DataFrame(
            data=np.column_stack((rotated_points + sif, rotated_points_probabilities * shifts_probabilities[s])),
            columns=['x', 'y', 'z', 'p'])
        df = df.append(temp, ignore_index=True)
    temp = pd.DataFrame(
        data=np.column_stack((points, points_probabilities)),
        columns=['x', 'y', 'z', 'p'])
    df = df.append(temp, ignore_index=True)
    xyzp = df.round(2).groupby(['x', 'y', 'z']).max().reset_index().to_numpy()

    return xyzp[:, :3], xyzp[:, 3]
