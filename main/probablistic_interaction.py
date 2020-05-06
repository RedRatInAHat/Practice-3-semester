import numpy as np


def create_new_probabilistic_position(moving_object_points, probability_of_points, environment_points):
    interaction_points_global = find_interactions_global(moving_object_points, environment_points)
    if interaction_points_global.shape[0] == 0:
        print("no points found")
        return 0
    find_interaction_precise(moving_object_points, environment_points)


def find_interactions_global(moving_object_points, environment_points, number_of_dimensions=3):
    for i in range(number_of_dimensions):
        p_min, p_max = np.min(moving_object_points[:, i]), np.max(moving_object_points[:, i])
        in_this_interval = np.logical_and(environment_points[:, i] >= p_min, environment_points[:, i] <= p_max)
        environment_points = environment_points[in_this_interval]
    return environment_points


def find_interaction_precise(moving_object_points, environment_points):
    print(environment_points.shape[0])
