import numpy as np
import random
import time
import sys

from points_object import PointsObject
import image_processing
import visualization
import download_point_cloud
import shape_recognition
import moving_prediction
import open3d_icp
import data_generation
import probablistic_interaction


def check_RANSAC():
    ball = download_point_cloud.download_to_object("preDiploma_PC/box.pcd")
    full_model = ball

    found_shapes = shape_recognition.RANSAC(full_model.get_points()[0], full_model.get_normals())
    shapes = [full_model]
    for _, s in enumerate(found_shapes):
        new_shape = PointsObject()
        new_shape.add_points(s, np.asarray([[random.random(), random.random(), random.random()]] * s.shape[0]))
        shapes.append(new_shape)
    visualization.visualize_object(shapes)


def check_probabilistic_prediction():
    # load the model
    stable_object = download_point_cloud.download_to_object("models/grey plane.ply", 3000)
    stable_object.scale(0.3)
    stable_object.rotate([90, 0, 0])

    falling_object = download_point_cloud.download_to_object("models/red cube.ply", 3000)
    falling_object.scale(0.3)
    falling_object.shift([0, 3, 0])
    shapes = [falling_object]
    center = falling_object.get_center()

    # temp_1()

    # generate observation data
    rotation_params = np.asarray([[0, 70], [0, 50], [0, 80]])
    moving_params = np.asarray([[0, 0.1, -0.3], [0, -1.5, 0.1], [0, 1, 0]])
    observation_step_time = 0.2
    number_of_observations = 6
    observation_moments = np.arange(0, round(number_of_observations * observation_step_time, 3), observation_step_time)

    rotation_angles_gt, center_position_gt, moving_objects = data_generation.create_movement_path(falling_object,
                                                                                                  rotation_params,
                                                                                                  moving_params,
                                                                                                  observation_moments)

    for i, m in enumerate(moving_objects):
        found_shapes = shape_recognition.RANSAC(m.get_points()[0], m.get_normals())
        moving_objects[i].set_points(found_shapes[-1])

    found_rotation, found_center_positions = moving_prediction.find_observations(moving_objects,
                                                                                 falling_object.get_center())

    print(center_position_gt)
    print(found_center_positions)

    shapes = []
    shapes += moving_objects
    # visualization.visualize(shapes)

    # find functions for xyz trajectory
    start = time.time()
    trajectory_functions_x = moving_prediction.find_functions(observation_moments, found_center_positions[:, 0])
    trajectory_functions_y = moving_prediction.find_functions(observation_moments, found_center_positions[:, 1])
    trajectory_functions_z = moving_prediction.find_functions(observation_moments, found_center_positions[:, 2])

    angle_functions_x = moving_prediction.find_functions(observation_moments, found_rotation[:, 0])
    angle_functions_y = moving_prediction.find_functions(observation_moments, found_rotation[:, 1])
    angle_functions_z = moving_prediction.find_functions(observation_moments, found_rotation[:, 2])
    print(time.time() - start)

    future_time = np.arange(0, round(number_of_observations * observation_step_time * 6, 3), observation_step_time)
    future_angles_gt, future_center_gt, _ = data_generation.create_movement_path(falling_object, rotation_params,
                                                                                 moving_params, future_time)
    moving_prediction.show_found_functions(trajectory_functions_x, observation_moments,
                                           found_center_positions[:, 0], future_time, future_center_gt[:, 0], 't, s',
                                           'x, m', 'x coordinate of center')
    moving_prediction.show_found_functions(trajectory_functions_y, observation_moments,
                                           found_center_positions[:, 1], future_time, future_center_gt[:, 1], 't, s',
                                           'y, m', 'y coordinate of center')
    moving_prediction.show_found_functions(trajectory_functions_z, observation_moments,
                                           found_center_positions[:, 2], future_time, future_center_gt[:, 2], 't, s',
                                           'z, m', 'z coordinate of center')
    moving_prediction.show_found_functions(angle_functions_x, observation_moments,
                                           found_rotation[:, 0], future_time, future_angles_gt[:, 0], 't, s',
                                           'angle, deg', 'x axis angle')
    moving_prediction.show_found_functions(angle_functions_y, observation_moments,
                                           found_rotation[:, 1], future_time, future_angles_gt[:, 1], 't, s',
                                           'angle, deg', 'y axis angle')
    moving_prediction.show_found_functions(angle_functions_z, observation_moments,
                                           found_rotation[:, 2], future_time, future_angles_gt[:, 2], 't, s',
                                           'angle, deg', 'z axis angle')

    # prediction part
    time_of_probability = 2.
    d_x = 0.1
    d_angle = 1
    threshold_p = 0.5

    prob_x, x = moving_prediction.probability_of_being_in_point(trajectory_functions_x, time_of_probability, d_x, True)
    prob_y, y = moving_prediction.probability_of_being_in_point(trajectory_functions_y, time_of_probability, d_x, True)
    prob_z, z = moving_prediction.probability_of_being_in_point(trajectory_functions_z, time_of_probability, d_x, True)

    prob_x_angle, x_angle = moving_prediction.probability_of_being_in_point(angle_functions_x, time_of_probability,
                                                                            d_angle, True)
    prob_y_angle, y_angle = moving_prediction.probability_of_being_in_point(angle_functions_y, time_of_probability,
                                                                            d_angle, True)
    prob_z_angle, z_angle = moving_prediction.probability_of_being_in_point(angle_functions_z, time_of_probability,
                                                                            d_angle, True)

    prediction_object = download_point_cloud.download_to_object("models/red cube.ply", 3000)
    prediction_object.scale(0.3)
    prediction_points = prediction_object.get_points()[0]

    xyz_dict = moving_prediction.get_xyz_probabilities_from_angles_probabilities(prediction_points, x_angle,
                                                                                 prob_x_angle, y_angle,
                                                                                 prob_y_angle, z_angle,
                                                                                 prob_z_angle, d_x, threshold_p)

    points, probabilities = moving_prediction.probability_of_all_points(xyz_dict, prob_x, x, prob_y, y, prob_z, z,
                                                                        threshold_p)

    if xyz_dict == -1:
        print("всё сломалось")
        sys.exit(0)

    high_probabilities = np.where(probabilities >= threshold_p, True, False)
    high_probable_points, high_probable_points_probabilities = points[high_probabilities], probabilities[
        high_probabilities]

    shapes.append(
        data_generation.generate_color_of_probable_shapes(high_probable_points, high_probable_points_probabilities))

    # generate ground truth
    observation_moment = np.asarray([time_of_probability])

    _, _, moving_objects = data_generation.create_movement_path(falling_object, rotation_params, moving_params,
                                                                observation_moment)
    points = moving_objects[0].get_points()[0]
    gt_object = PointsObject()
    gt_object.add_points(points, falling_object.get_points()[1])
    shapes += [gt_object]

    visualization.visualize_object(shapes)
    visualization.get_histogram_of_probabilities(high_probable_points, high_probable_points_probabilities)


def check_physical_objects_interaction():
    # parameters
    shapes = []
    # data generation parameters
    rotation_params = np.asarray([[0, 70], [0, 50], [0, 80]])
    moving_params = np.asarray([[0, 0.1, -0.3], [0, -1.5, 0.1], [0, 1, 0]])
    observation_step_time = 0.2
    number_of_observations = 5
    observation_moments = np.arange(0, round(number_of_observations * observation_step_time, 3), observation_step_time)
    future_time = np.arange(0, round(number_of_observations * observation_step_time * 6, 3), observation_step_time)
    # prediciton parameters
    time_of_probability = 1.9
    d_x = 0.1
    d_angle = 1
    threshold_p = 0.5
    observation_moment = np.asarray([time_of_probability])

    # load the models
    stable_object = download_point_cloud.download_to_object("models/grey plane.ply", 6000)
    stable_object.scale(0.6)
    stable_object.rotate([90, 0, 0])

    falling_object = download_point_cloud.download_to_object("models/red cube.ply", 3000)
    falling_object.scale(0.3)
    falling_object.shift([0, 3, 0])

    prediction_object = download_point_cloud.download_to_object("models/red cube.ply", 3000)
    prediction_object.scale(0.3)
    prediction_points = prediction_object.get_points()[0]

    shapes += [stable_object]

    # generate observation data
    rotation_angles_gt, center_position_gt, moving_objects = data_generation.create_movement_path(falling_object,
                                                                                                  rotation_params,
                                                                                                  moving_params,
                                                                                                  observation_moments)

    found_rotation, found_center_positions = moving_prediction.find_observations(moving_objects,
                                                                                 falling_object.get_center())

    # find functions for xyz trajectory
    start = time.time()
    trajectory_functions_x = moving_prediction.find_functions(observation_moments, found_center_positions[:, 0])
    trajectory_functions_y = moving_prediction.find_functions(observation_moments, found_center_positions[:, 1])
    trajectory_functions_z = moving_prediction.find_functions(observation_moments, found_center_positions[:, 2])

    angle_functions_x = moving_prediction.find_functions(observation_moments, found_rotation[:, 0])
    angle_functions_y = moving_prediction.find_functions(observation_moments, found_rotation[:, 1])
    angle_functions_z = moving_prediction.find_functions(observation_moments, found_rotation[:, 2])
    print(time.time() - start)

    # generation of future positions
    future_angles_gt, future_center_gt, _ = data_generation.create_movement_path(falling_object, rotation_params,
                                                                                 moving_params, future_time)

    # prediction part
    prob_x, x = moving_prediction.probability_of_being_in_point(trajectory_functions_x, time_of_probability, d_x, True)
    prob_y, y = moving_prediction.probability_of_being_in_point(trajectory_functions_y, time_of_probability, d_x, True)
    prob_z, z = moving_prediction.probability_of_being_in_point(trajectory_functions_z, time_of_probability, d_x, True)

    prob_x_angle, x_angle = moving_prediction.probability_of_being_in_point(angle_functions_x, time_of_probability,
                                                                            d_angle, True)
    prob_y_angle, y_angle = moving_prediction.probability_of_being_in_point(angle_functions_y, time_of_probability,
                                                                            d_angle, True)
    prob_z_angle, z_angle = moving_prediction.probability_of_being_in_point(angle_functions_z, time_of_probability,
                                                                            d_angle, True)

    xyz_dict = moving_prediction.get_xyz_probabilities_from_angles_probabilities(prediction_points, x_angle,
                                                                                 prob_x_angle, y_angle,
                                                                                 prob_y_angle, z_angle,
                                                                                 prob_z_angle, d_x, threshold_p)

    points, probabilities = moving_prediction.probability_of_all_points(xyz_dict, prob_x, x, prob_y, y, prob_z, z,
                                                                        threshold_p)

    if xyz_dict == -1:
        print("всё сломалось")
        sys.exit(0)

    high_probabilities = np.where(probabilities >= threshold_p, True, False)
    high_probable_points, high_probable_points_probabilities = points[high_probabilities], probabilities[
        high_probabilities]

    shapes.append(
        data_generation.generate_color_of_probable_shapes(high_probable_points, high_probable_points_probabilities))

    visualization.visualize_object(shapes)

    start = time.time()
    probablistic_interaction.create_new_probabilistic_position(high_probable_points, high_probable_points_probabilities,
                                                               stable_object)
    print(time.time() - start)


if __name__ == "__main__":
    # check_RANSAC()
    # check_probabilistic_prediction()
    check_physical_objects_interaction()
