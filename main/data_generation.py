import numpy as np
import matplotlib.colors

from points_object import PointsObject
import image_processing
import vrep_functions


def generate_func(params, ttime):
    trajectory = np.zeros((ttime.shape[0], 3))
    for i in range(3):
        trajectory[:, i] = np.polyval(params[i], ttime)
    return trajectory


def create_movement_path(object, angles_params, shift_params, observation_time):
    angles_params = np.flip(angles_params, axis=1)
    shift_params = np.flip(shift_params, axis=1)

    number_of_points = observation_time.shape[0]

    shifts = generate_func(shift_params, observation_time)
    rotations = generate_func(angles_params, observation_time)

    points = []
    zero_t_points = object.get_points()[0]
    for i, t in enumerate(observation_time):
        points.append(PointsObject())
        points[-1].add_points(zero_t_points)
        points[-1].rotate(rotations[i])
        points[-1].shift(shifts[i])
    return rotations, shifts + object.get_center(), points


def save_point_cloud_from_images():
    rgb_im = image_processing.load_image("preDiploma_PC/", "rgb_box_0.png")
    depth_im = image_processing.load_image("preDiploma_PC/", "depth_box_0.png", "depth")
    points, color = image_processing.calculate_point_cloud(rgb_im / 255, depth_im / 255)
    current_object = PointsObject()
    current_object.add_points(points, color)
    current_object.save_all_points("preDiploma_PC/", "box")


def save_point_cloud_from_VREP():
    depth_im, rgb_im = save_images_from_VREP("preDiploma_PC/")

    depth, rgb = image_processing.calculate_point_cloud(rgb_im, depth_im)

    current_object = PointsObject()
    current_object.add_points(depth, rgb)
    current_object.save_all_points("preDiploma_PC/", "box")


def save_images_from_VREP(path="3d_map/"):
    client_id = vrep_functions.vrep_connection()
    vrep_functions.vrep_start_sim(client_id)
    kinect_rgb_id = vrep_functions.get_object_id(client_id, 'kinect_rgb')
    kinect_depth_id = vrep_functions.get_object_id(client_id, 'kinect_depth')
    depth_im, rgb_im = vrep_functions.vrep_get_kinect_images(client_id, kinect_rgb_id, kinect_depth_id)

    depth_im, rgb_im = np.flip(depth_im, (0)), np.flip(rgb_im, (0))

    vrep_functions.vrep_stop_sim(client_id)

    image_processing.save_image(rgb_im, path, 0, "room_rgb")
    image_processing.save_image(depth_im, path, 0, "room_depth")
    return depth_im, rgb_im


def generate_color_of_probable_shapes(found_points, probabilities):
    blue = 0.7
    hsv = np.ones([probabilities.shape[0], 3])
    hsv[:, 0] = blue - probabilities / np.max(probabilities) * blue
    color = matplotlib.colors.hsv_to_rgb(hsv)
    object_to_return = PointsObject()
    object_to_return.add_points(found_points, color)
    return object_to_return


def reduce_environment_points(environment_points, d_x):
    environment_points = np.round(environment_points / d_x) * d_x
    return environment_points, [np.unique(environment_points[:, 0]), np.unique(environment_points[:, 1]), np.unique(
        environment_points[:, 2])]
