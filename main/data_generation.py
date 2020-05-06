import numpy as np
import matplotlib.colors

from points_object import PointsObject
import image_processing


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
    import vrep_functions
    import image_processing

    """Function for checking if vrep_functions and PointsObject are working fine"""
    client_id = vrep_functions.vrep_connection()
    vrep_functions.vrep_start_sim(client_id)
    kinect_rgb_id = vrep_functions.get_object_id(client_id, 'kinect_rgb')
    kinect_depth_id = vrep_functions.get_object_id(client_id, 'kinect_depth')
    depth_im, rgb_im = vrep_functions.vrep_get_kinect_images(client_id, kinect_rgb_id, kinect_depth_id)
    image_processing.save_image(rgb_im, "preDiploma_PC/", 0, "rgb_box_")
    image_processing.save_image(depth_im, "preDiploma_PC/", 0, "depth_box_")

    print(depth_im.shape, rgb_im.shape)
    vrep_functions.vrep_stop_sim(client_id)

    depth, rgb = image_processing.calculate_point_cloud(rgb_im, depth_im)

    current_object = PointsObject()
    current_object.add_points(depth, rgb)
    current_object.save_all_points("preDiploma_PC/", "box")

def generate_color_of_probable_shapes(found_points, probabilities):
    blue = 0.7
    hsv = np.ones([probabilities.shape[0], 3])
    hsv[:, 0] = blue - probabilities / np.max(probabilities) * blue
    color = matplotlib.colors.hsv_to_rgb(hsv)
    object_to_return = PointsObject()
    object_to_return.add_points(found_points, color)
    return object_to_return
