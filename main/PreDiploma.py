import numpy as np
import time
import random
import math
import matplotlib.colors
import sys

from points_object import PointsObject
import image_processing
import visualization
import download_point_cloud
import shape_recognition
import moving_prediction
import open3d_icp


def create_mask():
    color_mask = image_processing.load_image("tracking_results", "global_two_different3.png")
    binary_mask = np.where(np.sum(color_mask, axis=2), 1, 0)
    image_processing.save_image(binary_mask, "Mask", "mask")


def apply_mask(rgb, depth, mask):
    print(np.max(rgb), np.max(depth), np.max(mask))
    return rgb * mask, depth * mask[:, :, 0]


def create_points_cloud():
    rgb_im = image_processing.load_image("falling ball and cube", "rgb_3.png")
    depth_im = image_processing.load_image("falling ball and cube", "depth_3.png", "depth")
    mask_im = image_processing.load_image("Mask", "mask.png")
    rgb_im, depth_im = apply_mask(rgb_im, depth_im, mask_im / 255)
    return image_processing.calculate_point_cloud(rgb_im / 255, depth_im / 255)


def save_points_cloud():
    points_cloud, points_color = create_points_cloud()
    object = PointsObject()
    object.set_points(points_cloud, points_color)
    object.save_all_points("Test", "ball")


def point_cloud_from_VREP():
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

    # ball = download_point_cloud.download_to_object("preDiploma_PC/box.pcd")
    # visualization.visualize_object([ball])

def save_point_cloud_from_images():
    rgb_im = image_processing.load_image("preDiploma_PC/", "rgb_box_0.png")
    depth_im = image_processing.load_image("preDiploma_PC/", "depth_box_0.png", "depth")
    points, color = image_processing.calculate_point_cloud(rgb_im / 255, depth_im / 255)
    current_object = PointsObject()
    current_object.add_points(points, color)
    current_object.save_all_points("preDiploma_PC/", "box")


def temp():
    ground_truth_vector = [0, 1, 0]
    vector_model = PointsObject()
    vector_model.add_points(np.asarray([ground_truth_vector]), np.asarray([[1, 0, 0]]))
    vector_model_2 = PointsObject()
    vector_model_2.add_points(np.asarray([ground_truth_vector]))
    vector_model.rotate([1, 1, 1], math.radians(60))

    normal = vector_model.get_points()[0][0]
    angle = shape_recognition.angle_between_normals(ground_truth_vector, normal)
    axis = np.cross(ground_truth_vector, normal)
    vector_model.rotate(axis, angle)

    visualization.visualize_object([vector_model, vector_model_2])


def temp_2():
    import open3d as o3d
    full_model = download_point_cloud.download_to_object("models/blue conus.ply", 3000)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(full_model.get_points()[0])
    # pcd = o3d.io.read_point_cloud("models/brown cylinder.ply")
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw_geometries([downpcd])
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([downpcd])


def fill_the_shape_part():
    # save_points_cloud()
    ball = PointsObject()
    ball = download_point_cloud.download_to_object("preDiploma_PC/box.pcd")
    # visualization.visualize_object([ball])
    full_model = ball
    # full_model = download_point_cloud.download_to_object("models/blue conus.ply", 3000)
    # full_model.scale(0.1)
    # full_model.shift([0.01, 0.05, 0.01])
    # full_model.rotate([1, 1, 1], math.radians(35))
    #
    # full_model_2 = download_point_cloud.download_to_object("models/orange sphere.ply", 3000)
    # full_model_2.scale(0.1)
    # full_model_2.shift([-0.1, -0.1, 0.1])
    # full_model.rotate([1, 1, 1], math.radians(60))
    # full_model.add_points(full_model_2.get_points()[0], full_model_2.get_points()[1])
    #
    # full_model_2 = download_point_cloud.download_to_object("models/orange sphere.ply", 3000)
    # full_model_2.scale(0.1)
    # full_model_2.shift([-0.01, 0.1, 0.3])
    # full_model.rotate([1, 0, 1], math.radians(30))
    # full_model.add_points(full_model_2.get_points()[0], full_model_2.get_points()[1])
    # visualization.visualize_object([full_model])

    # temp()
    # temp_2()

    start = time.time()
    found_shapes = shape_recognition.RANSAC(full_model.get_points()[0], full_model.get_normals())
    print(time.time() - start)
    shapes = [full_model]
    for _, s in enumerate(found_shapes):
        new_shape = PointsObject()
        new_shape.add_points(s, np.asarray([[random.random(), random.random(), random.random()]] * s.shape[0]))
        shapes.append(new_shape)
    visualization.visualize_object(shapes)


def generate_trajectory(points, trajectory_fun, trajectory_param, ttime):
    current_points = points.get_points()[0]
    center = np.mean(current_points, axis=0)
    shapes_to_return = []
    center_trajectory = []
    shifts = trajectory_fun(trajectory_param, ttime)
    for shift in shifts:
        current_shape = PointsObject()
        current_shape.add_points(current_points + shift)
        shapes_to_return.append(current_shape)

        center_trajectory.append(center + shift)
    return shapes_to_return, np.asarray(center_trajectory)


def generate_func(params, ttime):
    trajectory = np.zeros((ttime.shape[0], 3))
    for i in range(3):
        trajectory[:, i] = np.polyval(params[i], ttime)
    return trajectory


def generate_found_shapes(object, found_centers, probabilities_of_centers, number_of_points=100):
    found_shapes = []
    blue = 0.7
    hsv = np.ones([probabilities_of_centers.shape[0], 3])
    hsv[:, 0] = blue - probabilities_of_centers / np.max(probabilities_of_centers) * blue
    rgb = matplotlib.colors.hsv_to_rgb(hsv)

    center = object.get_center()
    points = object.get_points()[0]
    points -= center

    for f, f_center in enumerate(found_centers):
        current_shape = PointsObject()
        current_rgb = np.zeros([points.shape[0], 3]) + rgb[f]
        current_shape.add_points(points + f_center, current_rgb, number_of_points)
        found_shapes.append(current_shape)
    return found_shapes


def generate_color_shapes(found_points, probabilities):
    blue = 0.7
    hsv = np.ones([probabilities.shape[0], 3])
    hsv[:, 0] = blue - probabilities / np.max(probabilities) * blue
    color = matplotlib.colors.hsv_to_rgb(hsv)
    object_to_return = PointsObject()
    object_to_return.add_points(found_points, color)
    return object_to_return


def linear_movement():
    # load the model
    stable_object = download_point_cloud.download_to_object("models/grey plane.ply", 3000)
    stable_object.scale(0.2)
    stable_object.rotate([1, 0, 0], math.radians(90))

    falling_object = download_point_cloud.download_to_object("models/orange sphere.ply", 3000)
    falling_object.scale(0.4)
    falling_object.shift([0, 2, 0])

    shapes = [stable_object]

    # generating parameters and trajectory
    number_of_steps = 5
    step_time = 0.2
    parameters = np.array([[1, -3], [0, -9.8], []])
    # training data
    time_ = np.arange(step_time, (number_of_steps + 1) * step_time, step_time)
    points_trajectory, center_trajectory = generate_trajectory(falling_object, generate_func, parameters, time_)
    # data to compare
    ttime = np.arange(step_time, (number_of_steps + 1) * step_time * 1.5, step_time / 10)
    _, real_trajectory = generate_trajectory(falling_object, generate_func, parameters, ttime)

    # add noise
    center_trajectory += np.random.normal(0, 0.05, center_trajectory.shape)

    # find functions for xyz trajectory
    start = time.time()
    found_functions_x = moving_prediction.find_functions(time_, center_trajectory[:, 0])
    found_functions_y = moving_prediction.find_functions(time_, center_trajectory[:, 1])
    found_functions_z = moving_prediction.find_functions(time_, center_trajectory[:, 2])
    print(time.time() - start)

    # show prediction results
    visualization.show_found_functions(found_functions_x, time_, center_trajectory[:, 0], ttime,
                                           real_trajectory[:, 0])
    visualization.show_found_functions(found_functions_y, time_, center_trajectory[:, 1], ttime,
                                           real_trajectory[:, 1])
    visualization.show_found_functions(found_functions_z, time_, center_trajectory[:, 2], ttime,
                                           real_trajectory[:, 2])

    # estimation probability of being in points in time t
    time_of_probability = 1.
    d_x = 0.2
    # moving_prediction.show_gaussians(found_functions_x, .6, .1)
    prob_x, x = moving_prediction.probability_of_being_in_point(found_functions_x, time_of_probability, d_x, True)
    prob_y, y = moving_prediction.probability_of_being_in_point(found_functions_y, time_of_probability, d_x, True)
    prob_z, z = moving_prediction.probability_of_being_in_point(found_functions_z, time_of_probability, d_x, True)

    # create points where probability > threshold_p
    threshold_p = 0.7
    prob_x, x = prob_x[prob_x > threshold_p], x[prob_x > threshold_p]
    prob_y, y = prob_y[prob_y > threshold_p], y[prob_y > threshold_p]
    prob_z, z = prob_z[prob_z > threshold_p], z[prob_z > threshold_p]
    if x.shape[0] * y.shape[0] * z.shape[0] > 10000:
        print("Слишком много точек")
    else:
        points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
        probabilities = np.array(np.meshgrid(prob_x, prob_y, prob_z)).T.reshape(-1, 3)

        high_probabilities = np.where(np.prod(probabilities, axis=1) >= threshold_p, True, False)
        high_probable_points, high_probable_points_probabilities = points[high_probabilities], \
                                                                   np.prod(probabilities, axis=1)[high_probabilities]
        # print(high_probable_points, np.prod(high_probable_points_probabilities, axis=1))
        # shapes += points_trajectory

        shapes += generate_found_shapes(falling_object, high_probable_points, high_probable_points_probabilities)
        time_ = np.asarray([time_of_probability])
        shapes += generate_trajectory(falling_object, generate_func, parameters, time_)[0]
        visualization.visualize_object(shapes)


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


def find_observations_v1(objects, initial_center):
    found_dt_rotations = np.zeros((len(objects) - 1, 3))
    found_dt_center_shifts = np.zeros((len(objects) - 1, 3))
    for i in range(len(objects) - 1):
        source_points = objects[i].get_points()[0]
        target_points = objects[i + 1].get_points()[0]
        transformation = open3d_icp.get_transformation_matrix_p2p(source_points, target_points)
        found_dt_rotations[i] = moving_prediction.get_angles_from_transformation(transformation[:3, :3])
        found_dt_center_shifts[i] = moving_prediction.get_movement_from_transformation(transformation, source_points,
                                                                                       target_points)
    # print(found_dt_rotations, found_dt_center_shifts)
    found_rotations = np.zeros((len(objects), 3))
    found_rotations[1:] = np.cumsum(found_dt_rotations, axis=0)
    found_center_shifts = np.zeros((len(objects), 3))
    found_center_shifts[1:] = np.cumsum(found_dt_center_shifts, axis=0)

    return found_rotations, found_center_shifts + initial_center


def find_observations_v2(objects, initial_center, initial_points):
    found_rotations = np.zeros((len(objects) - 1, 3))
    found_center_shifts = np.zeros((len(objects) - 1, 3))
    for i in range(len(objects) - 1):
        target_points = objects[i + 1].get_points()[0]
        transformation = open3d_icp.get_transformation_matrix_p2p(initial_points, target_points)
        found_rotations[i] = moving_prediction.get_angles_from_transformation(transformation[:3, :3])
        found_center_shifts[i] = moving_prediction.get_movement_from_transformation(transformation, initial_points,
                                                                                    target_points)
    # print(found_dt_rotations, found_dt_center_shifts)
    found_rotations = np.vstack((np.zeros(3), found_rotations))
    found_center_shifts = np.vstack((np.zeros(3), found_center_shifts))
    return found_rotations, found_center_shifts + initial_center


def find_observations_v3(objects, initial_center):
    from scipy.spatial.transform import Rotation as R

    found_dt_rotations = np.zeros((len(objects) - 1, 3))
    found_dt_center_shifts = np.zeros((len(objects) - 1, 3))
    rotation_funcs = []
    for i in range(len(objects) - 1):
        source_points = objects[i].get_points()[0]
        target_points = objects[i + 1].get_points()[0]
        transformation = open3d_icp.get_transformation_matrix_p2p(source_points, target_points)
        rotation_funcs.append(R.from_matrix(transformation[:3, :3]))
        found_dt_center_shifts[i] = moving_prediction.get_movement_from_transformation(transformation, source_points,
                                                                                       target_points)

    for i in range(1, len(rotation_funcs)):
        rotation_funcs[i] = rotation_funcs[i] * rotation_funcs[i - 1]
        found_dt_rotations[i] = rotation_funcs[i].as_euler('xyz', degrees=True)

    found_rotations = np.vstack((np.zeros(3), found_dt_rotations))
    found_center_shifts = np.zeros((len(objects), 3))
    found_center_shifts[1:] = np.cumsum(found_dt_center_shifts, axis=0)
    #
    return found_rotations, found_center_shifts + initial_center


def find_observations(objects, initial_center):
    found_rotations = np.zeros((len(objects) - 1, 3))
    found_dt_center_shifts = np.zeros((len(objects) - 1, 3))
    previous_transformation = np.eye(4)
    for i in range(len(objects) - 1):
        source_points = objects[i].get_points()[0]
        target_points = objects[i + 1].get_points()[0]
        transformation = open3d_icp.get_transformation_matrix_p2p(source_points, target_points)
        current_transformation = transformation.dot(previous_transformation)
        previous_transformation = np.copy(current_transformation)
        found_rotations[i] = moving_prediction.get_angles_from_transformation(current_transformation[:3, :3])
        found_dt_center_shifts[i] = moving_prediction.get_movement_from_transformation(transformation, source_points,
                                                                                       target_points)
    # print(found_dt_rotations, found_dt_center_shifts)
    found_rotations = np.vstack((np.zeros(3), found_rotations))
    found_center_shifts = np.zeros((len(objects), 3))
    found_center_shifts[1:] = np.cumsum(found_dt_center_shifts, axis=0)

    return found_rotations, found_center_shifts + initial_center


def temp_1():
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as R

    rotation = np.asarray([15, 10, 20])
    current_rotation = np.asarray([0, 0, 0])
    falling_object = download_point_cloud.download_to_object("models/red cube.ply", 3000)
    falling_object.scale(0.3)
    previous_points = falling_object.get_points()[0]
    initial_points = np.copy(previous_points)
    rotation_from_initial = []
    rotation_from_previous = []
    rotation_quarterions = []
    t = np.arange(0, 15, 1)
    for i in t:
        current_rotation += rotation
        falling_object = download_point_cloud.download_to_object("models/red cube.ply", 3000)
        falling_object.scale(0.3)
        falling_object.rotate(current_rotation)
        current_points = falling_object.get_points()[0]
        transformation = open3d_icp.get_transformation_matrix_p2p(initial_points, current_points)
        rotation_from_initial.append(moving_prediction.get_angles_from_transformation(transformation[:3, :3]))
        transformation = open3d_icp.get_transformation_matrix_p2p(previous_points, current_points)
        rotation_from_previous.append(moving_prediction.get_angles_from_transformation(transformation[:3, :3]))
        previous_points = np.copy(current_points)
    angles = np.cumsum(rotation_from_previous, axis=0)
    rotation_from_initial = np.asarray(rotation_from_initial)
    rotation_from_previous = np.asarray(rotation_from_previous)
    angles = np.asarray(angles)
    for i in range(3):
        plt.plot(t, rotation_from_initial[:, i], '-', t, angles[:, 0], '--')
        plt.show()
        plt.plot(t, rotation_from_previous[:, i], '-')
        plt.show()


def get_histogram(points, probabilities, step=0.1, y_start=0, y_stop=1.5):
    import matplotlib.pyplot as plt
    points_there = np.bitwise_and(points[:, 1] >= y_start, points[:, 1] <= y_stop)
    points = points[points_there][:, [0, 2]]
    probabilities = probabilities[points_there]
    dict = moving_prediction.sum_dif_probabilities_of_same_points({}, points, probabilities)
    probabilities = np.fromiter(dict.values(), dtype=float)
    points = moving_prediction.tuple_to_array(np.fromiter(dict.keys(), dtype=np.dtype('float, float')))
    x_min, x_max = np.min(points[:, 0]) - 2*step, np.max(points[:, 0]) + 2*step
    y_min, y_max = np.min(points[:, 1]) - 2*step, np.max(points[:, 1]) + 2*step
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


def observation_momental():
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
    number_of_observations = 5
    observation_moments = np.arange(0, round(number_of_observations * observation_step_time, 3), observation_step_time)

    rotation_angles_gt, center_position_gt, moving_objects = create_movement_path(falling_object, rotation_params,
                                                                                  moving_params, observation_moments)

    for i, m in enumerate(moving_objects):
        found_shapes = shape_recognition.RANSAC(m.get_points()[0], m.get_normals())
        moving_objects[i].set_points(found_shapes[-1])

    found_rotation, found_center_positions = find_observations(moving_objects, falling_object.get_center())

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
    future_angles_gt, future_center_gt, _ = create_movement_path(falling_object, rotation_params, moving_params,
                                                                 future_time)
    visualization.show_found_functions(trajectory_functions_x, observation_moments,
                                           found_center_positions[:, 0], future_time, future_center_gt[:, 0], 't, s',
                                           'x, m', 'x coordinate of center')
    visualization.show_found_functions(trajectory_functions_y, observation_moments,
                                           found_center_positions[:, 1], future_time, future_center_gt[:, 1], 't, s',
                                           'y, m', 'y coordinate of center')
    visualization.show_found_functions(trajectory_functions_z, observation_moments,
                                           found_center_positions[:, 2], future_time, future_center_gt[:, 2], 't, s',
                                           'z, m', 'z coordinate of center')
    visualization.show_found_functions(angle_functions_x, observation_moments,
                                           found_rotation[:, 0], future_time, future_angles_gt[:, 0], 't, s',
                                           'angle, deg', 'x axis angle')
    visualization.show_found_functions(angle_functions_y, observation_moments,
                                           found_rotation[:, 1], future_time, future_angles_gt[:, 1], 't, s',
                                           'angle, deg', 'y axis angle')
    visualization.show_found_functions(angle_functions_z, observation_moments,
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

    shapes.append(generate_color_shapes(high_probable_points, high_probable_points_probabilities))

    # generate ground truth
    observation_moment = np.asarray([time_of_probability])

    _, _, moving_objects = create_movement_path(falling_object, rotation_params, moving_params, observation_moment)
    points = moving_objects[0].get_points()[0]
    gt_object = PointsObject()
    gt_object.add_points(points, falling_object.get_points()[1])
    shapes += [gt_object]

    visualization.visualize_object(shapes)
    get_histogram(high_probable_points, high_probable_points_probabilities)


if __name__ == "__main__":
    # point_cloud_from_VREP()
    save_point_cloud_from_images()
    fill_the_shape_part()
    # linear_movement()

    # observation_momental()
