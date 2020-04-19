import numpy as np
import time
import random
import math
import matplotlib.colors

from points_object import PointsObject
import image_processing
import visualization
import download_point_cloud
import shape_recognition
import moving_prediction


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
    ball = download_point_cloud.download_to_object("preDiploma_PC/ball.pcd")
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


def generate_func(params=np.array([[], [], []]), ttime=np.asarray([])):
    trajectory = np.zeros((ttime.shape[0], 3))
    for i, t in enumerate(ttime):
        for j in range(3):
            param = np.array(params[j])
            powers = np.arange(1, param.shape[0] + 1)
            trajectory[i, j] = np.sum(param * np.power(t, powers))
            # trajectory[i, j] = 1 + 0.1/t + 0.001/t**2
    return trajectory


def generate_found_shapes(object, found_centers, probabilities_of_centers, number_of_points = 100):
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
        current_shape.add_points(points + f_center, current_rgb, 100)
        found_shapes.append(current_shape)
    return found_shapes


if __name__ == "__main__":
    # fill_the_shape_part()

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
    moving_prediction.show_found_functions(found_functions_x, time_, center_trajectory[:, 0], ttime,
                                           real_trajectory[:, 0])
    moving_prediction.show_found_functions(found_functions_y, time_, center_trajectory[:, 1], ttime,
                                           real_trajectory[:, 1])
    moving_prediction.show_found_functions(found_functions_z, time_, center_trajectory[:, 2], ttime,
                                           real_trajectory[:, 2])

    # estimation probability of being in points in time t
    time_of_probability = 1.6
    d_x = 0.2
    # moving_prediction.show_gaussians(found_functions_x, .6, .1)
    prob_x, x = moving_prediction.probability_of_being_between(found_functions_x, time_of_probability, d_x, True)
    prob_y, y = moving_prediction.probability_of_being_between(found_functions_y, time_of_probability, d_x, True)
    prob_z, z = moving_prediction.probability_of_being_between(found_functions_z, time_of_probability, d_x, True)

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
