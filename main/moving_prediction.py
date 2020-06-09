import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy.spatial.transform import Rotation as R

import open3d_icp
from set_of_math_functions import *


class MovementFunctions:

    def __init__(self, functions, max_number_of_params, number_of_gaussians=None, coefficient_of_forgetting=1., t=0.,
                 weight_threshold=0.1):
        if number_of_gaussians is None:
            number_of_gaussians = len(functions)
        self.functions_coefficients = np.zeros((number_of_gaussians, max_number_of_params))
        self.covariance_matrices = np.zeros((number_of_gaussians, max_number_of_params, max_number_of_params))
        self.standard_deviations = np.zeros((number_of_gaussians, max_number_of_params))
        self.weights = np.zeros(number_of_gaussians)
        self.time_of_creation = np.full(number_of_gaussians, t)
        self.coefficient_of_forgetting = coefficient_of_forgetting
        self.forgetting = np.zeros(number_of_gaussians)
        self.weight_threshold = weight_threshold
        self.extract_parameters(functions)

    def extract_parameters(self, functions):
        for f, func in enumerate(functions):
            f_c = functions[func]['function_params']
            self.functions_coefficients[f, -f_c.shape[0]:] = f_c
            covs = functions[func]['covariance_matrix']
            self.covariance_matrices[f, -covs.shape[0]:, -covs.shape[0]:] = covs
            self.standard_deviations[f] = np.sqrt(np.diag(self.covariance_matrices[f]))
            self.weights[f] = 1 / functions[func]['error']
        self.weights /= np.sum(self.weights)

    def get_gaussians_parameters_at_time(self, moment):
        means, standard_deviations = self.get_solution(self.functions_coefficients, self.standard_deviations, moment)
        return self.values_selection(means, standard_deviations, moment)

    def get_velocity(self, moment):
        coefficients = np.arange(self.functions_coefficients.shape[1] - 1, 0, -1)
        velocity_coefficients = self.functions_coefficients[:, :-1] * coefficients[np.newaxis, :]
        velocity_sd = self.standard_deviations[:, :-1] * coefficients[np.newaxis, :]
        means, standard_deviations = self.get_solution(velocity_coefficients, velocity_sd, moment)
        return self.values_selection(means, standard_deviations, moment)

    def values_selection(self, means, standard_deviations, moment):
        time_difference = moment - self.time_of_creation
        weights = np.copy(self.weights)
        weights[time_difference < 0] = 0
        weights = weights / (1 + self.coefficient_of_forgetting * self.forgetting)
        w_condition = weights >= self.weight_threshold
        return means[w_condition], standard_deviations[w_condition], weights[w_condition] / np.sum(weights[w_condition])

    def get_solution(self, coefficients, sd, moment):
        means = np.polyval(coefficients.T, moment)
        min_deviation = np.abs(means - np.polyval(coefficients.T - sd.T, moment))
        max_deviation = np.abs(means - np.polyval(coefficients.T + sd.T, moment))
        standard_deviations = np.maximum(min_deviation, max_deviation)
        return means, standard_deviations

    def update_gaussians(self, means, standard_deviations, weights, time):
        new_means = np.zeros((means.shape[0], self.functions_coefficients.shape[1]))
        new_covariance_matrices = np.zeros(
            (means.shape[0], self.functions_coefficients.shape[1], self.functions_coefficients.shape[1]))
        new_standard_deviations = np.zeros((means.shape[0], self.functions_coefficients.shape[1]))

        new_means[:, -means.shape[1]:] = np.flip(means)
        new_standard_deviations[:, -means.shape[1]:] = np.flip(standard_deviations)
        row, col = np.diag_indices(new_standard_deviations.shape[1])
        new_covariance_matrices[:, row, col] = new_standard_deviations

        self.functions_coefficients = np.append(self.functions_coefficients, new_means, axis=0)
        self.covariance_matrices = np.append(self.covariance_matrices, new_covariance_matrices, axis=0)
        self.standard_deviations = np.append(self.standard_deviations, new_standard_deviations, axis=0)
        self.weights = np.append(self.weights, weights, axis=0)
        self.time_of_creation = np.append(self.time_of_creation, np.full(means.shape[0], time))
        self.forgetting += np.max(weights)
        self.forgetting = np.append(self.forgetting, np.zeros(means.shape[0]))

    def get_number_of_gaussians(self):
        return self.functions_coefficients.shape[0]

    def get_gaussian_presentation(self, number_of_gaussian, time_row):
        exact_time_row = time_row[time_row >= self.time_of_creation[number_of_gaussian]]
        means = np.polyval(self.functions_coefficients[number_of_gaussian], exact_time_row)
        return means, exact_time_row

class PointsVelocities():

    def __init__(self):
        self.points_idx = np.empty(0)
        self.points_linear_velocity = np.empty((0, 3))
        self.points_linear_sd = np.empty((0, 3))
        self.points_linear_weight = np.empty(0)
        self.points_linear_angular_velocity = np.empty((0, 3))
        self.points_linear_angular_sd = np.empty((0, 3))
        self.points_linear_angular_weight = np.empty(0)
        self.radii = np.empty((0, 3))
        self.radii_length = np.empty(0)
        self.center = np.empty((0, 3))
        self.angle = np.empty((0, 3))
        self.center_sd = np.empty((0, 3))
        self.angle_sd = np.empty((0, 3))

    def add_points(self, center, center_sd, angle, angle_sd, linear_velocity, points_idx, radii, angular_velocities,
                   linear_weight, angular_weight, linear_sd, angular_sd):
        all_points_idx = np.repeat(points_idx, angular_velocities.shape[0], axis=0)
        self.points_idx = np.append(self.points_idx, all_points_idx).astype(int)
        self.radii = np.append(self.radii, np.repeat(radii, angular_velocities.shape[0], axis=0), axis=0)
        self.radii_length = np.append(self.radii_length,
                                      np.linalg.norm(np.repeat(radii, angular_velocities.shape[0], axis=0), axis=1))

        self.center = np.append(self.center, np.tile([center], (all_points_idx.shape[0], 1)), axis=0)
        self.angle = np.append(self.angle, np.tile(angle, (points_idx.shape[0], 1)), axis=0)
        self.center_sd = np.append(self.center_sd, np.tile([center_sd], (all_points_idx.shape[0], 1)), axis=0)
        self.angle_sd = np.append(self.angle_sd, np.tile(angle_sd, (points_idx.shape[0], 1)), axis=0)

        self.points_linear_velocity = np.append(self.points_linear_velocity,
                                                np.tile([linear_velocity], (all_points_idx.shape[0], 1)), axis=0)
        self.points_linear_sd = np.append(self.points_linear_sd, np.tile([linear_sd], (all_points_idx.shape[0], 1)),
                                          axis=0)
        self.points_linear_weight = np.append(self.points_linear_weight,
                                              np.tile([linear_weight], all_points_idx.shape[0]))

        w_l_v, w_l_s = self.calculate_angular_to_linear_velocity(np.tile(angular_velocities, (points_idx.shape[0], 1)),
                                                                 np.tile(angular_sd, (points_idx.shape[0], 1)),
                                                                 np.repeat(radii, angular_velocities.shape[0], axis=0))
        self.points_linear_angular_velocity = np.append(self.points_linear_angular_velocity, w_l_v, axis=0)
        self.points_linear_angular_sd = np.append(self.points_linear_angular_sd, w_l_s, axis=0)
        self.points_linear_angular_weight = np.append(self.points_linear_angular_weight,
                                                      np.tile(angular_weight, points_idx.shape[0]))

        # print(self.points_idx.shape, self.radii_length.shape, self.center_sd.shape, self.points_linear_velocity.shape,
        #       self.points_linear_sd.shape, self.points_linear_weight.shape, self.points_linear_angular_velocity.shape,
        #       self.points_linear_angular_sd.shape, self.points_linear_angular_weight.shape)

    def calculate_angular_to_linear_velocity(self, angular_velocity, angular_velocity_sd, radii):
        linear_velocity = np.cross(angular_velocity, radii)
        linear_velocity_min = np.cross(angular_velocity - angular_velocity_sd, radii)
        linear_velocity_max = np.cross(angular_velocity + angular_velocity_sd, radii)

        return linear_velocity, np.maximum(np.abs(linear_velocity - linear_velocity_min),
                                           np.abs(linear_velocity - linear_velocity_max))

    def repeat_n_times(self, n):
        self.points_idx = np.repeat(self.points_idx, n)
        self.center = np.repeat(self.center, n, axis=0)
        self.angle = np.repeat(self.angle, n, axis=0)
        self.center_sd = np.repeat(self.center_sd, n, axis=0)
        self.angle_sd = np.repeat(self.angle_sd, n, axis=0)
        self.points_linear_velocity = np.repeat(self.points_linear_velocity, n, axis=0)
        self.points_linear_sd = np.repeat(self.points_linear_sd, n, axis=0)
        self.points_linear_weight = np.repeat(self.points_linear_weight, n)
        self.points_linear_angular_velocity = np.repeat(self.points_linear_angular_velocity, n, axis=0)
        self.points_linear_angular_sd = np.repeat(self.points_linear_angular_sd, n, axis=0)
        self.points_linear_angular_weight = np.repeat(self.points_linear_angular_weight, n)
        self.radii = np.repeat(self.radii, n, axis=0)
        self.radii_length = np.repeat(self.radii_length, n)


def generate_poly_trajectory(x=None, trajectory_param=None, number_of_steps=10, step=0.01, return_x=False):
    if x is None:
        x = np.arange(0, number_of_steps * step, step)
    y = np.polyval(trajectory_param, x)
    # y = np.zeros(x.shape[0])
    # powers = np.arange(trajectory_param.shape[0])
    # for i, x_ in enumerate(x):
    #     y[i] = np.sum(trajectory_param * np.power(x_, powers))
    if return_x:
        return x, y
    else:
        return y


def generate_func_trajectory(trajectory_fun, trajectory_param, x=None, number_of_steps=10, step=0.01, return_x=False):
    if x is None:
        x = np.arange(0, number_of_steps * step, step)
    y = trajectory_fun(x, *trajectory_param)
    if return_x:
        return x, y
    else:
        return y


def trajectory_poly_fitting(trajectory_time, trajectory_points, poly_number):
    popt, pcov = np.polyfit(trajectory_time, trajectory_points, poly_number, cov=True)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, pcov


def trajectory_fun_fitting(trajectory_time, trajectory_points, func, params_number):
    popt, pcov = curve_fit(func, trajectory_time, trajectory_points, p0=np.ones(params_number))
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def find_functions(t, points, threshold_accuracy=1e-01):
    found_functions = {}
    max_poly = points.shape[0] - 1 if points.shape[0] - 1 <= 15 else 15
    for i in range(max_poly):
        poly_params, poly_sd, poly_cov = trajectory_poly_fitting(t, points, i)
        y_ = generate_poly_trajectory(t, poly_params)
        if np.sum(np.isnan(y_)) > 0:
            continue
        mean_error = sum_of_the_squares_of_the_residuals(points, y_)
        if mean_error < threshold_accuracy:
            found_functions["polyfit_model" + str(i)] = {'function': generate_poly_trajectory,
                                                         'function_params': poly_params, 'error': mean_error,
                                                         'standard_deviation': poly_sd, 'covariance_matrix': poly_cov}
    # use additional functions
    # functions = get_all_functions()
    # for f, func in enumerate(functions):
    #     try:
    #         func_params, func_sd = trajectory_fun_fitting(t, points, func, functions[func])
    #         y_ = generate_func_trajectory(func, func_params, t)
    #         if np.sum(np.isnan(y_)) > 0:
    #             continue
    #         mean_error = sum_of_the_squares_of_the_residuals(points, y_) / points.shape[0]
    #         if mean_error < threshold_accuracy and not np.any(np.isinf(func_sd)):
    #             found_functions["curve_fit_model" + str(f)] = {'function': func,
    #                                                            'function_params': func_params, 'error': mean_error,
    #                                                            'standard_deviation': func_sd}
    #     except:
    #         pass

    if not found_functions:
        print('function wasn\'t found')

    return found_functions


def find_center_and_rotation_functions(observation_moments, found_center_positions, found_rotation):
    center_functions = []
    angles_functions = []
    for i in range(3):
        center_functions.append(find_functions(observation_moments, found_center_positions[:, i]))
        angles_functions.append(find_functions(observation_moments, found_rotation[:, i]))
    return center_functions, angles_functions


def show_found_functions_with_deviation(found_functions, t, points, tt, real_y):
    trajectory = get_future_points(found_functions, tt)
    trajectory_up = get_future_points(found_functions, tt, 'up')
    trajectory_down = get_future_points(found_functions, tt, 'down')
    legend = ["given points", 'ground truth']
    plt.plot(t, points, 'o', tt, real_y, '-')
    for func, y_, y_up, y_down in zip(found_functions, trajectory, trajectory_up, trajectory_down):
        plt.plot(tt, y_, '--')
        legend.append(func)
        plt.fill_between(tt, y_up, y_down, alpha=.25)
    plt.legend(legend)
    plt.show()


def get_future_points(found_functions, tt, deviation=''):
    y = np.zeros((len(found_functions), tt.shape[0]))
    for f_, func in enumerate(found_functions):
        f = found_functions[func]['function']
        f_p = found_functions[func]['function_params']
        if deviation == 'up':
            f_p += found_functions[func]['standard_deviation']
        elif deviation == 'down':
            f_p -= found_functions[func]['standard_deviation']
        if "polyfit_model" in func:
            y[f_] = f(tt, f_p)
        else:
            y[f_] = generate_func_trajectory(f, f_p, tt)
    return y


def show_gaussians(found_functions, t=0, step=.1):
    mean, standard_deviation, weights = get_gaussian_params(found_functions, t)
    p_min, p_max, s_max = np.min(mean), np.max(mean), 4 * np.max(standard_deviation)
    points = np.arange(p_min - s_max, p_max + s_max, step)
    if points.shape[0] < 3:
        points = np.asarray([p_min - s_max - step, p_min - s_max, p_max + s_max, p_max + s_max + step])
    all_c = np.zeros(points.shape[0])
    for f, func in enumerate(found_functions):
        d = stats.norm(mean[f], standard_deviation[f])
        c = d.cdf(points) * weights[f]
        all_c += c
        # plt.plot(points, c)
        # plt.legend([func])
        # plt.show()
    plt.plot(points, all_c / np.sum(all_c))
    plt.legend("mixture")
    plt.show()


def get_gaussian_params(found_functions, t=0, threshold_sd=0.2):
    mean = get_future_points(found_functions, np.asarray([t]))
    standard_deviation_up = np.absolute(get_future_points(found_functions, np.asarray([t]), 'up') - mean)
    standard_deviation_down = np.absolute(
        get_future_points(found_functions, np.asarray([t]), 'down') - mean)
    standard_deviation = np.maximum(standard_deviation_up, standard_deviation_down)
    standard_deviation = np.where(standard_deviation > threshold_sd, standard_deviation, threshold_sd)
    weights = np.ones(len(found_functions))
    weights /= np.sum(weights)
    return mean, standard_deviation, weights


def probability_of_being_in_point(found_functions, t=0, step=0.1, align_to_maximum=False):
    mean, standard_deviation, weights = get_gaussian_params(found_functions, t, step)
    p_min, p_max, s_max = np.min(mean), np.max(mean), 4 * np.max(standard_deviation)
    p_min = round((p_min - s_max) / step) * step - step / 2
    try:
        points = np.arange(p_min, p_max + s_max, step)
    except:
        step *= 10
        points = np.arange(p_min - s_max, p_max + s_max, step)
    if points.shape[0] < 3:
        points = np.asarray([p_min - s_max - step, p_min - s_max, p_max + s_max, p_max + s_max + step])
    all_c = np.zeros(points.shape[0])
    for f, func in enumerate(found_functions):
        d = stats.norm(mean[f], standard_deviation[f])
        c = d.cdf(points) * weights[f]
        all_c += c
    probabilities = all_c[1:] - all_c[:-1]
    points_between = points[1:] - (step / 2)
    if align_to_maximum:
        probabilities /= np.max(probabilities)
    # plt.plot(points_between, probabilities, '--')
    # plt.show()
    return probabilities, points_between


def get_points_in_area(points, area):
    area = np.around(area, 5)
    potential_environment_idx = np.ones(points.shape[0], dtype=bool)
    for i in range(3):
        potential_environment_idx = np.logical_and(potential_environment_idx, np.logical_and(points[:, i] >= area[i, 0],
                                                                                             points[:, i] <= area[
                                                                                                 i, 1]))
    return potential_environment_idx


def probable_points_in_area(center_funcs, angles_funcs, points, area, moment, d_x, d_a, probability_th):
    angles_probability_x = get_values_at_moment(angles_funcs[0], moment, d_a)
    angles_probability_y = get_values_at_moment(angles_funcs[1], moment, d_a)
    angles_probability_z = get_values_at_moment(angles_funcs[2], moment, d_a)

    angles = np.array(
        np.meshgrid(angles_probability_x[:, 0], angles_probability_y[:, 0], angles_probability_z[:, 0])).T.reshape(-1,
                                                                                                                   3)
    angles_probabilities = np.prod(np.array(
        np.meshgrid(angles_probability_x[:, 1], angles_probability_y[:, 1], angles_probability_z[:, 1])).T.reshape(-1,
                                                                                                                   3),
                                   axis=1)

    angles, angles_probabilities = angles[angles_probabilities > probability_th], angles_probabilities[
        angles_probabilities > probability_th]

    deviations, deviations_probability = get_deviations(angles, angles_probabilities, points)

    centers_probability_x = get_values_at_moment(center_funcs[0], moment, d_x, True)
    centers_probability_y = get_values_at_moment(center_funcs[1], moment, d_x, True)
    centers_probability_z = get_values_at_moment(center_funcs[2], moment, d_x, True)

    center_positions = np.array(
        np.meshgrid(centers_probability_x[:, 0], centers_probability_y[:, 0], centers_probability_z[:, 0])).T.reshape(
        -1, 3)
    center_probabilities = np.prod(np.array(
        np.meshgrid(centers_probability_x[:, 1], centers_probability_y[:, 1], centers_probability_z[:, 1])).T.reshape(
        -1, 3), axis=1)
    center_positions, center_probabilities = center_positions[center_probabilities > probability_th], \
                                             center_probabilities[center_probabilities > probability_th]

    points, probability = get_points_position(center_positions, center_probabilities, deviations,
                                              deviations_probability, d_x)
    return points, probability


def view1D(a, b):  # a, b are arrays
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    return a.view(void_dt).ravel(), b.view(void_dt).ravel()


def find_matches_in_two_arrays(array_1, array_2):
    A, B = view1D(array_1, array_2)
    c = np.r_[A, B]
    idx = np.argsort(c, kind='mergesort')
    cs = c[idx]
    m0 = cs[:-1] == cs[1:]
    return idx[:-1][m0], idx[1:][m0] - A.shape[0]


def get_values_at_moment(center_func, moment, d_x, use_correction=False, probability_correction=0.8):
    df = pd.DataFrame(columns=list('vp'))
    means, standard_deviations, weights = center_func.get_gaussians_parameters_at_time(moment)
    for m, s, w in zip(means, standard_deviations, weights):
        d = stats.norm(m, s)
        intervals = stats.norm.interval(0.9, loc=m, scale=s)
        values = np.arange(intervals[0] - d_x / 2, intervals[1] + d_x / 2, d_x)
        c = d.cdf(values) * w
        probabilities = c[1:] - c[:-1]
        values_between = np.around((values[1:] - d_x / 2) / d_x) * d_x
        temp_df = pd.DataFrame(np.vstack((values_between, probabilities)).T, columns=list('vp'))
        df = df.append(temp_df, ignore_index=True)
    points_probabilities = df.groupby(['v']).sum().reset_index().to_numpy()
    if use_correction:
        correction = np.asarray([[np.min(points_probabilities[:, 0]) - d_x,
                                  points_probabilities[np.argmin(points_probabilities[:, 0])][
                                      1] * probability_correction],
                                 [np.max(points_probabilities[:, 0]) + d_x,
                                  points_probabilities[np.argmax(points_probabilities[:, 0])][
                                      1] * probability_correction]])
        points_probabilities = np.concatenate((points_probabilities, correction))

    return points_probabilities


def get_deviations(angles, angles_probabilities, points):
    deviations, deviations_probability = np.empty((0, 3)), np.empty(0)

    for angle, probability in zip(angles, angles_probabilities):
        r = R.from_euler('xyz', angle, degrees=True)
        new_points = r.apply(points)
        deviations = np.concatenate((deviations, new_points))
        deviations_probability = np.concatenate(
            (deviations_probability, np.asarray([probability] * new_points.shape[0])))

    return deviations, deviations_probability

    # for a, angles in enumerate(high_probable_angles):
    #     r = R.from_euler('xyz', angles, degrees=True)
    #     new_points = np.round(r.apply(object_points) / round_step) * round_step
    #     temp = pd.DataFrame(
    #         data=np.column_stack((new_points, np.full(new_points.shape[0], high_probable_points_probabilities[a]))),
    #         columns=['x', 'y', 'z', 'p'])
    #     df = df.append(temp, ignore_index=True)
    # xyzp = df.round(2).groupby(['x', 'y', 'z']).max().reset_index().to_numpy()
    # return xyzp[:, :3], xyzp[:, 3]


def get_points_position(centers, centers_p, deviations, deviations_p, d_x):
    points, probabilities = np.zeros((0, 3)), np.zeros(0)
    for center, center_p in zip(centers, centers_p):
        points = np.concatenate((points, deviations + center))
        probabilities = np.concatenate((probabilities, deviations_p * center_p))
    df = pd.DataFrame(np.column_stack((around_to_step(points, d_x), probabilities)), columns=list('xyzp'))
    points_probability = df.groupby(['x', 'y', 'z']).max().reset_index().to_numpy()
    return points_probability[:, :3], points_probability[:, 3]


def find_min_max_of_function(func, t):
    means, standard_deviations, weights = func.get_gaussians_parameters_at_time(t)
    return stats.norm.interval(0.68, loc=means, scale=standard_deviations)


def find_min_max_deviation(min_max_angles, points, d_angle):
    points_len = points.shape[0]

    min_max_angles = np.round(min_max_angles / d_angle) * d_angle
    x_interval = np.arange(min_max_angles[0, 0], min_max_angles[0, 1] + d_angle, d_angle)
    y_interval = np.arange(min_max_angles[1, 0], min_max_angles[1, 1] + d_angle, d_angle)
    z_interval = np.arange(min_max_angles[2, 0], min_max_angles[2, 1] + d_angle, d_angle)
    angles = np.array(np.meshgrid(x_interval, y_interval, z_interval)).T.reshape(-1, 3)

    new_points = np.zeros((angles.shape[0] * points_len, 3))

    for a, angles_set in enumerate(angles):
        r = R.from_euler('xyz', angles_set, degrees=True)
        new_points[a * points_len:(a + 1) * points_len] = r.apply(points)

    return np.asarray([[np.min(new_points[:, 0]), np.max(new_points[:, 0])],
                       [np.min(new_points[:, 1]), np.max(new_points[:, 1])],
                       [np.min(new_points[:, 2]), np.max(new_points[:, 2])]])


def get_angles_from_transformation(transformation_matrix):
    r = R.from_matrix(transformation_matrix)
    return r.as_euler('xyz', degrees=True)


def get_movement_from_transformation(transformation_matrix, source, target):
    point = np.asarray([0, 0, 0, 1])
    point = np.ones([4])
    point = np.dot(transformation_matrix, point.T).T
    rotation_only = np.copy(transformation_matrix)[:3, :3]
    new_source = np.dot(rotation_only, source.T).T
    # print(np.mean(new_source, axis=0), np.mean(source, axis=0), np.mean(target, axis=0))
    # print(np.mean(new_source, axis=0) - np.mean(source, axis=0))
    final_movement = transformation_matrix[:3, 3] + (np.mean(new_source, axis=0) - np.mean(source, axis=0))

    return final_movement


def probabilities_of_movement(center_functions, angle_functions, time_of_probability, d_x, d_angle):
    center_points = []
    object_angles = []
    for i in range(len(center_functions)):
        center_points.append(probability_of_being_in_point(center_functions[i], time_of_probability, d_x, True))
        object_angles.append(probability_of_being_in_point(angle_functions[i], time_of_probability, d_angle, True))
    return center_points, object_angles


def get_movement_from_icp(transformation_matrix, source, target, icp):
    rotation_transformation = np.copy(transformation_matrix)
    rotation_transformation[:3, 3] = 0
    source_points = np.ones([source.shape[0], 4])
    source_points[:, :3] = source
    new_source = np.dot(rotation_transformation, source_points.T).T[:, :3]
    new_transformation = icp(new_source, target)

    return new_transformation


def get_xyz_probabilities_from_angles_probabilities_v0(object_points, x_angles, x_prob, y_angles, y_prob, z_angles,
                                                       z_prob,
                                                       round_step, threshold_p=0.5):
    x_prob, x_angles = x_prob[x_prob > threshold_p], x_angles[x_prob > threshold_p]
    y_prob, y_angles = y_prob[y_prob > threshold_p], y_angles[y_prob > threshold_p]
    z_prob, z_angles = z_prob[z_prob > threshold_p], z_angles[z_prob > threshold_p]

    if x_angles.shape[0] * y_angles.shape[0] * z_angles.shape[0] > 10000:
        print("Слишком много точек")
        return -1
    else:
        angles = np.array(np.meshgrid(x_angles, y_angles, z_angles)).T.reshape(-1, 3)
        probabilities = np.array(np.meshgrid(x_prob, y_prob, z_prob)).T.reshape(-1, 3)
        high_probabilities = np.where(np.prod(probabilities, axis=1) >= threshold_p, True, False)
        high_probable_angles, high_probable_points_probabilities = angles[high_probabilities], \
                                                                   np.prod(probabilities, axis=1)[high_probabilities]
        xyz_dict = {}

        for a, angles in enumerate(high_probable_angles):
            r = R.from_euler('xyz', angles, degrees=True)
            new_points = r.apply(object_points)
            new_points = np.round(new_points / round_step) * round_step
            xyz_dict = sum_probabilities_of_same_points(xyz_dict, new_points, high_probable_points_probabilities[a])
        return xyz_dict


def get_xyz_probabilities_from_angles_probabilities(object_points, x_angles, x_prob, y_angles, y_prob, z_angles,
                                                    z_prob, round_step, threshold_p=0.5):
    x_prob, x_angles = x_prob[x_prob > threshold_p], x_angles[x_prob > threshold_p]
    y_prob, y_angles = y_prob[y_prob > threshold_p], y_angles[y_prob > threshold_p]
    z_prob, z_angles = z_prob[z_prob > threshold_p], z_angles[z_prob > threshold_p]

    if x_angles.shape[0] * y_angles.shape[0] * z_angles.shape[0] > 10000:
        print("Слишком много точек")
        return -1
    else:
        angles = np.array(np.meshgrid(x_angles, y_angles, z_angles)).T.reshape(-1, 3)
        probabilities = np.array(np.meshgrid(x_prob, y_prob, z_prob)).T.reshape(-1, 3)
        high_probabilities = np.where(np.prod(probabilities, axis=1) >= threshold_p, True, False)
        high_probable_angles, high_probable_points_probabilities = angles[high_probabilities], \
                                                                   np.prod(probabilities, axis=1)[high_probabilities]

    df = pd.DataFrame(columns=['x', 'y', 'z', 'p'])

    for a, angles in enumerate(high_probable_angles):
        r = R.from_euler('xyz', angles, degrees=True)
        new_points = np.round(r.apply(object_points) / round_step) * round_step
        temp = pd.DataFrame(
            data=np.column_stack((new_points, np.full(new_points.shape[0], high_probable_points_probabilities[a]))),
            columns=['x', 'y', 'z', 'p'])
        df = df.append(temp, ignore_index=True)
    xyzp = df.round(2).groupby(['x', 'y', 'z']).max().reset_index().to_numpy()
    return xyzp[:, :3], xyzp[:, 3]


def probability_of_all_points_v0(xyz_dict, prob_x, x, prob_y, y, prob_z, z, threshold_p):
    points = np.fromiter(xyz_dict.keys(), dtype=np.dtype('float, float, float'))
    points_prob = np.fromiter(xyz_dict.values(), dtype=float)
    points_prob /= np.max(points_prob)

    dict = {}

    points_prob, points = points_prob[points_prob > threshold_p], points[points_prob > threshold_p]

    prob_x, x = prob_x[prob_x > threshold_p], x[prob_x > threshold_p]
    prob_y, y = prob_y[prob_y > threshold_p], y[prob_y > threshold_p]
    prob_z, z = prob_z[prob_z > threshold_p], z[prob_z > threshold_p]

    # print(np.asarray(list(zip(x, prob_x))))
    # print(np.asarray(list(zip(y, prob_y))))
    # print(np.asarray(list(zip(z, prob_z))))

    center_points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    center_probabilities = np.prod(np.array(np.meshgrid(prob_x, prob_y, prob_z)).T.reshape(-1, 3), axis=1)
    # print(center_points, center_probabilities)

    points = tuple_to_array(points)

    for c, center in enumerate(center_points):
        dict = sum_dif_probabilities_of_same_points(dict, points + center, points_prob * center_probabilities[c])

    points = tuple_to_array(np.fromiter(dict.keys(), dtype=np.dtype('float, float, float')))
    probabilities = np.fromiter(dict.values(), dtype=float)

    return points, probabilities


def probability_of_all_points(points, probabilities, prob_x, x, prob_y, y, prob_z, z, threshold_p):
    probabilities, points = probabilities[probabilities > threshold_p], points[probabilities > threshold_p]

    prob_x, x = prob_x[prob_x > threshold_p], x[prob_x > threshold_p]
    prob_y, y = prob_y[prob_y > threshold_p], y[prob_y > threshold_p]
    prob_z, z = prob_z[prob_z > threshold_p], z[prob_z > threshold_p]

    center_points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    center_probabilities = np.prod(np.array(np.meshgrid(prob_x, prob_y, prob_z)).T.reshape(-1, 3), axis=1)

    df = pd.DataFrame(columns=['x', 'y', 'z', 'p'])

    for c, center in enumerate(center_points):
        temp = pd.DataFrame(
            data=np.column_stack((points + center, probabilities * center_probabilities[c])),
            columns=['x', 'y', 'z', 'p'])
        df = df.append(temp, ignore_index=True)
    xyzp = df.round(2).groupby(['x', 'y', 'z']).max().reset_index().to_numpy()
    return xyzp[:, :3], xyzp[:, 3]


def sum_probabilities_of_same_points(dict, points, probability):
    points = np.round(points, 2)
    for p in points:
        tuple_p = tuple(p)
        if tuple_p in dict:
            dict[tuple_p] = np.max([dict[tuple_p], probability])
        else:
            dict[tuple_p] = probability

    return dict


def sum_dif_probabilities_of_same_points(dict, points, probability):
    points = np.round(points, 2)
    for p_, point in enumerate(points):
        tuple_p = tuple(point)
        if tuple_p in dict:
            dict[tuple_p] = np.max([dict[tuple_p], probability[p_]])
        else:
            dict[tuple_p] = probability[p_]

    return dict


def sum_dif_probabilities_of_one_type(dict, points, probability):
    points = np.round(points, 2)
    for p_, point in enumerate(points):
        if point in dict:
            dict[point] = np.max([dict[point], probability[p_]])
        else:
            dict[point] = probability[p_]

    return dict


def tuple_to_array(tuple_array):
    tuple_to_list = [list(i) for i in tuple_array]
    return np.asarray(tuple_to_list)


def sum_of_the_squares_of_the_residuals(a0, a1):
    return np.sum((a0 - a1) ** 2)


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
        found_rotations[i] = get_angles_from_transformation(current_transformation[:3, :3])
        found_dt_center_shifts[i] = get_movement_from_transformation(transformation, source_points, target_points)
    # print(found_dt_rotations, found_dt_center_shifts)
    found_rotations = np.vstack((np.zeros(3), found_rotations))
    found_center_shifts = np.zeros((len(objects), 3))
    found_center_shifts[1:] = np.cumsum(found_dt_center_shifts, axis=0)

    return found_rotations, found_center_shifts + initial_center


def combine_results(m, sd, w, threshold_w=1e-4):
    means = np.array(np.meshgrid(m[0], m[1], m[2])).T.reshape(-1, 3)
    standard_deviations = np.array(np.meshgrid(sd[0], sd[1], sd[2])).T.reshape(-1, 3)
    weights = np.prod(np.array(np.meshgrid(w[0], w[1], w[2])).T.reshape(-1, 3), axis=1)

    threshold_weights = weights > threshold_w

    return means[threshold_weights], standard_deviations[threshold_weights], weights[threshold_weights]


def get_velocities(functions, moment):
    m_x, sd_x, w_x = functions[0].get_velocity(moment)
    m_y, sd_y, w_y = functions[1].get_velocity(moment)
    m_z, sd_z, w_z = functions[2].get_velocity(moment)

    return combine_results([m_x, m_y, m_z], [sd_x, sd_y, sd_z], [w_x, w_y, w_z])


def get_centers(functions, moment):
    m_x, sd_x, w_x = functions[0].get_gaussians_parameters_at_time(moment)
    m_y, sd_y, w_y = functions[1].get_gaussians_parameters_at_time(moment)
    m_z, sd_z, w_z = functions[2].get_gaussians_parameters_at_time(moment)

    return combine_results([m_x, m_y, m_z], [sd_x, sd_y, sd_z], [w_x, w_y, w_z])


def get_unique_values_3(m, sd, w):
    df = pd.DataFrame(np.c_[m, sd, w], columns=['vx', 'vy', 'vz', 'sdx', 'sdy', 'sdz', 'w'])
    mss = df.round(5).groupby(['vx', 'vy', 'vz']).max().reset_index().to_numpy()
    msw = df.round(5).groupby(['vx', 'vy', 'vz']).sum().reset_index().to_numpy()
    return msw[:, :3], mss[:, 3:6], msw[:, -1]


def get_unique_values_6(m_p, m_v, sd_p, sd_v, w, round_value=5):
    m_p, m_v = np.round(m_p, round_value), np.round(m_v, round_value)
    unique_values = np.unique(np.c_[m_p, m_v], axis=0)
    means_p = np.zeros((unique_values.shape[0], 3))
    means_v = np.zeros((unique_values.shape[0], 3))
    standard_deviation_p = np.zeros((unique_values.shape[0], 3))
    standard_deviation_v = np.zeros((unique_values.shape[0], 3))
    weight = np.zeros(unique_values.shape[0])
    for u, unique_value in enumerate(unique_values):
        means_p[u] = unique_value[:3]
        means_v[u] = unique_value[-3:]
        idx_mask = (np.c_[m_p, m_v] == unique_value).all(axis=1)
        standard_deviation_p[u] = np.max(sd_p[idx_mask], axis=0)
        standard_deviation_v[u] = np.max(sd_v[idx_mask], axis=0)
        weight[u] = np.sum(w[idx_mask])

    return means_p, means_v, standard_deviation_p, standard_deviation_v, np.asarray(weight)

    # df = pd.DataFrame(np.c_[m_p, m_v, sd_p, sd_v, w],
    #                   columns=['x', 'y', 'z', 'vx', 'vy', 'vz', 'sdx', 'sdy', 'sdz', 'sdvx', 'sdvy', 'sdvz', 'w'])
    # mss = df.round(5).groupby(['x', 'y', 'z', 'vx', 'vy', 'vz']).max().reset_index().to_numpy()
    # msw = df.round(5).groupby(['x', 'y', 'z', 'vx', 'vy', 'vz']).sum().reset_index().to_numpy()
    # print(msw[:, :3], msw[:, 3:6], mss[:, 6:9], mss[:, 9:12], msw[:, -1])
    # return msw[:, :3], msw[:, 3:6], mss[:, 6:9], mss[:, 9:12], msw[:, -1]


def find_points_in_radius(points, center, radius):
    radii = np.abs(points - center)
    distances = np.linalg.norm(radii, axis=1)
    return np.asarray(np.where(distances < radius))[0], radii[distances < radius]


def get_particles_velocities(points, linear_velocities_functions, angular_velocities_functions, moment, max_radius):
    c_means, c_standard_deviations, c_weights = get_centers(linear_velocities_functions, moment)
    v_means, v_standard_deviations, v_weights = get_velocities(linear_velocities_functions, moment)
    a_means, a_standard_deviations, a_weights = get_centers(angular_velocities_functions, moment)
    w_means, w_standard_deviations, w_weights = get_velocities(angular_velocities_functions, moment)

    a_means, w_means, a_standard_deviations, w_standard_deviations, a_weights = get_unique_values_6(
        a_means, w_means, a_standard_deviations, w_standard_deviations, a_weights)
    c_means, v_means, c_standard_deviations, v_standard_deviations, c_weights = get_unique_values_6(
        c_means, v_means, c_standard_deviations, v_standard_deviations, c_weights)

    pv = PointsVelocities()

    for c, center in enumerate(c_means):
        points_idx, radii = find_points_in_radius(points, center, max_radius + np.sum(c_standard_deviations[c] ** 2))
        pv.add_points(center, c_standard_deviations[c], a_means, a_standard_deviations, v_means[c], points_idx, radii,
                      w_means, c_weights[c], a_weights, v_standard_deviations[c], w_standard_deviations)
    return pv


def find_angular_to_linear_velocity(angular_velocity, radii, linear_velocity):
    velocity_radii = np.zeros([radii.shape[0], angular_velocity.shape[0], 3])
    if angular_velocity.shape <= radii.shape:
        for a, a_vel in enumerate(angular_velocity):
            velocity_radii[:, a] = np.cross(a_vel, radii) + linear_velocity
    else:
        for r, radius in enumerate(radii):
            velocity_radii[r] = np.cross(angular_velocity, radius) + linear_velocity
    return velocity_radii.flatten().reshape((-1, 3))


def temp():
    def generate_array(dev, cent):
        array_to_return = np.zeros((cent.shape[0], dev.shape[0]))
        for c in range(cent.shape[0]):
            array_to_return[c] = dev + cent[c]
        return array_to_return

    d_x = 0.5
    deviations = np.arange(4.2, 6.6, 0.1)
    centers = np.arange(-1.2, 1.2, 0.1)
    ground_truth = around_to_step(generate_array(deviations, centers), d_x)
    around_to_step_deviation = around_to_step(generate_array(around_to_step(deviations, d_x), centers), d_x)
    around_to_half_step_deviation = around_to_step(generate_array(around_to_step(deviations, d_x / 2), centers), d_x)

    print(np.sum(np.abs(ground_truth - around_to_step_deviation)),
          np.max(np.abs(ground_truth - around_to_step_deviation)))
    print(np.sum(np.abs(ground_truth - around_to_half_step_deviation)),
          np.max(np.abs(ground_truth - around_to_half_step_deviation)))

    around_to_step_center = around_to_step(generate_array(deviations, around_to_step(centers, d_x)), d_x)
    around_to_half_step_center = around_to_step(generate_array(deviations, around_to_step(centers, d_x / 2)), d_x)

    print(np.sum(np.abs(ground_truth - around_to_step_center)),
          np.max(np.abs(ground_truth - around_to_step_center)))
    print(np.sum(np.abs(ground_truth - around_to_half_step_center)),
          np.max(np.abs(ground_truth - around_to_half_step_center)))

    around_to_step_deviation_center = around_to_step(
        generate_array(around_to_step(deviations, d_x), around_to_step(centers, d_x)), d_x)
    around_to_half_step_deviation_center = around_to_step(
        generate_array(around_to_step(deviations, d_x / 2), around_to_step(centers, d_x / 2)), d_x)

    print(np.sum(np.abs(ground_truth - around_to_step_deviation_center)),
          np.max(np.abs(ground_truth - around_to_step_deviation_center)))
    print(np.sum(np.abs(ground_truth - around_to_half_step_deviation_center)),
          np.max(np.abs(ground_truth - around_to_half_step_deviation_center)))

    half_of_half = around_to_half_step_deviation_center = around_to_step(around_to_step(
        generate_array(around_to_step(deviations, d_x / 2), around_to_step(centers, d_x / 2)), d_x / 2), d_x)
    print(np.sum(np.abs(ground_truth - half_of_half)),
          np.max(np.abs(ground_truth - half_of_half)))


def around_to_step(array, dx):
    return np.round(array / dx) * dx


def i_have_a_theory():
    import math
    vx = 2.
    vy = 1.
    xy_p = np.asarray([0., 0., 0])
    xy_c = np.asarray([-1., 1., 0])
    w = -1.
    phi = math.radians(45.)
    r = np.linalg.norm(xy_c - xy_p)
    alpha = 1.
    gamma = 0.5
    mu = 0.5
    q = np.asarray([vx, vy, w])

    print("theory from here")

    print("v: ", vx, " ", vy, " ", 0)

    # variant 1
    T_inv = np.asarray([[math.cos(phi), math.sin(phi), 0],
                        [-math.sin(phi), math.cos(phi), 0],
                        [0, 0, 1]])

    T = np.linalg.inv(T_inv)

    mu_e = - alpha * math.fabs(vx - w * r) / (vy * (1 + gamma) * (1 + alpha))

    xi = 1 if vx > w * r else -1

    if mu <= mu_e:
        P = np.asarray([[1, xi * mu * (1 + gamma), 0],
                        [0, -gamma, 0],
                        [0, -xi * mu * (1 + gamma) / (r * alpha), 1]])
    else:
        P = np.asarray([[1 / (1 + alpha), 0, alpha * r / (1 + alpha)],
                        [0, -gamma, 0],
                        [1 / (r * (1 + alpha)), 0, alpha / (1 + alpha)]])

    q = np.dot(np.dot(np.dot(T, P), T_inv), q)

    print("version 0: ", q)

    # version 2
    v = np.asarray([vx, vy, 0])
    w = np.asarray([0, 0, w])
    distances = xy_p - xy_c
    n = np.dot(np.asarray([0, 1, 0]), T_inv)

    v_projection = v - (np.dot(v, n) / (np.sqrt(sum(n ** 2))) ** 2) * n
    v_projection /= np.linalg.norm(v_projection)

    v_w = np.cross(w, distances)
    q = np.asarray([v, v_w]).flatten()

    T_inv = np.asarray([[v_projection[0], v_projection[1], v_projection[2], 0, 0, 0],
                        [n[0], n[1], n[2], 0, 0, 0],
                        [0, 0, 0, v_projection[0], v_projection[1], v_projection[2]]])
    new_q = np.dot(T_inv, q)

    xi = 1 if vx > w * r else -1

    mu_e = - alpha * math.fabs(new_q[0] - new_q[2]) / (new_q[1] * (1 + gamma) * (1 + alpha))

    if mu <= mu_e:
        P = np.asarray([[1, xi * mu * (1 + gamma), 0],
                        [0, -gamma, 0],
                        [0, -xi * mu * (1 + gamma) / (r * alpha), 1 / r]])
    else:
        P = np.asarray([[1 / (1 + alpha), 0, alpha / (1 + alpha)],
                        [0, -gamma, 0],
                        [1 / (r * (1 + alpha)), 0, alpha / r * (1 + alpha)]])

    q_ref = np.dot(P, new_q)

    T = np.asarray([[v_projection[0], n[0], 0],
                    [v_projection[1], n[1], 0],
                    [v_projection[2], n[2], 0],
                    [0, 0, v_projection[0]],
                    [0, 0, v_projection[1]],
                    [0, 0, v_projection[2]]])

    print("version 1: ", np.dot(T, q_ref))
    print("version 2: ", np.dot(np.dot(np.dot(T, P), T_inv), q))
    q_now = np.dot(T, q_ref)
    xyz = q_now[:3]
    w = np.cross(distances, q_now[-3:]) / np.linalg.norm(distances)
    print(distances, q_now[-3:], np.linalg.norm(distances))
    print(xyz, w)


def find_max_radius(points):
    center = np.mean(points, axis=0)
    distances = points - center
    return np.max(np.sqrt(np.sum(distances ** 2, axis=1)))


def find_new_velocities(points_velocities, normals, point_probabilities):
    number_of_intervals = 10
    gamma_max = 1.0
    mu_max = 1.0
    gamma = np.linspace(0.0, gamma_max, num=number_of_intervals, endpoint=True)
    gamma_probability = np.full(number_of_intervals, gamma_max / number_of_intervals)
    mu = np.linspace(0.0, 1.0, num=number_of_intervals, endpoint=True)
    mu_probability = np.full(number_of_intervals, mu_max / number_of_intervals)

    mu_gamma = np.array(np.meshgrid(mu, gamma)).T.reshape(-1, 2)
    mu_gamma_probability = np.array(np.meshgrid(mu_probability, gamma_probability)).T.reshape(-1, 2)

    # use in case of many points_velocities_values
    # plane_y = normals[points_velocities.points_idx]
    #
    # plane_x = points_velocities.points_linear_velocity - plane_y * \
    #           (np.einsum('ij,ij->i', points_velocities.points_linear_velocity, plane_y) / (
    #                   np.linalg.norm(plane_y, axis=1) ** 2))[:, np.newaxis]
    #
    # plane_x /= np.linalg.norm(plane_x, axis=1)[:, np.newaxis]
    #
    # T_inv = np.zeros((plane_x.shape[0], 3, 6))
    # T_inv[:, 0, :3] = np.copy(plane_x)
    # T_inv[:, 1, :3] = np.copy(plane_y)
    # T_inv[:, 2, -3:] = np.copy(plane_x)
    #
    # T = np.zeros((plane_x.shape[0], 6, 3))
    # T[:, :3, 0] = np.copy(plane_x)
    # T[:, :3, 1] = np.copy(plane_y)
    # T[:, -3:, 2] = np.copy(plane_x)
    #
    # q = np.zeros((plane_x.shape[0], 6))
    # q[:, :3] = points_velocities.points_linear_velocity
    # q[:, -3:] = points_velocities.points_linear_angular_velocity
    #
    # q_new = np.matmul(T_inv, q[:, :, np.newaxis]).reshape(-1, 3)
    #
    # xi = np.where(q_new[:, 0] > q_new[:, 2], np.ones(plane_y.shape[0]), -np.ones(plane_y.shape[0]))
    #
    #
    # import time
    # start = time.time()
    # for m_g, m_g_p in zip(mu_gamma, mu_gamma_probability):
    #     v_xyz, w_xyz = find_all_new_velocities(points_velocities, q, q_new, plane_x, plane_y, T_inv, T, xi, mu=m_g[0],
    #                                            gamma=m_g[1])
    # print(time.time() - start)

    m_g = np.tile(mu_gamma, points_velocities.points_idx.shape[0]).reshape(-1, 2)
    m_g_p = np.tile(mu_gamma_probability, points_velocities.points_idx.shape[0]).reshape(-1, 2)
    points_velocities.repeat_n_times(mu_gamma.shape[0])

    plane_y = normals[points_velocities.points_idx]

    plane_x = points_velocities.points_linear_velocity - plane_y * \
              (np.einsum('ij,ij->i', points_velocities.points_linear_velocity, plane_y) / (
                      np.linalg.norm(plane_y, axis=1) ** 2))[:, np.newaxis]

    plane_x /= np.linalg.norm(plane_x, axis=1)[:, np.newaxis]

    T_inv = np.zeros((plane_x.shape[0], 3, 6))
    T_inv[:, 0, :3] = np.copy(plane_x)
    T_inv[:, 1, :3] = np.copy(plane_y)
    T_inv[:, 2, -3:] = np.copy(plane_x)

    T = np.zeros((plane_x.shape[0], 6, 3))
    T[:, :3, 0] = np.copy(plane_x)
    T[:, :3, 1] = np.copy(plane_y)
    T[:, -3:, 2] = np.copy(plane_x)

    q = np.zeros((plane_x.shape[0], 6))
    q[:, :3] = points_velocities.points_linear_velocity
    q[:, -3:] = points_velocities.points_linear_angular_velocity

    q_new = np.matmul(T_inv, q[:, :, np.newaxis]).reshape(-1, 3)

    xi = np.where(q_new[:, 0] > q_new[:, 2], np.ones(plane_y.shape[0]), -np.ones(plane_y.shape[0]))
    v_xyz, w_xyz, slip_or_not = find_all_new_velocities(points_velocities, q, q_new, plane_x, plane_y, T_inv, T, xi,
                                                        mu=m_g[:, 0],
                                                        gamma=m_g[:, 1])

    q_min = np.zeros((plane_x.shape[0], 6))
    q_min[:, :3] = points_velocities.points_linear_velocity - points_velocities.points_linear_sd
    q_min[:, -3:] = points_velocities.points_linear_angular_velocity - points_velocities.points_linear_angular_sd
    q_new_min = np.matmul(T_inv, q_min[:, :, np.newaxis]).reshape(-1, 3)
    xi = np.where(q_new_min[:, 0] > q_new_min[:, 2], np.ones(plane_y.shape[0]), -np.ones(plane_y.shape[0]))
    v_xyz_min, w_xyz_min, _ = find_all_new_velocities(points_velocities, q_min, q_new_min, plane_x, plane_y, T_inv, T,
                                                      xi,
                                                      mu=m_g[:, 0], gamma=m_g[:, 1])

    q_max = np.zeros((plane_x.shape[0], 6))
    q_max[:, :3] = points_velocities.points_linear_velocity + points_velocities.points_linear_sd
    q_max[:, -3:] = points_velocities.points_linear_angular_velocity + points_velocities.points_linear_angular_sd
    q_new_max = np.matmul(T_inv, q_max[:, :, np.newaxis]).reshape(-1, 3)
    xi = np.where(q_new_max[:, 0] > q_new_max[:, 2], np.ones(plane_y.shape[0]), -np.ones(plane_y.shape[0]))
    v_xyz_max, w_xyz_max, _ = find_all_new_velocities(points_velocities, q_max, q_new_max, plane_x, plane_y, T_inv, T,
                                                      xi,
                                                      mu=m_g[:, 0], gamma=m_g[:, 1])

    v_xyz_sd = np.maximum(np.abs(v_xyz - v_xyz_min), np.abs(v_xyz - v_xyz_max))
    w_xyz_sd = np.maximum(np.abs(w_xyz - w_xyz_min), np.abs(w_xyz - w_xyz_max))
    weights = point_probabilities[points_velocities.points_idx] * points_velocities.points_linear_weight * \
              points_velocities.points_linear_angular_weight * m_g_p[:, 0] * m_g_p[:, 1]
    return v_xyz, w_xyz, v_xyz_sd, w_xyz_sd, weights, points_velocities, slip_or_not
    # df = pd.DataFrame(np.c_[points_velocities.points_idx, point_probabilities[points_velocities.points_idx],
    #                         points_velocities.points_linear_weight, points_velocities.points_linear_angular_weight,
    #                         m_g_p[:, 0], m_g_p[:, 1]],
    #                   columns=['idx', 'point prob', 'velocity weight', 'angular velocity weight', 'mu weight',
    #                            'gamma weight'])
    # pd.set_option('display.max_rows', 500)
    # pd.set_option('display.max_columns', 500)
    # pd.set_option('display.width', 1000)
    # print(df.groupby(['idx']).min().reset_index())


def find_all_new_velocities(points_velocities, q, q_new, plane_x, plane_y, T_inv, T, xi, gamma=0.5, alpha=1., mu=0.5):
    mu_e = - alpha * np.abs(q_new[:, 0] - q_new[:, 2]) / (q_new[:, 1] * (1 + gamma) * (1 + alpha))

    P = np.where((mu <= mu_e)[:, np.newaxis, np.newaxis], slip_p(xi, mu, gamma, alpha, points_velocities.radii_length),
                 no_slip_p(gamma, alpha, points_velocities.radii_length))

    slip_or_not = mu <= mu_e

    q = np.matmul(np.matmul(np.matmul(T, P), T_inv), q[:, :, np.newaxis]).reshape(-1, 6)

    xyz_v = q[:, :3]
    w = np.cross(points_velocities.radii, q[:, -3:]) / np.linalg.norm(points_velocities.radii)

    return xyz_v, w, slip_or_not


def slip_p(xi, mu, gamma, alpha, r):
    P = np.zeros((r.shape[0], 3, 3))
    P[:, 0, 0] = 1
    P[:, 0, 1] = xi * mu * (1 + gamma)
    P[:, 1, 1] = -gamma
    P[:, 2, 1] = -xi * mu * (1 + gamma) / (r * alpha)
    P[:, 2, 2] = 1 / r
    return P


def no_slip_p(gamma, alpha, r):
    P = np.zeros((r.shape[0], 3, 3))
    P[:, 0, 0] = 1 / (1 + alpha)
    P[:, 0, 2] = alpha / (1 + alpha)
    P[:, 1, 1] = -gamma
    P[:, 2, 0] = 1 / (r * (1 + alpha))
    P[:, 2, 2] = alpha / r * (1 + alpha)

    return P


def long_dot(a, b, shape):
    result = np.zeros(shape)
    for i, _ in enumerate(a):
        result[i] = np.dot(a[i], b[i])
    return result


def update_gaussians(v_xyz, w_xyz, v_xyz_sd, w_xyz_sd, weights, points_velocities, linear_functions, angular_functions,
                     moment, slip_or_not):
    unique_centers = np.unique(points_velocities.center, axis=0)
    unique_angles = np.unique(points_velocities.angle, axis=0)

    l_velocity = np.zeros((unique_centers.shape[0] * 2, 3, 3))
    a_velocity = np.zeros((unique_angles.shape[0] * 2, 3, 3))
    l_sd = np.zeros((unique_centers.shape[0] * 2, 3, 3))
    a_sd = np.zeros((unique_angles.shape[0] * 2, 3, 3))
    l_weight = np.zeros(unique_centers.shape[0] * 2)
    a_weight = np.zeros(unique_angles.shape[0] * 2)

    for c, center in enumerate(unique_centers):
        center_idx = (points_velocities.center == center).all(axis=1)

        # slip part
        this_slip = slip_or_not[center_idx]

        w = points_velocities.points_linear_weight[center_idx][this_slip]

        center_sd = points_velocities.center_sd[center_idx][this_slip]

        l_velocity[c * 2], l_sd[c * 2], l_weight[c * 2] = create_mean_std_weight(v_xyz, v_xyz_sd, weights, center,
                                                                                 center_sd, w, center_idx, this_slip)

        # no slip part

        this_not_slip = np.logical_not(this_slip)

        if np.sum(this_not_slip) == 0:
            l_velocity[c * 2 + 1], l_sd[c * 2 + 1], l_weight[c * 2 + 1] = l_velocity[c * 2], l_sd[c * 2], l_weight[
                c * 2]
        else:
            w = points_velocities.points_linear_weight[center_idx][this_not_slip]

            center_sd = points_velocities.center_sd[center_idx][this_not_slip]

            l_velocity[c * 2 + 1], l_sd[c * 2 + 1], l_weight[c * 2 + 1] = \
                create_mean_std_weight(v_xyz, v_xyz_sd, weights, center, center_sd, w, center_idx, this_not_slip)
        if np.sum(this_slip) == 0:
            l_velocity[c * 2], l_sd[c * 2], l_weight[c * 2] = l_velocity[c * 2 + 1], l_sd[c * 2 + 1], l_weight[
                c * 2 + 1]

    for a, angle in enumerate(unique_angles):
        angle_idx = (points_velocities.angle == angle).all(axis=1)

        # slip part
        this_slip = slip_or_not[angle_idx]

        w = points_velocities.points_linear_angular_weight[angle_idx][this_slip]

        angle_sd = points_velocities.angle_sd[angle_idx][this_slip]

        a_velocity[a * 2], a_sd[a * 2], a_weight[a * 2] = create_mean_std_weight(w_xyz, w_xyz_sd, weights, angle,
                                                                                 angle_sd, w, angle_idx, this_slip)
        # no slip part

        this_not_slip = np.logical_not(this_slip)

        if np.sum(this_not_slip) == 0:
            a_velocity[a * 2 + 1], a_sd[a * 2 + 1], a_weight[a * 2 + 1] = a_velocity[a * 2], a_sd[a * 2], a_weight[
                a * 2]
        else:
            w = points_velocities.points_linear_angular_weight[angle_idx][this_not_slip]

            angle_sd = points_velocities.center_sd[angle_idx][this_not_slip]

            a_velocity[a * 2 + 1], a_sd[a * 2 + 1], a_weight[a * 2 + 1] = \
                create_mean_std_weight(w_xyz, w_xyz_sd, weights, angle, angle_sd, w, angle_idx, this_not_slip)
        if np.sum(this_slip) == 0:
            a_velocity[a * 2], a_sd[a * 2], a_weight[a * 2] = a_velocity[a * 2 + 1], a_sd[a * 2 + 1], a_weight[
                a * 2 + 1]

    linear_functions[0].update_gaussians(l_velocity[:, 0], l_sd[:, 0], l_weight, moment)
    linear_functions[1].update_gaussians(l_velocity[:, 1], l_sd[:, 1], l_weight, moment)
    linear_functions[2].update_gaussians(l_velocity[:, 2], l_sd[:, 2], l_weight, moment)
    angular_functions[0].update_gaussians(a_velocity[:, 0], a_sd[:, 0], a_weight, moment)
    angular_functions[1].update_gaussians(a_velocity[:, 1], a_sd[:, 1], a_weight, moment)
    angular_functions[2].update_gaussians(a_velocity[:, 2], a_sd[:, 2], a_weight, moment)


def create_mean_std_weight(means, sd_s, weights, value, value_sd, w, value_idx, slip_idx):
    x_v, y_v, z_v = means[value_idx, 0][slip_idx], means[value_idx, 1][slip_idx], means[value_idx, 2][slip_idx]
    x_v_sd, y_v_sd, z_v_sd = sd_s[value_idx, 0][slip_idx], sd_s[value_idx, 1][slip_idx], sd_s[value_idx, 2][slip_idx]

    weight = weights[value_idx][slip_idx]

    weighed_velocities = np.asarray([x_v * weight, y_v * weight, z_v * weight])

    velocity_means = np.mean(weighed_velocities, axis=1)
    velocity_std = np.std(weighed_velocities, axis=1)

    new_velocity = np.asarray([[value[0], velocity_means[0], 0],
                               [value[1], velocity_means[1], -9.8],
                               [value[2], velocity_means[2], 0]])
    new_sd = np.asarray([[np.max(value_sd[:, 0]), velocity_std[0] + np.max(x_v_sd), 0],
                         [np.max(value_sd[:, 1]), velocity_std[1] + np.max(y_v_sd), 0],
                         [np.max(value_sd[:, 2]), velocity_std[2] + np.max(z_v_sd), 0]])
    new_weight = np.max(w)
    return new_velocity, new_sd, new_weight
