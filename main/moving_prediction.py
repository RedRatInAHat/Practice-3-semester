import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy.spatial.transform import Rotation as R

import open3d_icp
from set_of_math_functions import *


class MovementFunctions:

    def __init__(self, functions, max_number_of_params, number_of_gaussians=None):
        if number_of_gaussians is None:
            number_of_gaussians = len(functions)
        self.functions_coefficients = np.zeros((number_of_gaussians, max_number_of_params))
        self.covariance_matrices = np.zeros((number_of_gaussians, max_number_of_params, max_number_of_params))
        self.standard_deviations = np.zeros((number_of_gaussians, max_number_of_params))
        self.weights = np.zeros(number_of_gaussians)
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
        return means, standard_deviations, self.weights

    def get_velocity(self, moment):
        coefficients = np.arange(self.functions_coefficients.shape[1] - 1, 0, -1)
        velocity_coefficients = self.functions_coefficients[:, :-1] * coefficients[np.newaxis, :]
        velocity_sd = self.standard_deviations[:, :-1] * coefficients[np.newaxis, :]
        means, standard_deviations = self.get_solution(velocity_coefficients, velocity_sd, moment)
        return means, standard_deviations, self.weights

    def get_solution(self, coefficients, sd, moment):
        means = np.polyval(coefficients.T, moment)
        min_deviation = np.abs(means - np.polyval(coefficients.T - sd.T, moment))
        max_deviation = np.abs(means - np.polyval(coefficients.T + sd.T, moment))
        standard_deviations = np.maximum(min_deviation, max_deviation)
        return means, standard_deviations


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

    points_idx = get_points_in_area(points, area)
    return points[points_idx], probability[points_idx]


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
