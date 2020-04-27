import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy.spatial.transform import Rotation as R
import time

from set_of_math_functions import *


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
    return popt, perr


def trajectory_fun_fitting(trajectory_time, trajectory_points, func, params_number):
    popt, pcov = curve_fit(func, trajectory_time, trajectory_points, p0=np.ones(params_number))
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def find_functions(t, points, threshold_accuracy=1e-01):
    found_functions = {}
    max_poly = points.shape[0] - 1 if points.shape[0] - 1 <= 15 else 15
    for i in range(max_poly):
        poly_params, poly_sd = trajectory_poly_fitting(t, points, i)
        y_ = generate_poly_trajectory(t, poly_params)
        if np.sum(np.isnan(y_)) > 0:
            continue
        mean_error = sum_of_the_squares_of_the_residuals(points, y_) / points.shape[0]
        if mean_error < threshold_accuracy:
            found_functions["polyfit_model" + str(i)] = {'function': generate_poly_trajectory,
                                                         'function_params': poly_params, 'error': mean_error,
                                                         'standard_deviation': poly_sd}
    functions = get_all_functions()
    for f, func in enumerate(functions):
        try:
            func_params, func_sd = trajectory_fun_fitting(t, points, func, functions[func])
            y_ = generate_func_trajectory(func, func_params, t)
            if np.sum(np.isnan(y_)) > 0:
                continue
            mean_error = sum_of_the_squares_of_the_residuals(points, y_) / points.shape[0]
            if mean_error < threshold_accuracy and not np.any(np.isinf(func_sd)):
                found_functions["curve_fit_model" + str(f)] = {'function': func,
                                                               'function_params': func_params, 'error': mean_error,
                                                               'standard_deviation': func_sd}
        except:
            pass

    if not found_functions:
        print('function wasn\'t found')

    return found_functions


def show_found_functions(found_functions, t, points, tt, real_y):
    trajectory = get_future_points(found_functions, tt)
    legend = ["given points", 'ground truth']
    plt.plot(t, points, 'o', tt, real_y, '-')
    for func, y_ in zip(found_functions, trajectory):
        plt.plot(tt, y_, '--')
        legend.append(func)
    plt.legend(legend)
    plt.show()


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


def get_movement_from_icp(transformation_matrix, source, target, icp):
    rotation_transformation = np.copy(transformation_matrix)
    rotation_transformation[:3, 3] = 0
    source_points = np.ones([source.shape[0], 4])
    source_points[:, :3] = source
    new_source = np.dot(rotation_transformation, source_points.T).T[:, :3]
    new_transformation = icp(new_source, target)
    return new_transformation


def get_xyz_probabilities_from_angles_probabilities(object_points, x_angles, x_prob, y_angles, y_prob, z_angles, z_prob,
                                                    round_step, threshold_p=0.5):
    x_prob, x_angles = x_prob[x_prob > threshold_p], x_angles[x_prob > threshold_p]
    y_prob, y_angles = y_prob[y_prob > threshold_p], y_angles[y_prob > threshold_p]
    z_prob, z_angles = z_prob[z_prob > threshold_p], z_angles[z_prob > threshold_p]

    if x_angles.shape[0] * y_angles.shape[0] * z_angles.shape[0] > 10000:
        print("Слишком много точек")
        return -1, -1, -1
    else:
        angles = np.array(np.meshgrid(x_angles, y_angles, z_angles)).T.reshape(-1, 3)
        probabilities = np.array(np.meshgrid(x_prob, y_prob, z_prob)).T.reshape(-1, 3)
        high_probabilities = np.where(np.prod(probabilities, axis=1) >= threshold_p, True, False)
        high_probable_angles, high_probable_points_probabilities = angles[high_probabilities], \
                                                                   probabilities[high_probabilities]
        x_dict = {}
        y_dict = {}
        z_dict = {}
        for a, angles in enumerate(high_probable_angles):
            r = R.from_euler('xyz', angles, degrees=True)
            new_points = r.apply(object_points)
            new_points = np.round(new_points / round_step) * round_step
            x_dict = sum_probabilities_of_same_points(x_dict, new_points[:, 0],
                                                      high_probable_points_probabilities[a, 0])
            y_dict = sum_probabilities_of_same_points(y_dict, new_points[:, 1],
                                                      high_probable_points_probabilities[a, 1])
            z_dict = sum_probabilities_of_same_points(z_dict, new_points[:, 2],
                                                      high_probable_points_probabilities[a, 2])
        return x_dict, y_dict, z_dict


def probability_of_all_points(dict_x, dict_y, dict_z, prob_x, x, prob_y, y, prob_z, z, threshold_p):
    points_x = np.fromiter(dict_x.keys(), dtype=float)
    points_x_prob = np.fromiter(dict_x.values(), dtype=float)
    points_x_prob /= np.max(points_x_prob)
    points_y = np.fromiter(dict_y.keys(), dtype=float)
    points_y_prob = np.fromiter(dict_y.values(), dtype=float)
    points_y_prob /= np.max(points_y_prob)
    points_z = np.fromiter(dict_z.keys(), dtype=float)
    points_z_prob = np.fromiter(dict_z.values(), dtype=float)
    points_z_prob /= np.max(points_z_prob)

    # print(np.asarray(list(zip(points_x, points_x_prob))))
    # print(np.asarray(list(zip(points_y, points_y_prob))))
    # print(np.asarray(list(zip(points_z, points_z_prob))))

    dict_x = {}
    dict_y = {}
    dict_z = {}

    points_x_prob, points_x = points_x_prob[points_x_prob > threshold_p], points_x[points_x_prob > threshold_p]
    points_y_prob, points_y = points_y_prob[points_y_prob > threshold_p], points_y[points_y_prob > threshold_p]
    points_z_prob, points_z = points_z_prob[points_z_prob > threshold_p], points_z[points_z_prob > threshold_p]

    prob_x, x = prob_x[prob_x > threshold_p], x[prob_x > threshold_p]
    prob_y, y = prob_y[prob_y > threshold_p], y[prob_y > threshold_p]
    prob_z, z = prob_z[prob_z > threshold_p], z[prob_z > threshold_p]

    print(np.asarray(list(zip(x, prob_x))).shape)
    print(np.asarray(list(zip(y, prob_y))).shape)
    print(np.asarray(list(zip(z, prob_z))).shape)

    print(np.asarray(list(zip(points_x, points_x_prob))).shape)
    print(np.asarray(list(zip(points_y, points_y_prob))).shape)
    print(np.asarray(list(zip(points_z, points_z_prob))).shape)

    for xx, x_point in enumerate(x):
        dict_x = sum_dif_probabilities_of_same_points(dict_x, points_x + x_point, points_x_prob * prob_x[xx])
    for yy, y_point in enumerate(y):
        dict_y = sum_dif_probabilities_of_same_points(dict_y, points_y + y_point, points_y_prob * prob_y[yy])
    for zz, z_point in enumerate(z):
        dict_z = sum_dif_probabilities_of_same_points(dict_z, points_z + z_point, points_z_prob * prob_z[zz])

    points_x = np.fromiter(dict_x.keys(), dtype=float)
    points_x_prob = np.fromiter(dict_x.values(), dtype=float)
    points_x_prob /= np.max(points_x_prob)
    points_y = np.fromiter(dict_y.keys(), dtype=float)
    points_y_prob = np.fromiter(dict_y.values(), dtype=float)
    points_y_prob /= np.max(points_y_prob)
    points_z = np.fromiter(dict_z.keys(), dtype=float)
    points_z_prob = np.fromiter(dict_z.values(), dtype=float)
    points_z_prob /= np.max(points_z_prob)

    return points_x, points_x_prob, points_y, points_y_prob, points_z, points_z_prob

def sum_probabilities_of_same_points(dict, points, probability):
    # model 1
    # for p in points:
    #     try:
    #         dict[p] += probability
    #     except:
    #         dict[p] = probability

    # model 2
    for p in points:
        try:
            dict[p] = np.max(dict[p], probability)
        except:
            dict[p] = probability

    return dict


def sum_dif_probabilities_of_same_points(dict, points, probability):
    # model 1
    # for p_, point in enumerate(points):
    #     try:
    #         dict[point] += probability[p_]
    #     except:
    #         dict[point] = probability[p_]
    # model 2
    for p_, point in enumerate(points):
        try:
            dict[point] = np.max(dict[point], probability[p_])
        except:
            dict[point] = probability[p_]

    return dict


def sum_of_the_squares_of_the_residuals(a0, a1):
    return np.sum((a0 - a1) ** 2)
