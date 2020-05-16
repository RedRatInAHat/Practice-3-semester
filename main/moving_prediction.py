import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy.spatial.transform import Rotation as R

import open3d_icp
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


def get_xyz_probabilities_from_angles_probabilities_v0(object_points, x_angles, x_prob, y_angles, y_prob, z_angles, z_prob,
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
