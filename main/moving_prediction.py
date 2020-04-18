import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
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


def show_found_functions(found_functions, t, points, tt, real_y, zero_shift=0):
    trajectory = get_future_points(found_functions, tt, zero_shift)
    legend = ["given points", 'ground truth']
    plt.plot(t, points + zero_shift, 'o', tt, real_y, '-')
    for func, y_ in zip(found_functions, trajectory):
        plt.plot(tt, y_, '--')
        legend.append(func)
    plt.legend(legend)
    plt.show()


def show_found_functions_with_deviation(found_functions, t, points, tt, real_y, zero_shift=0):
    trajectory = get_future_points(found_functions, tt, zero_shift)
    trajectory_up = get_future_points(found_functions, tt, zero_shift, 'up')
    trajectory_down = get_future_points(found_functions, tt, zero_shift, 'down')
    legend = ["given points", 'ground truth']
    plt.plot(t, points + zero_shift, 'o', tt, real_y, '-')
    for func, y_, y_up, y_down in zip(found_functions, trajectory, trajectory_up, trajectory_down):
        plt.plot(tt, y_, '--')
        legend.append(func)
        plt.fill_between(tt, y_up, y_down, alpha=.25)
    plt.legend(legend)
    plt.show()


def get_future_points(found_functions, tt, zero_shift=0, deviation=''):
    y = np.zeros((len(found_functions), tt.shape[0]))
    for f_, func in enumerate(found_functions):
        f = found_functions[func]['function']
        f_p = found_functions[func]['function_params']
        if deviation == 'up':
            f_p += found_functions[func]['standard_deviation']
        elif deviation == 'down':
            f_p -= found_functions[func]['standard_deviation']
        if "polyfit_model" in func:
            y[f_] = f(tt, f_p) + zero_shift
        else:
            y[f_] = generate_func_trajectory(f, f_p, tt) + zero_shift
    return y


def show_gaussians(found_functions, points, t=0, zero_shift=0):
    mean, standard_deviation, weights = get_gaussian_params(found_functions, t, zero_shift)
    all_c = np.zeros(points.shape[0])
    for f, func in enumerate(found_functions):
        print(mean[f], standard_deviation[f], weights[f])
        d = stats.norm(mean[f], standard_deviation[f])
        c = d.cdf(points)*weights[f]
        all_c += c
        plt.plot(points, c)
        plt.legend([func])
        plt.show()
    plt.plot(points, all_c)
    plt.legend("mixture")
    plt.show()


def get_gaussian_params(found_functions, t=0, zero_shift=0, threshold_sd = 0.01):
    mean = get_future_points(found_functions, np.asarray([t]), zero_shift)
    standard_deviation_up = np.absolute(get_future_points(found_functions, np.asarray([t]), zero_shift, 'up') - mean)
    standard_deviation_down = np.absolute(
        get_future_points(found_functions, np.asarray([t]), zero_shift, 'down') - mean)
    standard_deviation = np.maximum(standard_deviation_up, standard_deviation_down)
    standard_deviation = np.where(standard_deviation > threshold_sd, standard_deviation, threshold_sd)
    weights = np.ones(len(found_functions))
    weights /= np.sum(weights)
    return mean, standard_deviation, weights


# def probability_of_being_between(found_functions, points, t = 0, zero_shift = 0):



def sum_of_the_squares_of_the_residuals(a0, a1):
    return np.sum((a0 - a1) ** 2)
