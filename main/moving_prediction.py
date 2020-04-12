import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time

from set_of_math_functions import *


def generate_poly_trajectory(x=None, trajectory_param=None, number_of_steps=10, step=0.01, return_x=False):
    if x is None:
        x = np.arange(0, number_of_steps * step, step)
    y = np.zeros(x.shape[0])
    powers = np.arange(trajectory_param.shape[0])
    for i, x_ in enumerate(x):
        y[i] = np.sum(trajectory_param * np.power(x_, powers))
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
    popt = np.polyfit(trajectory_time, trajectory_points, poly_number)
    return np.flip(popt)


def trajectory_fun_fitting(trajectory_time, trajectory_points, func, params_number):
    popt, pcov = curve_fit(func, trajectory_time, trajectory_points, p0=np.ones(params_number))
    return popt


def find_functions(t, points, threshold_accuracy=1.1e-03):
    found_functions = {}
    for i in range(points.shape[0]):
        poly_params = trajectory_poly_fitting(t, points, i)
        y_ = generate_poly_trajectory(t, poly_params)
        if np.sum(np.isnan(y_)) > 0:
            continue
        if sum_of_the_squares_of_the_residuals(points, y_) < threshold_accuracy:
            found_functions["polyfit_model" + str(i)] = {'function': generate_poly_trajectory,
                                                         'function_params': poly_params}
    functions = get_all_functions()
    for f, func in enumerate(functions):
        try:
            func_params = trajectory_fun_fitting(t, points, func, functions[func])
            y_ = generate_func_trajectory(func, func_params, t)
            print(func_params)
            if np.sum(np.isnan(y_)) > 0:
                continue
            if sum_of_the_squares_of_the_residuals(points, y_) < threshold_accuracy:
                found_functions["curve_fit_model" + str(f)] = {'function': func,
                                                               'function_params': func_params}
        except:
            pass

    if not found_functions:
        print('function wasn\'t found')

    return found_functions


def show_found_functions(found_functions, t, points, tt):
    for func in found_functions:
        f = found_functions[func]['function']
        f_p = found_functions[func]['function_params']
        if "polyfit_model" in func:
            y_ = f(tt, f_p)
        else:
            y_ = generate_func_trajectory(f, f_p, tt)
        plt.plot(t, points, 'o', tt, y_, '--')
        plt.legend(["ground truth", func])
        plt.show()


def sum_of_the_squares_of_the_residuals(a0, a1):
    return np.sum((a0 - a1) ** 2)


if __name__ == "__main__":

    number_of_steps, step = 10, 0.1
    time = np.arange(step, (number_of_steps+1)*step, step)

    # poly block
    # x, y = generate_poly_trajectory(None, np.asarray([1, 3, -9.8]), return_x=True)
    # params = trajectory_poly_fitting(x, y, 4)
    # x_, y_ = generate_poly_trajectory(x, params)

    # func block
    func = simple_neg
    func_arguments = np.asarray([3, 1])
    y = generate_func_trajectory(func, func_arguments,x=time)
    print(y)
    params, _ = curve_fit(func, time, y, p0=np.ones(func_arguments.shape[0]))
    y_ = generate_func_trajectory(func, params, x = time)

    plt.plot(time, y, 'o', time, y_, '--')
    plt.show()
