import numpy as np


def sin_fun(x, *a):
    return a[0] + a[1] * np.sin(a[2] * x + a[3])


def simple_neg(x, *a):
    return a[0] + a[1] / x


def negative_fun_x1(x, *a):
    return a[0] + a[1] / (a[2] * x + a[3])


def negative_fun_x2(x, *a):
    return a[0] + a[1] / (a[2] * x + a[3]) + a[4] / (a[5] * x + a[6])


def exp_fun(x, *a):
    return a[0] + a[1] * a[2] ** (x * a[3] + a[4])


def trig_poly_1x(x, *a):
    return a[0] + a[1] * np.sin(x) + a[2] * np.cos(x)


def trig_poly_2x(x, *a):
    return a[0] + a[1] * np.sin(x) + a[2] * np.cos(x) + a[3] * np.sin(2 * x) + a[4] * np.cos(2 * x)


def get_all_functions():
    return {sin_fun: 4, simple_neg: 2, negative_fun_x1: 4, negative_fun_x2: 6, exp_fun: 5, trig_poly_1x: 3,
            trig_poly_2x: 5}


def functions_help():
    print("sine function:            sin_fun         | a0 + a1 * sin(a2 * x + a3)")
    print("negative function:        simple_neg      | a0 + a1 / x ")
    print("                          negative_fun_x1 | a0 + a1 / (a2 * x + a3)")
    print("                          negative_fun_x2 | a0 + a1 / (a2 * x + a3) + a4 / (a5 * x*x + a6)")
    print("exponential function:     exp_fun         | a0 + a1 * a2**(a3 * x + a4)")
    print("trigonometric polynomial: trig_poly_1x    | a0 + a1 * sin(x) + a2 * cos(x)")
    print("                          trig_poly_2x    | a0 + a1 * sin(x) + a2 * cos(x) + a3 * sin(2x) + a4 * cos(2x)")
