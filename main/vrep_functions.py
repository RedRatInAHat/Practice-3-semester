import vrep
import cv2
import numpy as np
import array
from PIL import Image as I
import math


def vrep_connection():
    """Connecting to vrep"""
    vrep.simxFinish(-1)
    client_id = vrep.simxStart('127.0.0.1', 19997, True, True, 2000, 5)
    return client_id


def vrep_start_sim(client_id):
    """Starting vrep simulation"""
    vrep.simxStartSimulation(client_id, vrep.simx_opmode_blocking)


def vrep_stop_sim(client_id):
    """Finishing the simulation in vrep"""
    vrep.simxStopSimulation(client_id, vrep.simx_opmode_blocking)


def get_object_id(client_id, object_name):
    """Getting object id from vrep"""
    _, object_id = vrep.simxGetObjectHandle(client_id, object_name, vrep.simx_opmode_oneshot_wait)
    return object_id


def vrep_get_kinect_images(client_id, kinect_rgb_id, kinect_depth_id):
    """Getting rgb and depth images from VREP

    Gets one-dimension array and turns it into two(tree)-dimension numpy array

    Arguments:
        client_id (int): id which allows program connect with VREP API

    Returns:
        rgb (numpy.array): [resolution[0]xresolution[1]x3] rgb array normalized to [0, 1]
        depth (numpy.array): [resolution[0]xresolution[1]] depth array
    """

    _, depth_resolution, depth_image = vrep.simxGetVisionSensorDepthBuffer(client_id, kinect_depth_id,
                                                                           vrep.simx_opmode_oneshot_wait)
    _, rgb_resolution, rgb_image = vrep.simxGetVisionSensorImage(client_id, kinect_rgb_id, 0,
                                                                 vrep.simx_opmode_oneshot_wait)

    image_byte_array = bytes(array.array('b', rgb_image))
    image_buffer = I.frombuffer("RGB", [rgb_resolution[0], rgb_resolution[1]], image_byte_array, "raw", "RGB", 0, 1)
    rgb = np.asarray(image_buffer)
    rgb = cv2.normalize(rgb.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    depth = np.reshape(depth_image, (depth_resolution[1], depth_resolution[0]))

    return depth, rgb


def calculate_point_cloud(rgb, depth, cam_angle, near_clipping_plane, far_clipping_plane, step=1):
    """Calculation of point cloud from images arrays and kinect properties

    Arguments:
        rgb (float array): array contains colors of point cloud
        depth (float array): array contains depth values
        cam_angle (float): angle of camera view
        near_clipping_plane (float): distance to the nearest objects the camera sees
        far_clipping_plane (float): distance to the farthest objects the camera sees
        step (int): step for the cycle; use to reduce the number of returning points

    Returns:
        numpy.array 1: coordinates of points
        numpy.array 2: color of points
    """
    from math import tan, pi, atan, sin, radians

    xyzrgb = []

    depth_amplitude = far_clipping_plane - near_clipping_plane

    x_resolution = depth.shape[1]
    y_resolution = depth.shape[0]

    x_half_angle = radians(cam_angle) / 2.
    y_half_angle = radians(cam_angle) / 2. * y_resolution / x_resolution

    max_dist = 1.
    min_dist = near_clipping_plane * max_dist / far_clipping_plane

    for i in range(0, x_resolution, step):
        x_angle = atan((x_resolution / 2.0 - i - 0.5) / (x_resolution / 2.0) * tan(x_half_angle))
        for j in range(0, y_resolution, step):
            y_angle = atan((j - y_resolution / 2.0 + 0.5) / (y_resolution / 2.0) * tan(y_half_angle))
            point_depth = depth[j, i]
            if max_dist > point_depth > min_dist:
                z = near_clipping_plane + point_depth * depth_amplitude
                x = tan(x_angle) * z
                y = tan(y_angle) * z

                xyzrgb.append([0] * 6)
                xyzrgb[-1] = [x, y, z, rgb[j, i, 0], rgb[j, i, 1], rgb[j, i, 2]]

    xyzrgb = np.asarray(xyzrgb)

    return xyzrgb[:, :3], xyzrgb[:,3:6]


def vrep_change_properties(client_id, object_id, parameter_id, parameter_value):
    """Changing properties of sensors in vrep

    client_id (int): ID of current scene in vrep
    object_id (int): ID of sensor to change
    parameter_id (int): ID of parameter to change
    parameter_value (int/double): value of parameter to change
    """
    params_f = {'near_clipping_plane': 1000,
                'far_clipping_plane': 1001,
                'perspective_angle': 1004
                }

    params_i = {'vision_sensor_resolution_x': 1002,
                'vision_sensor_resolution_y': 1003
                }

    if parameter_id == 'perspective_angle':
        parameter_value = parameter_value/(180*2)*math.pi
    if parameter_id in params_f:
        error = vrep.simxSetObjectFloatParameter(client_id, object_id, params_f[parameter_id], parameter_value,
                                                 vrep.simx_opmode_blocking)
        vrep.simxSetFloatSignal(client_id, 'change_params', parameter_value, vrep.simx_opmode_blocking)
        vrep.simxClearFloatSignal(client_id, 'change_params', vrep.simx_opmode_blocking)
        return error
    elif parameter_id in params_i:
        error = vrep.simxSetObjectIntParameter(client_id, object_id, params_i[parameter_id], parameter_value,
                                               vrep.simx_opmode_blocking)
        vrep.simxSetFloatSignal(client_id, 'change_params', parameter_value, vrep.simx_opmode_blocking)
        vrep.simxClearFloatSignal(client_id, 'change_params', vrep.simx_opmode_blocking)
        return error
    else:
        return 'parameter wasn\'t found'
