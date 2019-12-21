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
