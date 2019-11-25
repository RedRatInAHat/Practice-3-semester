import vrep
import cv2
import numpy as np
import array
from PIL import Image as I


def get_images(client_id):
    """Getting rgb and depth images from VREP

    Gets one-dimension array and turns it into two(tree)-dimension numpy array

    Arguments:
        client_id (int): id which allows program connect with VREP API

    Returns:
        rgb (numpy.array): [resolution[0]xresolution[1]x3] rgb array normalized to [0, 1]
        depth (numpy.array): [resolution[0]xresolution[1]] depth array
    """

    _, kinect_rgb = vrep.simxGetObjectHandle(client_id, 'kinect_rgb', vrep.simx_opmode_oneshot_wait)
    _, kinect_depth = vrep.simxGetObjectHandle(client_id, 'kinect_depth', vrep.simx_opmode_oneshot_wait)

    _, rgb_resolution, rgb_image = vrep.simxGetVisionSensorImage(client_id, kinect_rgb, 0,
                                                                 vrep.simx_opmode_oneshot_wait)
    _, depth_resolution, depth_image = vrep.simxGetVisionSensorDepthBuffer(client_id, kinect_depth,
                                                                           vrep.simx_opmode_oneshot_wait)

    image_byte_array = array.array('b', rgb_image)
    image_buffer = I.frombuffer("RGB", (rgb_resolution[0], rgb_resolution[1]), image_byte_array, "raw", "RGB", 0, 1)
    rgb = np.asarray(image_buffer)
    rgb = cv2.normalize(rgb.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    depth = np.reshape(depth_image, (depth_resolution[0], depth_resolution[1]))

    return rgb, depth