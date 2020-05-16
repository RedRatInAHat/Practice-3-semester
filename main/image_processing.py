import numpy as np
from PIL import Image, ImageOps
import os
from math import radians, sin


def calculate_point_cloud(rgb, depth, cam_angle=57., near_clipping_plane=0.2, far_clipping_plane=3.5, step=1):
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

    depth_amplitude = far_clipping_plane - near_clipping_plane
    x_resolution, y_resolution = depth.shape[1], depth.shape[0]
    x_half_resolution, y_half_resolution = x_resolution / 2, y_resolution / 2
    x_half_angle = radians(cam_angle) / 2.
    y_half_angle = radians(cam_angle) / 2. * y_resolution / x_resolution

    xyz = np.zeros((y_resolution, x_resolution, 3))

    x_angles = (x_half_resolution - 0.5 - np.arange(x_resolution)) / x_half_resolution * x_half_angle
    y_angles = (y_half_resolution - 0.5 - np.arange(y_resolution)) / y_half_resolution * y_half_angle
    xx_angles, yy_angles = np.meshgrid(x_angles, y_angles)

    xyz[:, :, 2] = near_clipping_plane + depth_amplitude * depth
    xyz[:, :, 0] = np.tan(xx_angles) * xyz[:, :, 2]
    xyz[:, :, 1] = np.tan(yy_angles) * xyz[:, :, 2]

    xyz = xyz.reshape(y_resolution * x_resolution, 3)
    rgb = rgb.reshape(y_resolution * x_resolution, 3)

    reliable_depth = np.logical_and(xyz[:, 2] > near_clipping_plane, xyz[:, 2] < far_clipping_plane)

    return xyz[reliable_depth], rgb[reliable_depth]
#
#
# def create_3d_data_grid(xyz, d_x, observation_vector=np.asarray([0, 1, 0]), cam_angle=57.,
#                         x_resolution, y_resolution):


def create_dataset_from_vrep(number_of_frames, time_interval=0, resolution_x=640, resolution_y=480, path_to_images=""):
    """Creating dataset

    Dataset consists of images which were taken from vrep scene. Saves them in format "depth_number.png" and
    "rgb_number.png".

    Arguments:
        number_of_frames (int): number of frames to create for dataset
        time_interval (float): time between frames in dataset
        resolution_x (int): horizontal resolution of frames for dataset
        resolution_y (int): vertical resolution of frames for dataset
        path_to_images (string): path for folder where images will be stored
    """
    import vrep_functions
    import os
    import time

    frame_number = 0
    if not os.path.exists(path_to_images):
        os.mkdir(path_to_images)

    client_id = vrep_functions.vrep_connection()
    kinect_rgb_id = vrep_functions.get_object_id(client_id, 'kinect_rgb')
    kinect_depth_id = vrep_functions.get_object_id(client_id, 'kinect_depth')
    vrep_functions.vrep_change_properties(client_id, kinect_rgb_id, 'vision_sensor_resolution_x', resolution_x)
    vrep_functions.vrep_change_properties(client_id, kinect_rgb_id, 'vision_sensor_resolution_y', resolution_y)
    vrep_functions.vrep_change_properties(client_id, kinect_depth_id, 'vision_sensor_resolution_x', resolution_x)
    vrep_functions.vrep_change_properties(client_id, kinect_depth_id, 'vision_sensor_resolution_y', resolution_y)
    vrep_functions.vrep_start_sim(client_id)

    for i in range(number_of_frames):
        start = time.time()
        depth, rgb = vrep_functions.vrep_get_kinect_images(client_id, kinect_rgb_id, kinect_depth_id)
        save_image(depth, path_to_images, frame_number, "depth_")
        save_image(rgb, path_to_images, frame_number, "rgb_")
        frame_number += 1
        while time.time() - start < time_interval:
            pass

    vrep_functions.vrep_stop_sim(client_id)


def save_image(input_image, path_to_image, frame_number=0, image_name="unknown"):
    """Saving image in folder

    Depth image would be saved in greyscale, rgb = rgb. Images from VREP are coming upside-down and mirrored, so there
    is a rotation and mirror.
    """

    if not os.path.exists(path_to_image):
        os.mkdir(path_to_image)

    if input_image.ndim == 2:
        image = Image.fromarray(np.uint8(input_image * 255), 'L')
    elif input_image.ndim == 3:
        image = Image.fromarray(np.uint8(input_image * 255))
    # image = image.rotate(180)
    # image = ImageOps.mirror(image)
    image.save(path_to_image + "/" + image_name + str(frame_number) + ".png")


def load_image(path_to_image, name_of_image, mode="RGB"):
    import cv2

    if mode == "RGB":
        img = cv2.imread(path_to_image + "/" + name_of_image)
        img[:, :, 0], img[:, :, 2] = np.copy(img[:, :, 2]), np.copy(img[:, :, 0])
    else:
        img = cv2.imread(path_to_image + "/" + name_of_image, 0)

    return img


if __name__ == "__main__":
    # create_dataset_from_vrep(5, time_interval=0.5, resolution_x=64 * 2, resolution_y=48 * 2,
    #                          path_to_images="falling balls and cylinder")

    # create_dataset_from_vrep(5, time_interval=0.3, resolution_x=64 * 2, resolution_y=48 * 2,
    #                          path_to_images="falling ball 64x2_48x2")
    #
    rgb_im = load_image("falling ball", "rgb_0.png")
    depth_im = load_image("falling ball", "depth_0.png", "depth")
    calculate_point_cloud(rgb_im / 255, depth_im / 255)
