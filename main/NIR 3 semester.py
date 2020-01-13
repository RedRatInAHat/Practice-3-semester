import vrep_functions
from points_object import PointsObject
from moving_detection import FrameDifference, ViBЕ, DEVB, RGB_MoG, RGBD_MoG
import visualization
import image_processing
import numpy as np
import time
import cv2

cam_angle = 57.
near_clipping_plane = 0.1
far_clipping_plane = 3.5
number_of_active_points = 2000
resolution_x = 64 * 2
resolution_y = 48 * 2


def try_vrep_connection():
    """Function for checking if vrep_functions and PointsObject are working fine"""
    client_id = vrep_functions.vrep_connection()
    vrep_functions.vrep_start_sim(client_id)
    kinect_rgb_id = vrep_functions.get_object_id(client_id, 'kinect_rgb')
    kinect_depth_id = vrep_functions.get_object_id(client_id, 'kinect_depth')
    depth_im, rgb_im = vrep_functions.vrep_get_kinect_images(client_id, kinect_rgb_id, kinect_depth_id)
    print(depth_im.shape, rgb_im.shape)
    vrep_functions.vrep_stop_sim(client_id)

    depth, rgb = vrep_functions.calculate_point_cloud(rgb_im, depth_im, cam_angle, near_clipping_plane,
                                                      far_clipping_plane)

    background = PointsObject()
    background.set_points(depth, rgb, number_of_active_points)
    # visualization.visualize_object([background])
    xyz = np.asarray(background.return_n_last_points(number_of_active_points))
    print(xyz[0].shape)


def try_frame_difference():
    rgb_im = image_processing.load_image("falling balls and cylinder", "rgb_" + str(0) + ".png")
    depth_im = image_processing.load_image("falling balls and cylinder", "depth_" + str(0) + ".png", "depth")
    start = time.time()
    frame_difference = FrameDifference(depth_im/255, rgb_im/255, 0.3, 0.005)
    print("initialization: ", time.time() - start)

    for i in range(5):
        rgb_im = image_processing.load_image("falling balls and cylinder", "rgb_" + str(i) + ".png")
        depth_im = image_processing.load_image("falling balls and cylinder", "depth_" + str(i) + ".png", "depth")

        start = time.time()

        frame_difference.current_depth = depth_im/255
        frame_difference.current_rgb = rgb_im/255
        mask = frame_difference.subtraction_mask()
        mask = frame_difference.create_mask(mask)

        print(time.time() - start)

        mask = mask * 255

        all_masks = np.zeros_like(depth_im)
        all_masks = all_masks.astype(float)
        for j in range(len(mask)):
            all_masks += mask[j].astype(float)
        image_processing.save_image(all_masks/255, "Results/Frame difference", i, "mask")


def try_ViBE():
    rgb_im = image_processing.load_image("falling balls and cylinder", "rgb_" + str(0) + ".png")
    start = time.time()
    vibe = ViBЕ(rgb_im=rgb_im/255, number_of_samples=10, threshold_r=20 / 255, time_factor=16)
    print(time.time() - start)

    for i in range(5):
        rgb_im = image_processing.load_image("falling balls and cylinder", "rgb_" + str(i) + ".png")

        start = time.time()

        vibe.current_rgb = rgb_im/255
        vibe.set_mask()
        print(time.time() - start)

        mask = vibe.mask
        image_processing.save_image(mask, "Results/ViBE", i, "mask")


def try_DEVB():
    rgb_im = image_processing.load_image("falling balls and cylinder", "rgb_" + str(0) + ".png")
    depth_im = image_processing.load_image("falling balls and cylinder", "depth_" + str(0) + ".png", "depth")
    start = time.time()
    devb = DEVB(rgb_im=rgb_im/255, depth_im=depth_im/255, number_of_samples=10, time_factor=16)
    print(time.time() - start)

    for i in range(5):
        rgb_im = image_processing.load_image("falling balls and cylinder", "rgb_" + str(i) + ".png")
        depth_im = image_processing.load_image("falling balls and cylinder", "depth_" + str(i) + ".png", "depth")

        start = time.time()
        devb.set_images(rgb_im/255, depth_im/255)
        devb.set_mask()
        print(time.time() - start)
        mask = devb.mask

        image_processing.save_image(mask, "Results/DEVB", i, "mask")


def try_RGB_MoG():

    rgb_im = image_processing.load_image("falling balls and cylinder", "rgb_" + str(0) + ".png")
    start = time.time()
    mog = RGB_MoG(rgb_im, number_of_gaussians=3)
    print("initialization: ", time.time() - start)
    for i in range(5):
        rgb_im = image_processing.load_image("falling balls and cylinder", "rgb_" + str(i) + ".png")
        start = time.time()
        mask = mog.set_mask(rgb_im)
        print("frame updating: ", time.time() - start)
        image_processing.save_image(mask/255, "Results/RGB MoG", i, "mask")

def try_RGBD_MoG():
    rgb_im = image_processing.load_image("falling balls and cylinder", "rgb_" + str(0) + ".png")
    depth_im = image_processing.load_image("falling balls and cylinder", "depth_" + str(0) + ".png", "depth")
    start = time.time()
    mog = RGBD_MoG(rgb_im, depth_im, number_of_gaussians=3)
    print("initialization: ", time.time() - start)
    for i in range(5):
        rgb_im = image_processing.load_image("falling balls and cylinder", "rgb_" + str(i) + ".png")
        depth_im = image_processing.load_image("falling balls and cylinder", "depth_" + str(i) + ".png", "depth")
        start = time.time()
        mask = mog.set_mask(rgb_im, depth_im)
        print("frame updating: ", time.time() - start)
        image_processing.save_image(mask/255, "Results/RGBD MoG", i, "mask")


if __name__ == "__main__":
    # try_vrep_connection()
    try_frame_difference()
    # try_ViBE()
    # try_DEVB()
    # try_RGB_MoG()
    # try_RGBD_MoG()
