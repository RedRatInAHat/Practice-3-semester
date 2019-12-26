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

    client_id = vrep_functions.vrep_connection()
    kinect_rgb_id = vrep_functions.get_object_id(client_id, 'kinect_rgb')
    kinect_depth_id = vrep_functions.get_object_id(client_id, 'kinect_depth')
    vrep_functions.vrep_change_properties(client_id, kinect_rgb_id, 'vision_sensor_resolution_x', resolution_x)
    vrep_functions.vrep_change_properties(client_id, kinect_rgb_id, 'vision_sensor_resolution_y', resolution_y)
    vrep_functions.vrep_change_properties(client_id, kinect_depth_id, 'vision_sensor_resolution_x', resolution_x)
    vrep_functions.vrep_change_properties(client_id, kinect_depth_id, 'vision_sensor_resolution_y', resolution_y)
    vrep_functions.vrep_start_sim(client_id)
    depth_im, rgb_im = vrep_functions.vrep_get_kinect_images(client_id, kinect_rgb_id, kinect_depth_id)

    frame_difference = FrameDifference(depth_im, rgb_im, 0.3, 0.005)

    for i in range(2):
        depth_im, rgb_im = vrep_functions.vrep_get_kinect_images(client_id, kinect_rgb_id, kinect_depth_id)

        start = time.time()

        frame_difference.current_depth = depth_im
        frame_difference.current_rgb = rgb_im
        mask = frame_difference.subtraction_mask()
        mask = frame_difference.region_growing(mask)

        print(time.time() - start)

        mask = mask * 255

        all_masks = np.zeros_like(depth_im)
        for i in range(len(mask)):
            all_masks += mask[i]

        # cv2.imshow("image", all_masks);
        # cv2.waitKey();

    vrep_functions.vrep_stop_sim(client_id)


def try_ViBE():

    client_id = vrep_functions.vrep_connection()
    kinect_rgb_id = vrep_functions.get_object_id(client_id, 'kinect_rgb')
    kinect_depth_id = vrep_functions.get_object_id(client_id, 'kinect_depth')
    vrep_functions.vrep_change_properties(client_id, kinect_rgb_id, 'vision_sensor_resolution_x', resolution_x)
    vrep_functions.vrep_change_properties(client_id, kinect_rgb_id, 'vision_sensor_resolution_y', resolution_y)
    vrep_functions.vrep_start_sim(client_id)
    _, rgb_im = vrep_functions.vrep_get_kinect_images(client_id, kinect_rgb_id, kinect_depth_id)

    start = time.time()
    vibe = ViBЕ(rgb_im=rgb_im, number_of_samples=10, threshold_r=20 / 255, time_factor=16)
    print(time.time() - start)

    for i in range(3):
        _, rgb_im = vrep_functions.vrep_get_kinect_images(client_id, kinect_rgb_id, kinect_depth_id)
        start = time.time()
        vibe.current_rgb = rgb_im
        vibe.set_mask()
        print(time.time() - start)
        mask = vibe.mask
        mask *= 255
        cv2.imshow("image", mask)
        if cv2.waitKey(1) and 0xff == ord('q'):
            cv2.destroyAllWindows()
    vrep_functions.vrep_stop_sim(client_id)


def try_DEVB():

    client_id = vrep_functions.vrep_connection()
    kinect_rgb_id = vrep_functions.get_object_id(client_id, 'kinect_rgb')
    kinect_depth_id = vrep_functions.get_object_id(client_id, 'kinect_depth')
    vrep_functions.vrep_change_properties(client_id, kinect_rgb_id, 'vision_sensor_resolution_x', resolution_x)
    vrep_functions.vrep_change_properties(client_id, kinect_rgb_id, 'vision_sensor_resolution_y', resolution_y)
    vrep_functions.vrep_change_properties(client_id, kinect_depth_id, 'vision_sensor_resolution_x', resolution_x)
    vrep_functions.vrep_change_properties(client_id, kinect_depth_id, 'vision_sensor_resolution_y', resolution_y)
    vrep_functions.vrep_start_sim(client_id)
    depth_im, rgb_im = vrep_functions.vrep_get_kinect_images(client_id, kinect_rgb_id, kinect_depth_id)

    start = time.time()
    devb = DEVB(rgb_im=rgb_im, depth_im=depth_im, number_of_samples=10, time_factor=16)
    print(time.time() - start)

    for i in range(4):
        depth_im, rgb_im = vrep_functions.vrep_get_kinect_images(client_id, kinect_rgb_id, kinect_depth_id)
        start = time.time()
        devb.set_images(rgb_im, depth_im)
        devb.set_mask()
        print(time.time() - start)
        mask = devb.mask
        mask *= 255
        cv2.imshow("image", mask)
        if cv2.waitKey(1) and 0xff == ord('q'):
            cv2.destroyAllWindows()
    vrep_functions.vrep_stop_sim(client_id)


def try_RGB_MoG():

    frame_number = 0
    rgb_im = image_processing.load_image("falling ball 64x2_48x2", "rgb_" + str(frame_number) + ".png")
    start = time.time()
    mog = RGB_MoG(rgb_im, number_of_gaussians=3)
    print("initialization: ", time.time() - start)
    for i in range(4):
        frame_number += 1
        rgb_im = image_processing.load_image("falling ball 64x2_48x2", "rgb_" + str(frame_number) + ".png")
        start = time.time()
        mask = mog.set_mask(rgb_im)
        print("frame updating: ", time.time() - start)
        cv2.imshow("image", mask)
        cv2.waitKey(0)



if __name__ == "__main__":
    # try_vrep_connection()
    # try_frame_difference()
    # try_ViBE()
    # try_DEVB()
    try_RGB_MoG()
