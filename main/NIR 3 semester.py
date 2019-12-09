import vrep_functions
from points_object import PointsObject
from moving_detection import FrameDifference
from moving_detection import ViBЕ
import visualization
import numpy as np
import time

cam_angle = 57.
near_clipping_plane = 0.1
far_clipping_plane = 3.5
number_of_active_points = 2000

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
    import cv2

    client_id = vrep_functions.vrep_connection()
    vrep_functions.vrep_start_sim(client_id)
    kinect_rgb_id = vrep_functions.get_object_id(client_id, 'kinect_rgb')
    kinect_depth_id = vrep_functions.get_object_id(client_id, 'kinect_depth')
    depth_im, rgb_im = vrep_functions.vrep_get_kinect_images(client_id, kinect_rgb_id, kinect_depth_id)

    frame_difference = FrameDifference(depth_im, rgb_im, 0.3, 0.005)

    for i in range(2):
        depth_im, rgb_im = vrep_functions.vrep_get_kinect_images(client_id, kinect_rgb_id, kinect_depth_id)
        frame_difference.current_depth = depth_im
        frame_difference.current_rgb = rgb_im
        mask = frame_difference.subtraction_mask()
        mask = frame_difference.region_growing(mask)

        mask = mask*255

        all_masks = np.zeros_like(mask[0])
        for i in range(len(mask)):
            all_masks += mask[i]

        cv2.imshow("image", all_masks);
        cv2.waitKey();

    vrep_functions.vrep_stop_sim(client_id)

def try_ViBE():
    temp_array = np.random.rand(10, 10, 3)
    start = time.time()
    vibe_ = ViBЕ(temp_array)
    print(time.time() - start)

if __name__ == "__main__":
    # try_vrep_connection()
    # try_frame_difference()
    try_ViBE()
