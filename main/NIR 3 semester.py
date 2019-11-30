import vrep_functions
from points_object import PointsObject
import visualization
import numpy as np

cam_angle = 57.
near_clipping_plane = 0.1
far_clipping_plane = 3.5
number_of_active_points = 2000

client_id = vrep_functions.vrep_connection()
vrep_functions.vrep_start_sim(client_id)
kinect_rgb_id = vrep_functions.get_object_id(client_id, 'kinect_rgb')
kinect_depth_id = vrep_functions.get_object_id(client_id, 'kinect_depth')
depth_im, rgb_im = vrep_functions.vrep_get_kinect_images(client_id, kinect_rgb_id, kinect_depth_id)
vrep_functions.vrep_stop_sim(client_id)

depth, rgb = vrep_functions.calculate_point_cloud(rgb_im, depth_im, cam_angle, near_clipping_plane,
                                                  far_clipping_plane)


background = PointsObject()
background.set_points(depth, rgb, number_of_active_points)
# visualization.visualize_object([background])
xyz = np.asarray(background.return_n_last_points(number_of_active_points))
print(xyz[0].shape)

