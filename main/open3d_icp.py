import open3d as o3d
import numpy as np
import copy


def get_transformation_matrix_p2p(source_points, target_points, distance_threshold=1,
                                  init_transformation=np.identity(4)):
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    target.points = o3d.utility.Vector3dVector(target_points)

    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    reg_p2p = o3d.registration.registration_icp(source, target, distance_threshold, init_transformation,
                                                o3d.registration.TransformationEstimationPointToPlane())

    source_temp = copy.deepcopy(source)
    source_temp.transform(reg_p2p.transformation)
    # o3d.visualization.draw_geometries([source_temp, target])

    return reg_p2p.transformation


def get_transformation_matrix_cp2p(source_points, target_points, source_color, target_color, distance_threshold=1,
                                   init_transformation=np.identity(4), radius=0.05):
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    source.colors = o3d.utility.Vector3dVector(source_color)
    target.points = o3d.utility.Vector3dVector(target_points)
    target.colors = o3d.utility.Vector3dVector(target_color)

    source_down = source.voxel_down_sample(radius)
    target_down = target.voxel_down_sample(radius)

    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

    result_icp = o3d.registration.registration_colored_icp(source_down, target_down, distance_threshold,
                                                           init_transformation,
                                                           o3d.registration.ICPConvergenceCriteria(
                                                               relative_fitness=1e-1, relative_rmse=1e-1,
                                                               max_iteration=50))
    source_temp = copy.deepcopy(source_down)
    source_temp.transform(result_icp.transformation)
    o3d.visualization.draw_geometries([source_temp, target_down])

    print(result_icp.transformation)
