import numpy as np
import random
import math


def RANSAC(xyz, xyz_normals, point_to_model_accuracy=0.01, normal_to_normal_accuracy=0.01,
           number_of_points_threshold=500,
           number_of_iterations=10, min_pc_number=300, number_of_subsets=10, use_planes=True, use_box=True,
           use_sphere=True, use_cylinder=True, use_cone=True):
    """RANSAC method for finding parameters of point cloud and it's primitive shape(s)

    While there are points in the point cloud and number of itterations is below threshold algorithm comparing the
    number of inliners of every model and choosing the best one.

    Args:
        xyz (np.ndarray): points of point cloud in xyz format
        xyz_normals (np.ndarray): normals of corresponding points
        point_to_model_accuracy (float): distance between model and point to apply the point as inliner
        normal_to_normal_accuracy (float): angle between model normals and point normal to apply the point as inliner
        number_of_points_threshold (int): number of inliners to apply model as successful
        number_of_iterations (int): number of iterations before the best models will be applied as successful
        min_pc_number (int): number of points recognized too small to search for the model
        number_of_subsets (int): number of subsets for detecting shape model
        use_planes (bool): should RANSAC use plane model
        use_box (bool): should RANSAC use box model
        use_sphere (bool): should RANSAC use sphere model
        use_cylinder (bool): should RANSAC use cylinder model
        use_cone (bool): should RANSAC use cone
    """
    found_shapes = []
    itt = 0
    if min_pc_number < number_of_points_threshold:
        min_pc_number = number_of_points_threshold

    while itt < number_of_iterations and xyz.shape[0] > min_pc_number:
        itt += 1
        fitted_shapes = {}
        # box code
        if use_box:
            try:
                box_normals, box_ro, box_inliners, box_mean = get_best_box_model(xyz, xyz_normals,
                                                                                 point_to_model_accuracy,
                                                                                 normal_to_normal_accuracy,
                                                                                 number_of_subsets)
            except:
                print("box crushed")
                box_inliners = 0
            if np.sum(box_inliners) > number_of_points_threshold:
                box_params = {'parameters': [box_normals, box_ro], 'inliners': box_inliners, 'mean': box_mean,
                              'function': box_points}
                fitted_shapes['box'] = box_params

        # plane_code
        if use_planes:
            try:
                plane_normal, plane_ro, plane_inliners, plane_mean = get_best_plane_model(xyz, xyz_normals,
                                                                                          point_to_model_accuracy,
                                                                                          normal_to_normal_accuracy,
                                                                                          number_of_subsets)
            except:
                print("plane crushed")
                plane_inliners = 0

            if np.sum(plane_inliners) > number_of_points_threshold:
                plane_params = {'parameters': [plane_normal, plane_ro], 'inliners': plane_inliners, 'mean': plane_mean,
                                'function': plane_points_long_one}
                fitted_shapes['plane'] = plane_params

        # sphere code
        if use_sphere:
            try:
                sphere_center, sphere_radius, sphere_inliners, sphere_mean = get_best_sphere_model(xyz,
                                                                                                   point_to_model_accuracy,
                                                                                                   number_of_subsets)
            except:
                print('sphere crushed')
                sphere_inliners = 0
            if np.sum(sphere_inliners) > number_of_points_threshold:
                sphere_params = {'parameters': [sphere_center, sphere_radius], 'inliners': sphere_inliners,
                                 'mean': sphere_mean, 'function': sphere_points}
                fitted_shapes['sphere'] = sphere_params

        # cylinder code
        if use_cylinder:
            try:
                cylinder_axis, cylinder_radius, cylinder_center, cylinder_inliners, cylinder_mean = get_best_cylinder_model(
                    xyz,
                    xyz_normals,
                    point_to_model_accuracy,
                    number_of_subsets)
            except:
                print("cylinder crushed")
                cylinder_inliners = 0
            if np.sum(cylinder_inliners) > number_of_points_threshold:
                cylinder_params = {'parameters': [cylinder_axis, cylinder_radius, cylinder_center],
                                   'inliners': cylinder_inliners, 'mean': cylinder_mean, 'function': cylinder_points}
                fitted_shapes['cylinder'] = cylinder_params

        # cone code
        if use_cone:
            try:
                cone_apex, cone_axis, cone_alfa, cone_inliners, cone_mean = get_best_cone_model(xyz, xyz_normals,
                                                                                                point_to_model_accuracy,
                                                                                                number_of_subsets)
            except:
                print('cone crushed')
                cone_inliners = 0
            if np.sum(cone_inliners) > number_of_points_threshold:
                cone_params = {'parameters': [cone_apex, cone_axis, cone_alfa], 'inliners': cone_inliners,
                               'mean': cone_mean, 'function': cone_points}
                fitted_shapes['cone'] = cone_params

        # choosing the best model
        best_score, best_mean = 0, point_to_model_accuracy * 2
        best_model = None
        for model in fitted_shapes:
            if np.sum(fitted_shapes[model]['inliners']) > best_score or (
                    np.sum(fitted_shapes[model]['inliners']) == best_score and fitted_shapes[model][
                'mean'] < best_mean):
                best_model = model
                best_score, best_mean = np.sum(fitted_shapes[model]['inliners']), fitted_shapes[model]['mean']
        # if there is any model
        if not best_model == None:
            # if plane is good as well as any rounded model it must be chosen, because other model generate rounded
            # models with infinite radius
            if 'plane' in fitted_shapes and np.sum(fitted_shapes[best_model]['inliners']) == np.sum(
                    fitted_shapes['plane']['inliners']):
                best_model = 'plane'
            function = fitted_shapes[best_model]['function']
            params = fitted_shapes[best_model]['parameters']
            inliners = xyz[fitted_shapes[best_model]['inliners']]
            found_shapes.append(function(params, inliners))
            # delete found points from point cloud
            xyz = xyz[np.logical_not(fitted_shapes[best_model]['inliners'])]
            xyz_normals = xyz_normals[np.logical_not(fitted_shapes[best_model]['inliners'])]
            # print(best_model, best_score, best_mean, fitted_shapes[best_model]['parameters'])
    return found_shapes


def get_best_plane_model(xyz, xyz_normals, point_to_model_accuracy, normal_to_normal_accuracy, number_of_subsets):
    """Finding the best parameters of the plane

    Args:
        xyz (nd.asarray): points of the point cloud
        xyz_normals (nd.asarray): normals of the point cloud
        point_to_model_accuracy (float): threshold of maximum distance between points and found model
        normal_to_normal_accuracy (float): threshold of maximum angle between points and found model
        number_of_subsets (int): number of subsets to generate for choosing the best one

    Returns:
        best_normal (nd.asarray): normal of the best plane model
        best_ro (float): D-parameter of the best plane model
        best_inliners (nd.asarray): points, corresponding to best plane model
        best_mean (nd.asarray): mean value of distance to model of inliners
    """
    best_score = 0
    best_mean = point_to_model_accuracy * 2
    # plane fitting
    for _ in range(number_of_subsets):
        normal, ro = plane_fitting_one_point(xyz, xyz_normals)
        # finding plane inliners
        p_inliners, mean = plane_inliners(xyz, xyz_normals, normal, ro, point_to_model_accuracy,
                                          normal_to_normal_accuracy)

        if np.sum(p_inliners) >= best_score and mean < best_mean:
            best_score = np.sum(p_inliners)
            best_mean = mean
            # print(np.sum(abs(np.sum(xyz * normal, axis=1) - ro)[p_inliners])/np.sum(p_inliners))
            best_normal = normal
            best_ro = ro
            best_inliners = p_inliners

    return best_normal, best_ro, best_inliners, best_mean


def plane_fitting_one_point(points, normals):
    """ Finding the parameters of plane

    From equation n1(x-x0) + n2(y-y0) + n3(z-z0) = 0 find four parameters (n1 n2 n3 ro=n1x + n2y + n3z); xyz from point,
    n1 n2 n3 from normal. Need only one point

    Args:
        points (np.ndarray): points of fitting object
        normals (np.ndarray): normals of fitting object

    Returns:
        normal (np.ndarray): normal of found plane
        ro (float): offset parameter of found plane
    """

    # choose one point
    i = random.randint(0, points.shape[0] - 1)
    # finding parameters
    normal = normals[i] / np.linalg.norm(normals[i])
    ro = np.sum(points[i] * normals[i])
    return normal, ro


def plane_fitting_three_points(points):
    """Finding plane model with three points

    Args:
        points (nd.asarray):
    Returns:
        cp (nd.asarray): normal to the plane model
        d (float): D-parameter of the plane model
    """
    p = points[np.random.randint(points.shape[0], size=3), :]
    p0, p1, p2 = p

    v1 = p2 - p0
    v2 = p1 - p0

    cp = np.cross(v1, v2)
    cp = cp / np.linalg.norm(cp)
    d = np.dot(cp, p2)

    return cp, d


def plane_inliners(points, normals, plane_normal, plane_ro, d_accuracy, a_accuracy):
    """ Finding inliners for created model

    Inserting point into equation and looking for the difference between recieved ro and model ro.
    Finding angle between normals.
    If in for both cases difference and angle are below the threshold, it sets as inliner.

    Args:
        points (np.ndarray): points of fitting object
        normals (np.ndarray): normals of fitting object
        plane_normal (np.ndarray): normal of plane model
        plane_ro (float): offset parameter of plane model
        d_accuracy (float): threshold difference between model and point
        a_accuracy (float): threshold angle between normals of model and point

    Returns:
        arg1 (np.ndarray): map, showing whether point belongs to model
    """
    # finding angle between point normal and plane model normal
    angles = angle_between_normals(normals, plane_normal)
    # threshold check
    angle_truth = np.logical_or(np.where(abs(angles) < a_accuracy, True, False),
                                np.where(abs(angles) - math.pi < a_accuracy, True, False))
    distances = abs(np.sum(points * plane_normal, axis=1) - plane_ro)
    distance_truth = np.where(distances < d_accuracy, True, False)
    inliners = np.logical_and(angle_truth, distance_truth)
    return inliners, np.mean(distances[inliners])


def plane_points_long_one(parameters, points, step=0.05):
    """ Generating the points of model

    Generating points with respect of it's borders.

    Args:
        parameters (nd.asarray): normal and ro for plane model
        points (nd.asarray): inliners of plane model
        step (float): step of points generation
    Returns:
        xyz (nd.asarray): generated points
    """
    normal, ro = parameters
    around_points = np.around(points, decimals=get_count(step))
    xyz = np.ones((0, 3))

    # YZ plane case
    if np.unique(around_points[:, 0]).shape[0] == 1:
        for i, y_value in enumerate(np.unique(around_points[:, 1])):
            min_z, max_z = np.min(points[around_points[:, 1] == y_value, 2]), np.max(
                points[around_points[:, 1] == y_value, 2])
            z = np.arange(min_z, max_z + step, step)
            xy = np.ones([z.shape[0], 3])
            xy[:, 1] *= y_value
            xy[:, 2] *= z
            xyz = np.vstack((xyz, xy))
        xyz[:, 0] *= points[0, 0]
    # XZ plane case
    elif np.unique(around_points[:, 1]).shape[0] == 1:
        for i, x_value in enumerate(np.unique(around_points[:, 0])):
            min_z, max_z = np.min(points[around_points[:, 0] == x_value, 2]), np.max(
                points[around_points[:, 0] == x_value, 2])
            z = np.arange(min_z, max_z + step, step)
            xy = np.ones([z.shape[0], 3])
            xy[:, 0] *= x_value
            xy[:, 2] *= z
            xyz = np.vstack((xyz, xy))
        xyz[:, 1] *= points[0, 1]
    else:
        # XY plane case
        if np.abs(normal[2]) < 1e-5:
            for _, z_value in enumerate(np.unique(around_points[:, 2])):
                min_y, max_y = np.min(points[around_points[:, 2] == z_value, 1]), np.max(
                    points[around_points[:, 2] == z_value, 1])
                y = np.arange(min_y, max_y + step, step)
                xz = np.ones([y.shape[0], 3])
                xz[:, 2] *= z_value
                xz[:, 1] *= y
                xyz = np.vstack((xyz, xz))
            xyz[:, 0] = (ro - normal[2] * xyz[:, 2] - normal[1] * xyz[:, 1]) / normal[0]
        # XYZ plane case
        else:
            for _, x_value in enumerate(np.unique(around_points[:, 0])):
                min_y, max_y = np.min(points[around_points[:, 0] == x_value, 1]), np.max(
                    points[around_points[:, 0] == x_value, 1])
                y = np.arange(min_y, max_y + step, step)
                xy = np.ones([y.shape[0], 3])
                xy[:, 0] *= x_value
                xy[:, 1] *= y
                xyz = np.vstack((xyz, xy))
            xyz[:, 2] = (ro - normal[0] * xyz[:, 0] - normal[1] * xyz[:, 1]) / normal[2]
    return xyz


def plane_points_free_shape(normal, ro, points, step=0.01):
    """Generating the points of plane model

    The point is applied if it has inliner point in step radius.

    Args:
        normal (np.ndarray): normal of the plane model
        ro (float): D-parameter of the plane model
        points (np.ndarray): inliners of the plane model
        step (float): step of points generation

    Returns:
        _ (nd.asarray): generated points
    """
    x = np.arange(np.min(points[:, 0]), np.max(points[:, 0]) + step, step)
    y = np.arange(np.min(points[:, 1]), np.max(points[:, 1]) + step, step)

    x = np.tile(x, (y.shape[0], 1))
    y = np.tile(np.array([y]).transpose(), (1, x.shape[1]))

    points_around = np.around(points[:, :2], decimals=get_count(step))
    found_points = np.c_[x.flatten(), y.flatten()]
    found_points_around = np.around(found_points, decimals=get_count(step))
    print(found_points_around.shape)

    # code from here https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    # i'm not cool enough to understand why does it work. @TODO understand
    nrows, ncols = points_around.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [points_around.dtype]}

    C = np.intersect1d(points_around.view(dtype), found_points_around.view(dtype))
    xy_points = C.view(points_around.dtype).reshape(-1, ncols)

    z = (ro - normal[0] * xy_points[:, 0] - normal[1] * xy_points[:, 1]) / normal[2]
    return np.c_[xy_points, z]


def plane_points(normal, ro, x_min, x_max, y_min, y_max, z_min, z_max, step=0.01):
    """Creating the plane between minimum and maximum points"""
    x = np.arange(x_min, x_max + step, step)
    y = np.arange(y_min, y_max + step, step)

    x = np.tile(x, (y.shape[0], 1))
    y = np.tile(np.array([y]).transpose(), (1, x.shape[1]))

    z = (ro - normal[0] * x - normal[1] * y) / normal[2]
    z_condition = np.where((z.flatten() >= z_min) & (z.flatten() <= z_max), True, False)

    return np.c_[np.c_[x.flatten()[z_condition], y.flatten()[z_condition]], z.flatten()[z_condition]]


def get_best_box_model(xyz, xyz_normals, point_to_model_accuracy, normal_to_normal_accuracy,
                       number_of_subsets, full_model=False):
    """Finding the best parameters of the box

    Args:
        xyz (nd.asarray): points of the point cloud
        xyz_normals (nd.asarray): normals of the point cloud
        point_to_model_accuracy (float): threshold of maximum distance between points and found model
        normal_to_normal_accuracy (float): threshold of maximum angle between points and found model
        number_of_subsets (int): number of subsets to generate for choosing the best one
        full_model (bool): shows if there are six planes to extract or just three

    Returns:
        _ (nd.asarray): normals of the best box model
        _ (nd.asarray): D-parameters of the best box model
        _ (nd.asarray): points, corresponding to best box model
        _ (nd.asarray): mean value of distance to model of inliners
    """
    best_score = 0
    best_mean = point_to_model_accuracy * 2
    normal_0_best, normal_1_best = 0, 0
    # box fitting
    for _ in range(number_of_subsets):
        # two plane models obtaining
        normal_0, ro_0 = plane_fitting_one_point(xyz, xyz_normals)
        normal_1, ro_1 = plane_fitting_one_point(xyz, xyz_normals)
        # angle between plane's normals
        angle = np.abs(angle_between_normals(normal_0, normal_1))
        angle = angle if angle < math.pi / 2 else math.pi - angle

        # if angle between normals is around 90 degrees
        if math.pi / 2 + 0.1 > np.abs(angle_between_normals(normal_0, normal_1)) > math.pi / 2 - 0.1:
            # get planes inliners
            p0_inliners, mean_0 = plane_inliners(xyz, xyz_normals, normal_0, ro_0, point_to_model_accuracy,
                                                 normal_to_normal_accuracy)
            p1_inliners, mean_1 = plane_inliners(xyz, xyz_normals, normal_1, ro_1, point_to_model_accuracy,
                                                 normal_to_normal_accuracy)
            # get the best two planes model
            if np.sum(p0_inliners) + np.sum(p1_inliners) >= best_score and (mean_0 + mean_1) / 2 < best_mean:
                best_score = np.sum(p0_inliners) + np.sum(p1_inliners)
                best_mean = (mean_0 + mean_1) / 2
                normal_0_best = normal_0
                normal_1_best = normal_1
    # get the third plane of box
    normal_2 = np.cross(normal_0_best, normal_1_best)
    normal_2_best = normal_2 / np.linalg.norm(normal_2)
    # get planes's D-parameters
    ro_0 = get_most_frequent_ro(normal_0_best, xyz)
    ro_1 = get_most_frequent_ro(normal_1_best, xyz)
    ro_2 = get_most_frequent_ro(normal_2_best, xyz)

    # get inliners and distances of box model
    inliners_0, inliners_1, inliners_2, distances_0 = box_inliners(xyz, normal_0_best, ro_0, normal_1_best, ro_1,
                                                                   normal_2_best, ro_2, point_to_model_accuracy)
    box_inliners_0 = np.logical_or(np.logical_or(inliners_0, inliners_1), inliners_2)
    box_mean_0 = np.mean(distances_0[box_inliners_0])
    # if points cloud contains all 6 planes obtain other 3 planes the same way
    if full_model:
        xyz_ = xyz[np.bitwise_not(box_inliners_0)]
        ro_0_ = get_most_frequent_ro(normal_0_best, xyz_)
        ro_1_ = get_most_frequent_ro(normal_1_best, xyz_)
        ro_2_ = get_most_frequent_ro(normal_2_best, xyz_)
        inliners_0, inliners_1, inliners_2, distances_1 = box_inliners(xyz, normal_0_best, ro_0_, normal_1_best, ro_1_,
                                                                       normal_2_best, ro_2_, point_to_model_accuracy)
        box_inliners_1 = np.logical_or(np.logical_or(inliners_0, inliners_1), inliners_2)
        box_mean_1 = np.mean(distances_1[box_inliners_1])
        return np.asarray([normal_0_best, normal_1_best, normal_2_best]), np.asarray([
            [ro_0, ro_0_], [ro_1, ro_1_], [ro_2, ro_2_]]), np.logical_or(box_inliners_0, box_inliners_1), (
                       box_mean_0 + box_mean_1) / 2
    else:
        return np.asarray([normal_0_best, normal_1_best, normal_2_best]), np.asarray([
            [ro_0], [ro_1], [ro_2]]), box_inliners_0, box_mean_0


def get_random_projection(points, normal, ro):
    """Getting projection of random point on the plane"""
    point = points[random.randint(0, points.shape[0] - 1)]
    t = ro - np.sum(normal * point)
    projection = normal * t + point
    return point, projection


def get_most_frequent_ro(normal, points, around=3):
    """Getting the most frequent D-parameter of the plane with the normal

    Args:
        normal (nd.asarray): normal of the plane
        points (nd.asarray): inliners of the plane
    Returns:
        _ (float): D-parameter of the plane
    """
    all_ro = np.sum(normal * points, axis=1)
    all_ro = np.around(all_ro, around)
    (values, counts) = np.unique(all_ro, return_counts=True)
    return values[np.argmax(counts)]


def box_inliners(points, normal_0, ro_0, normal_1, ro_1, normal_2, ro_2, accuracy):
    """Obtaining inliners of every plane

    Args:
        points (np.ndarray): points of the point cloud
        normal_0 (np.ndarray): normal of the plane 0
        ro_0 (np.ndarray): D-parameter of the plane 0
        normal_1 (np.ndarray): normal of the plane 1
        ro_1 (np.ndarray): D-parameter of the plane 1
        normal_2 (np.ndarray): normal of the plane 2
        ro_2 (np.ndarray): D-parameter of the plane 2
        accuracy (float): threshold of max distance between model and points
    Returns:
        inliners_0 (np.ndarray): inliners of the plane_0
        inliners_1 (np.ndarray): inliners of the plane_1
        inliners_2 (np.ndarray): inliners of the plane_2
        _ (np.ndarray): min distances between points and plane model
    """
    distances = np.empty([points.shape[0], 3])
    distances[:, 0] = abs(np.sum(points * normal_0, axis=1) - ro_0)
    distances[:, 1] = abs(np.sum(points * normal_1, axis=1) - ro_1)
    distances[:, 2] = abs(np.sum(points * normal_2, axis=1) - ro_2)
    inliners_0 = np.where(distances[:, 0] < accuracy, True, False)
    inliners_1 = np.where(distances[:, 1] < accuracy, True, False)
    inliners_2 = np.where(distances[:, 2] < accuracy, True, False)

    distances[:, 0] = np.where(inliners_0, distances[:, 0], accuracy * 10)
    distances[:, 1] = np.where(inliners_0, distances[:, 1], accuracy * 10)
    distances[:, 2] = np.where(inliners_0, distances[:, 2], accuracy * 10)

    return inliners_0, inliners_1, inliners_2, np.min(distances, axis=1)


def box_points(parameters, inliners, step=0.02):
    """Generating points for box model

    Generating points for every of 6 planes of box

    Args:
        parameters (np.ndarray): normals and ro of the three planes of box model
        inliners (np.ndarray): inliners of the model
        step (float): step of points generation
    Returns:
        inliners (np.ndarray): points of the generated model
    """
    normals, ro = parameters

    # shift model and rotate it along to axises
    shift = np.mean(inliners, axis=0)
    inliners -= shift
    inliners, new_normals = go_to_standard_axises(normals, inliners)

    # generating new points
    inliners = generate_box_points(inliners, step)

    # rotate and shift model to original position
    inliners, _ = go_to_standard_axises(np.flip(new_normals, 0), inliners, np.flip(normals, 0))
    inliners += shift

    return inliners


def go_to_standard_axises(normals, inliners, axises=np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
    """Rotating of the shape along new axises

    Args:
        normals (np.ndarray): normals of the box
        inliners (np.ndarray): points of the model
        axises (np.ndarray): new axises to rotate to
    Return:
        inliners (np.ndarray): rotated inliners points
        normals (np.ndarray): normals in the new coordinate system
    """
    for i, axis in enumerate(axises):
        inliners = rotate(inliners, np.cross(axis, normals[i]), angle_between_normals(normals[i], axis))
        normals = rotate(normals, np.cross(axis, normals[i]), angle_between_normals(normals[i], axis))
    return inliners, normals


def generate_box_points(inliners, step=0.01):
    """Generating points of the box model

    For every of 6 planes of the box generate points.

    Args:
        inliners (np.ndarray): points of the oriented box
        step (float): step of generating points
    Return:
        box_points (np.ndarray): generated points of the oriented box
    """
    box_points = np.empty([0, 3])

    # Find borders of the planes
    x_min, x_max = np.min(inliners[:, 0]), np.max(inliners[:, 0])
    y_min, y_max = np.min(inliners[:, 1]), np.max(inliners[:, 1])
    z_min, z_max = np.min(inliners[:, 2]), np.max(inliners[:, 2])

    x = np.arange(x_min, x_max, step)
    y = np.arange(y_min, y_max, step)
    z = np.arange(z_min, z_max, step)

    # Generate YZ planes
    yz = np.empty([y.shape[0] * z.shape[0], 3])
    yz[:, 0] = x_min
    yz[:, 1] = np.repeat(y, z.shape[0])
    yz[:, 2] = np.tile(z, y.shape[0])

    box_points = np.concatenate((box_points, yz))

    yz[:, 0] = x_max
    box_points = np.concatenate((box_points, yz))

    # Generate XZ planes
    xz = np.empty([x.shape[0] * z.shape[0], 3])
    xz[:, 0] = np.repeat(x, z.shape[0])
    xz[:, 1] = y_min
    xz[:, 2] = np.tile(z, x.shape[0])
    box_points = np.concatenate((box_points, xz))

    xz[:, 1] = y_max
    box_points = np.concatenate((box_points, xz))

    # Generate XY plane
    xy = np.empty([x.shape[0] * y.shape[0], 3])
    xy[:, 0] = np.repeat(x, y.shape[0])
    xy[:, 1] = np.tile(y, x.shape[0])
    xy[:, 2] = z_min
    box_points = np.concatenate((box_points, xy))

    xy[:, 2] = z_max
    box_points = np.concatenate((box_points, xy))

    return box_points
    # np.repeat(y, z.shape[0]))


def get_best_sphere_model(points, point_to_model_accuracy, number_of_subsets):
    """Getting the best model of sphere

    Args:
        points (np.ndarray): points of the point cloud
        point_to_model_accuracy (float): max distance threshold between model and the points
        number_of_subsets (int): number of subsets to chose the best

    Returns:
        best_center (np.ndarray): center of the sphere model
        best_radius (float): radius of the sphere model
        best_inliners (np.ndarray): inliners of the sphere model
        best_mean (np.ndarray): mean error of inliners
    """
    best_score = best_center = best_radius = best_inliners = 0
    best_mean = point_to_model_accuracy * 2
    for _ in range(number_of_subsets):
        center, radius = sphere_fitting(points)
        inliners, mean = sphere_inliners(points, center, radius, point_to_model_accuracy)
        if np.sum(inliners) >= best_score and mean < best_mean:
            best_score = np.sum(inliners)
            best_mean = mean
            best_center = center
            best_radius = radius
            best_inliners = inliners
    return best_center, best_radius, best_inliners, best_mean


def sphere_fitting(xyz):
    """Getting parameters of the sphere

    Getting the parameters of the sphere by four points: https://mathworld.wolfram.com/Circumsphere.html

    Args:
        xyz (np.ndarray): points of the point cloud
    Returns:
        center (np.ndarray): center of the sphere model
        radius (float): radius of the sphere model
    """
    p = xyz[np.random.choice(xyz.shape[0], 4)]

    a = np.linalg.det([[p[0, 0], p[0, 1], p[0, 2], 1],
                       [p[1, 0], p[1, 1], p[1, 2], 1],
                       [p[2, 0], p[2, 1], p[2, 2], 1],
                       [p[3, 0], p[3, 1], p[3, 2], 1]])
    p_q = np.sum(p ** 2, axis=1)

    d_x = np.linalg.det([[p_q[0], p[0, 1], p[0, 2], 1],
                         [p_q[1], p[1, 1], p[1, 2], 1],
                         [p_q[2], p[2, 1], p[2, 2], 1],
                         [p_q[3], p[3, 1], p[3, 2], 1]])

    d_y = -np.linalg.det([[p_q[0], p[0, 0], p[0, 2], 1],
                          [p_q[1], p[1, 0], p[1, 2], 1],
                          [p_q[2], p[2, 0], p[2, 2], 1],
                          [p_q[3], p[3, 0], p[3, 2], 1]])

    d_z = np.linalg.det([[p_q[0], p[0, 0], p[0, 1], 1],
                         [p_q[1], p[1, 0], p[1, 1], 1],
                         [p_q[2], p[2, 0], p[2, 1], 1],
                         [p_q[3], p[3, 0], p[3, 1], 1]])

    c = np.linalg.det([[p_q[0], p[0, 0], p[0, 1], p[0, 2]],
                       [p_q[1], p[1, 0], p[1, 1], p[1, 2]],
                       [p_q[2], p[2, 0], p[2, 1], p[2, 2]],
                       [p_q[3], p[3, 0], p[3, 1], p[3, 2]]])

    center = np.asarray([d_x / (2 * a), d_y / (2 * a), d_z / (2 * a)])
    radius = math.sqrt(d_x ** 2 + d_y ** 2 + d_z ** 2 - 4 * a * c) / (2 * math.fabs(a))
    return center, radius


def sphere_inliners(points, center, radius, point_to_model_accuracy):
    """Getting sphere inliners

    Evaluate the distance between center and point and find the difference between the distance and the radius

    Args:
        points (np.ndarray): points of the point cloud
        center (np.ndarray): center of the sphere model
        radius (np.ndarray): radius of the sphere model
        point_to_model_accuracy (np.ndarray): max distance threshold between the model and the points
    Returns:
        accuracity (np.ndarray): inliners in fact
        mean (float): mean error of the inliners
    """
    dif = np.abs(np.linalg.norm(center - points, axis=1) - radius)
    accuracy = dif < point_to_model_accuracy
    mean = np.mean(dif[accuracy])
    return accuracy, mean


def sphere_points(parameters, inliners, step=math.radians(3)):
    """Generating sphere points

    Generates the points of the sphere

    Args:
        parameters (np.ndarray): center and radius of the sphere model
        inliners (np.ndarray): inliners of the sphere model; need it only for uniformity
        step (float): step of points generation
    Returns:
        _ (np.ndarray): points of the generated model
    """
    center, radius = parameters
    theta = np.arange(0, 2 * math.pi + step, step)
    phi = np.arange(0, math.pi + step, step)
    points = np.zeros((theta.shape[0] * phi.shape[0], 3))
    angles = np.zeros((theta.shape[0] * phi.shape[0], 2))
    angles[:, 0] = np.repeat(theta, phi.shape[0])
    angles[:, 1] = np.tile(phi, theta.shape[0])
    points[:, 0] = radius * np.cos(angles[:, 0]) * np.sin(angles[:, 1])
    points[:, 1] = radius * np.sin(angles[:, 0]) * np.sin(angles[:, 1])
    points[:, 2] = radius * np.cos(angles[:, 1])
    return points + center


def get_best_cylinder_model(points, normals, point_to_model_accuracy, number_of_subsets):
    """Getting the best cylinder model

    Args:
        points (np.ndarray): points of the point cloud
        normals (np.ndarray): normals of the point cloud
        point_to_model_accuracy (float): max distance threshold between model and the points
        number_of_subsets (int): number of subsets to chose the best
    Returns:
        best_axis (np.ndarray): vector of the cylinder model axis
        best_radius (float): radius of the cylinder model
        best_center_points (np.ndarray): point belonging to axis of the cylinder model
        best_inliners (np.ndarray): inliners of the cylinder model
        best_mean (float): best mean error of inliners
    """
    best_score = 0
    best_radius = best_center_point = best_axis = 0
    best_mean = point_to_model_accuracy * 2
    best_inliners = 0
    for _ in range(number_of_subsets):
        axis, radius, center = cylinder_fitting(points, normals)
        inliners, mean = cylinder_inliners(points, axis, radius, center, point_to_model_accuracy)
        if np.sum(inliners) >= best_score and mean < best_mean:
            best_score, best_mean = np.sum(inliners), mean
            best_axis, best_radius, best_center_point = axis, radius, center
            best_inliners = inliners
    return best_axis, best_radius, best_center_point, best_inliners, best_mean


def cylinder_fitting(xyz, xyz_normals):
    """ Getting parameters of the cylinder model

    Args:
        xyz (np.ndarray): points of the point cloud
        xyz_normals (np.ndarray): normals of the point cloud
    Returns:
        axis (np.ndarray): vector of the cylinder model axis
        radius (float): radius of the cylinder model axis
        center_point (np.ndarray): point belonging to the cylinder model axis
    """
    indexes = np.random.choice(xyz.shape[0], 2)
    points = xyz[indexes]
    normals = xyz_normals[indexes]
    # axis is a vector cross of two shape vectors
    axis = np.cross(normals[0], normals[1])
    axis /= np.linalg.norm(axis)

    # find the plane, containing axis and one of the points
    plane_normal = np.cross(axis, normals[0])
    plane_normal /= np.linalg.norm(plane_normal)
    ro = np.sum(points[0] * plane_normal)

    # find the crossing of found plane and vector of second points and its normal
    t = (ro - np.sum(plane_normal * points[1])) / np.sum(plane_normal * normals[1])
    center_point = normals[1] * t + points[1]

    # find radius which is the distance between found point belonging to axis and the second point
    radius = np.linalg.norm(center_point - points[1])

    return axis, radius, center_point


def cylinder_inliners(points, axis, radius, center_point, point_to_model_accuracy):
    """Getting the cylinder inliners

    Args:
        points (np.ndarray): points of the point cloud
        axis (np.ndarray): axis vector of the cylinder model
        radius (np.ndarray): radius of the cylinder model
        center_point (np.ndarray): point belonging to the axis
        point_to_model_accuracy (np.ndarray): max distance threshold between the model and the points
    """
    distances_dif = np.abs(np.linalg.norm(np.cross((points - center_point), axis), axis=1) - radius)
    inliners = distances_dif < point_to_model_accuracy
    return inliners, np.mean(distances_dif[inliners])


def cylinder_points(parameters, inliners, h_step=0.01, angle_step=math.radians(3)):
    """Generating points of the cylinder

    Args:
        parameters (np.ndarray): axis, radius and axis point of the cylinder model
        inliners (np.ndarray): inliners of the model
        h_step (float): step of the points generation along axis
        angle_step (float): step of the points generation along circle
    Returns:
        points (np.ndarray): points of the cylinder model
    """
    cylinder_axis, radius, center = parameters
    axis = [0, 1, 0]

    # shift points to the point and rotate to z-axis
    inliners -= center
    inliners = rotate(inliners, np.cross(axis, cylinder_axis), angle_between_normals(cylinder_axis, axis))

    # generate points
    h = np.arange(np.min(inliners[:, 1]), np.max(inliners[:, 1]) + h_step, h_step)
    phi = np.arange(0, math.pi * 2 + angle_step, angle_step)

    x = np.tile(radius * np.cos(phi), h.shape[0])
    z = np.tile(radius * np.sin(phi), h.shape[0])

    points = np.empty((x.shape[0], 3))
    points[:, 0], points[:, 1], points[:, 2] = x, np.repeat(h, points.shape[0] / h.shape[0]), z

    # return points to original position
    points = rotate(points, np.cross(cylinder_axis, axis), angle_between_normals(axis, cylinder_axis))
    points += center

    return points


def get_best_cone_model(points, normals, point_to_model_accuracy, number_of_subsets):
    """Getting the best cone model

    Args:
        points (np.ndarray): points of the point cloud
        normals (np.ndarray): normals of the point cloud
        point_to_model_accuracy (float): max distance threshold between model and the points
        number_of_subsets (int): number of subsets to chose the best
    Returns:
        best_apex (np.ndarray): apex of the cone model
        best_axis (np.ndarray): axis of the cone model
        best_alfa (float): opening angle of the cone model
        best_inliners (np.ndarray): inliners of the cone model
        best_mean (float): mean error of the inliners
    """
    best_score = 0
    best_apex = best_axis = best_alfa = 0
    best_mean = point_to_model_accuracy * 2
    best_inliners = 0
    for _ in range(number_of_subsets):
        apex, axis, alfa = cone_fitting(points, normals)
        inliners, mean = cone_inliners(points, apex, axis, alfa, point_to_model_accuracy)
        if np.sum(inliners) > best_score:
            best_score = np.sum(inliners)
            best_apex, best_axis, best_alfa = apex, axis, alfa
            best_inliners = inliners
            best_mean = mean
    return best_apex, best_axis, best_alfa, best_inliners, best_mean


def cone_fitting(xyz, xyz_normals):
    """Getting parameters of the cone

    Args:
        xyz (np.ndarray): points of the point cloud
        xyz_normals (np.ndarray): normals of the point cloud
    Returns:
        intersection_point (np.ndarray): apex of the cone model
        axis (np.ndarray): axis vector of the cone model
        alfa (float): opening model of the cone model
    """
    indexes = np.random.choice(xyz.shape[0], 3)
    points = xyz[indexes]
    normals = xyz_normals[indexes]

    # find three planes intersection
    # find two planes intersection
    n0 = np.asarray([[normals[0, 1], normals[0, 2]],
                     [normals[1, 1], normals[1, 2]]])
    n1 = np.asarray([[normals[0, 2], normals[0, 0]],
                     [normals[1, 2], normals[1, 0]]])
    n2 = np.asarray([[normals[0, 0], normals[0, 1]],
                     [normals[1, 0], normals[1, 1]]])
    direction_vector = np.asarray([np.linalg.det(n0), np.linalg.det(n1), np.linalg.det(n2)])
    direction_vector /= np.linalg.norm(direction_vector)

    ro_0 = np.sum(points[0] * normals[0])
    ro_1 = np.sum(points[1] * normals[1])
    delta = np.linalg.det(np.asarray([[normals[0, 0], normals[0, 1]],
                                      [normals[1, 0], normals[1, 1]]]))
    delta_x = np.linalg.det(np.asarray([[ro_0, normals[0, 1]],
                                        [ro_1, normals[1, 1]]]))
    delta_y = np.linalg.det(np.asarray([[normals[0, 0], ro_0],
                                        [normals[1, 0], ro_1]]))
    direction_point = np.asarray([delta_x / delta, delta_y / delta, 0])

    # line and point intersection
    ro_2 = np.sum(points[2] * normals[2])
    t = (ro_2 - np.sum(normals[2] * direction_point)) / np.sum(normals[2] * direction_vector)
    intersection_point = direction_vector * t + direction_point

    # find the axis

    plane_point_0 = intersection_point + (points[0] - intersection_point) / np.linalg.norm(
        points[0] - intersection_point)
    plane_point_1 = intersection_point + (points[1] - intersection_point) / np.linalg.norm(
        points[1] - intersection_point)
    plane_point_2 = intersection_point + (points[2] - intersection_point) / np.linalg.norm(
        points[2] - intersection_point)

    v1 = plane_point_2 - plane_point_0
    v2 = plane_point_1 - plane_point_0

    axis = np.cross(v1, v2)
    axis /= np.linalg.norm(axis)

    # find angle
    alfa = 0
    for i in range(3):
        angle = angle_between_normals(points[0] - intersection_point, axis)
        if angle > math.pi / 2:
            angle = math.pi - angle
        alfa += angle
    alfa /= 3

    return intersection_point, axis, alfa


def cone_inliners(points, apex, axis, alfa, point_to_model_accuracy):
    """Getting inliners of the cone model

    Args:
        points (np.ndarray): points of the point cloud
        apex (np.ndarray): apex of the cone model
        axis (np.ndarray): axis vector of the cone model
        alfa (np.ndarray): opening angle of the cone model
    Returns:
        inliners (np.ndarray): inliners of the cone model
        _ (float): mean axis of the inliners
    """
    p_a_vectors = points - apex
    p_a_cosang = np.dot(p_a_vectors, axis)
    p_a_sinang = np.linalg.norm(np.cross(p_a_vectors, axis), axis=1)
    p_a_angles = np.arctan2(p_a_sinang, p_a_cosang)
    p_a_angles = np.where(np.abs(p_a_angles) > math.pi / 2, math.pi - np.abs(p_a_angles), np.abs(p_a_angles))

    errors = np.sin(np.abs(p_a_angles - alfa)) * np.linalg.norm(p_a_vectors, axis=1)
    inliners = errors < point_to_model_accuracy
    return inliners, np.mean(errors[inliners])


def cone_points(parameters, points, h_step=0.005, angle_step=math.radians(3)):
    """Generating points of the cone model

    Args:
        parameters (np.ndarray): apex, axis and opening angle of the cone model
        points (np.ndarray): inliners of the cone model
        h_step (float): step of points generating along axis
        angle_step (float): step of points generating along circle
    Returns:
        points (np.ndarray): generated points of the cone model
    """
    apex, cone_axis, alfa = parameters
    axis = [0, 1, 0]

    # shift apex to origin and rotate to z-axis
    points -= apex
    inliners = rotate(points, np.cross(axis, cone_axis), angle_between_normals(cone_axis, axis))

    tan = math.tan(alfa)

    # generate points
    h = np.arange(np.min(inliners[:, 1]), np.max(inliners[:, 1]), h_step)
    phi = np.arange(0, math.pi * 2 + angle_step, angle_step)

    points = np.empty((h.shape[0] * phi.shape[0], 3))
    points[:, 1] = np.repeat(h, phi.shape[0])

    phi = np.tile(phi, h.shape[0])
    points[:, 0] = tan * points[:, 1] * np.cos(phi)
    points[:, 2] = tan * points[:, 1] * np.sin(phi)

    # return points to original position
    points = rotate(points, np.cross(cone_axis, axis), angle_between_normals(axis, cone_axis))
    points += apex

    return points


def angle_between_normals(n1, n2):
    """ Returns the angle in radians between vectors 'n1' and 'n2'"""
    cosang = np.dot(n1, n2)
    sinang = np.linalg.norm(np.cross(n1, n2))
    return np.abs(np.arctan2(sinang, cosang))


def get_count(number):
    """Counting numbers after the point"""
    s = str(number)
    if '.' in s:
        return abs(s.find('.') - len(s)) - 1
    else:
        return 0


def rotate(points, axis, angle):
    """Rotating points of the object

    Args:
        axis (numpy.array): axis according to which rotation must be done
        angle (numpy.array): angle on which rotation must be done
    """
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2.)
    b, c, d = -axis * np.sin(angle / 2.)
    R = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c), 0],
                  [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b), 0],
                  [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c, 0],
                  [0, 0, 0, 1]])

    A = np.zeros((points.shape[0], points.shape[1] + 1))
    A[:, :-1] = points[:, :]

    A = np.dot(R, A.T).T

    return A[:, :-1]
