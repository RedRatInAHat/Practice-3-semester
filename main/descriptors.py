import numpy as np
from sklearn.neighbors import NearestNeighbors


class CovarianceDescriptor:

    def __init__(self, xyz_points, color_points, normals, k_nearest_neighbours=None, use_alpha=False, use_beta=False,
                 use_theta=False, use_ro=False, use_psi=False, use_rgb=False, use_luv=False, use_normals=False):
        self.__xyz = xyz_points
        self.__color = color_points
        self.__normals = normals

        self.__k_nearest_neighbours = k_nearest_neighbours if k_nearest_neighbours is not None \
                                                              and k_nearest_neighbours < self.__xyz.shape[0] else None

        self.__features = {'alpha': use_alpha, 'beta': use_beta, 'theta': use_theta, 'ro': use_ro, 'psi': use_psi,
                           'rgb': use_rgb, 'luv': use_luv, 'normals': use_normals}
        self.__number_of_active_parameters = np.sum(np.asarray(
            [use_alpha, use_beta, use_theta, use_ro, use_psi, use_rgb, use_rgb, use_rgb, use_luv, use_luv, use_luv,
             use_normals, use_normals, use_normals]))

        self.__descriptor = np.zeros([xyz_points.shape[0], self.__number_of_active_parameters,
                                     self.__number_of_active_parameters])

        self.create_descriptors()

    def create_point_descriptor(self, point_xyz, point_color, point_normal, indices):
        feature_vectors = self.create_feature_vectors(point_xyz, point_color, point_normal, indices)
        mean_values = np.mean(feature_vectors, axis=0)
        temp_matrix = feature_vectors - mean_values
        covariance_matrix = (temp_matrix.T).dot(temp_matrix)
        return covariance_matrix

    def create_feature_vectors(self, point_xyz, point_color, point_normal, indices):
        feature_vectors = np.zeros([indices.shape[0] - 1, self.__number_of_active_parameters])
        current_feature_number = 0
        pp_vector = self.__xyz[indices[1:], :] - point_xyz
        normals = self.__normals[indices[1:], :]

        if self.__features['alpha'] or self.__features['beta'] or self.__features['theta']:
            pp_n = np.tensordot(pp_vector, point_normal, axes=([1], [0]))

        if self.__features['alpha'] or self.__features['beta']:
            pp_n_n = np.zeros_like(pp_vector)
            pp_n_n[:, 0], pp_n_n[:, 1], pp_n_n[:, 2] = point_normal[0] * pp_n, point_normal[1] * pp_n, point_normal[
                2] * pp_n

        if self.__features['alpha']:
            alpha = np.linalg.norm(pp_vector - pp_n_n, axis=1)
            feature_vectors[:, current_feature_number] = alpha
            current_feature_number += 1
        if self.__features['beta']:
            beta = np.linalg.norm(pp_n_n, axis=1)
            feature_vectors[:, current_feature_number] = beta
            current_feature_number += 1
        if self.__features['theta']:
            theta = np.arccos(pp_n / np.linalg.norm(pp_vector, axis=1))
            feature_vectors[:, current_feature_number] = theta
            current_feature_number += 1
        if self.__features['ro']:
            ro = np.linalg.norm(pp_vector, axis=1)
            feature_vectors[:, current_feature_number] = ro
            current_feature_number += 1
        if self.__features['psi']:
            psi = np.arccos(np.tensordot(normals, point_normal, axes=([1], [0])))
            feature_vectors[:, current_feature_number] = psi
            current_feature_number += 1
        if self.__features['normals']:
            n_x, n_y, n_z = normals[:, 0], normals[:, 1], normals[:, 2]
            feature_vectors[:, current_feature_number] = n_x
            current_feature_number += 1
            feature_vectors[:, current_feature_number] = n_y
            current_feature_number += 1
            feature_vectors[:, current_feature_number] = n_z
            current_feature_number += 1
        if self.__features['rgb']:
            r, g, b = point_color[:, 0], point_color[:, 1], point_color[:, 2]
            feature_vectors[:, current_feature_number] = r
            current_feature_number += 1
            feature_vectors[:, current_feature_number] = g
            current_feature_number += 1
            feature_vectors[:, current_feature_number] = b
            current_feature_number += 1
        if self.__features['luv']:
            l, u, v = point_color[:, 0], point_color[:, 1], point_color[:, 2]
            feature_vectors[:, current_feature_number] = l
            current_feature_number += 1
            feature_vectors[:, current_feature_number] = u
            current_feature_number += 1
            feature_vectors[:, current_feature_number] = v
            current_feature_number += 1

        return feature_vectors

    def create_descriptors(self):

        if self.__k_nearest_neighbours is not None:
            nbrs = NearestNeighbors(n_neighbors=self.__k_nearest_neighbours, algorithm='auto').fit(self.__xyz)
            distances, indices = nbrs.kneighbors(self.__xyz)
        else:
            indices = np.zeros([self.__xyz.shape[0], self.__xyz.shape[0]]).astype(int)
            indices[:] = np.arange(self.__xyz.shape[0])

        for i in range(self.__xyz.shape[0]):
            if self.__k_nearest_neighbours is None:
                indices[i, 0], indices[i, i] = indices[i, i], indices[i, 0]
            self.__descriptor[i] = self.create_point_descriptor(self.__xyz[i], self.__color[i], self.__normals[i],
                                                                indices[i])

    def compare_points(self):
        pass


if __name__ == "__main__":
    from moving_detection import RGBD_MoG
    from moving_detection import region_growing
    import image_processing
    import cv2
    from points_object import PointsObject
    import numpy as np
    import visualization
    import download_point_cloud
    import time
    import open3d


    def get_moving_mask(number_of_frame=1):
        rgb_im = image_processing.load_image("falling balls and cylinder", "rgb_" + str(0) + ".png")
        depth_im = image_processing.load_image("falling balls and cylinder", "depth_" + str(0) + ".png", "depth")
        mog = RGBD_MoG(rgb_im, depth_im, number_of_gaussians=3)

        rgb_im = image_processing.load_image("falling balls and cylinder", "rgb_" + str(number_of_frame) + ".png")
        depth_im = image_processing.load_image("falling balls and cylinder", "depth_" + str(number_of_frame) + ".png",
                                               "depth")
        mask = mog.set_mask(rgb_im, depth_im)
        return mask, depth_im, rgb_im


    def get_one_object_mask(mask, depth, depth_threshold=0.1, min_number_of_points=10, number_of_object=0):
        masks = region_growing(mask, depth, depth_threshold, min_number_of_points)
        if len(masks) == 0:
            return []
        elif number_of_object > len(masks) - 1:
            return masks[len(masks) - 1]
        else:
            return masks[number_of_object]


    def show_image(image):
        cv2.imshow("", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def object_from_picture():
        mask, depth, rgb = get_moving_mask()
        object_mask = get_one_object_mask(mask / 255, depth / 255, depth_threshold=0.05, number_of_object=1)
        xyz_points, rgb_points = image_processing.calculate_point_cloud(rgb / 255, depth * object_mask / 255)
        pc = PointsObject()
        pc.set_points(xyz_points, rgb_points)
        visualization.visualize_object([pc])
        # cov_desc = CovarianceDescriptor(points_cloud)


    def object_from_point_cloud(path):
        xyz_points, rgb_points = download_point_cloud.download_ply(path)
        object_points = PointsObject()
        object_points.set_points(xyz_points, rgb_points, 5000)
        object_points.scale(0.5)
        # visualization.visualize_object([object_points])
        return object_points


    # object_from_picture()
    orange_sphere = object_from_point_cloud("models/orange sphere.ply")
    coordinates, color = orange_sphere.get_points()
    norms = orange_sphere.get_normals()

    start = time.time()
    orange_sphere_descriptor = CovarianceDescriptor(coordinates, color, norms, k_nearest_neighbours=500, use_alpha=True,
                                                    use_beta=True)
    print(time.time() - start)
