import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy
import cv2


class CovarianceDescriptor:

    def __init__(self, xyz_points, color_points, normals, k_nearest_neighbours=None, relevant_distance=None,
                 use_alpha=False, use_beta=False, use_theta=False, use_ro=False, use_psi=False, use_rgb=False,
                 use_luv=False, use_normals=False):
        self.__xyz = xyz_points
        self.__color = color_points
        self.__normals = normals

        self.__k_nearest_neighbours = k_nearest_neighbours

        self.__features = {'alpha': use_alpha, 'beta': use_beta, 'theta': use_theta, 'ro': use_ro, 'psi': use_psi,
                           'rgb': use_rgb, 'luv': use_luv, 'normals': use_normals}
        self.__number_of_active_parameters = np.sum(np.asarray(
            [use_alpha, use_beta, use_theta, use_ro, use_psi, use_rgb, use_rgb, use_rgb, use_luv, use_luv, use_luv,
             use_normals, use_normals, use_normals]))

        self.__object_descriptor = self.create_descriptors(self.__xyz, self.__color, self.__normals)

    def create_point_descriptor(self, point_xyz, point_color, point_normal, indices):
        feature_vectors = self.create_feature_vectors(point_xyz, point_color, point_normal, indices)
        mean_values = np.mean(feature_vectors, axis=0)
        temp_matrix = feature_vectors - mean_values
        covariance_matrix = (temp_matrix.T).dot(temp_matrix) / (indices.shape[0])
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
            theta[np.isnan(theta)] = 100
            feature_vectors[:, current_feature_number] = theta
            current_feature_number += 1
        if self.__features['ro']:
            ro = np.linalg.norm(pp_vector, axis=1)
            feature_vectors[:, current_feature_number] = ro
            current_feature_number += 1
        if self.__features['psi']:
            psi = np.arccos(np.tensordot(normals, point_normal, axes=([1], [0])))
            psi[np.isnan(psi)] = 1000
            feature_vectors[:, current_feature_number] = psi
            current_feature_number += 1
        if self.__features['normals']:
            n_x, n_y, n_z = self.__normals[indices[1:], 0], self.__normals[indices[1:], 1], self.__normals[
                indices[1:], 2]
            feature_vectors[:, current_feature_number] = n_x
            current_feature_number += 1
            feature_vectors[:, current_feature_number] = n_y
            current_feature_number += 1
            feature_vectors[:, current_feature_number] = n_z
            current_feature_number += 1
        if self.__features['rgb']:
            r, g, b = self.__color[indices[1:], 0], self.__color[indices[1:], 1], self.__color[indices[1:], 2]
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

    def create_descriptors(self, xyz, color, normals):

        descriptor = np.zeros([xyz.shape[0], self.__number_of_active_parameters, self.__number_of_active_parameters])

        if self.__k_nearest_neighbours is not None and self.__k_nearest_neighbours < xyz.shape[0]:
            nbrs = NearestNeighbors(n_neighbors=self.__k_nearest_neighbours, algorithm='auto').fit(xyz)
            distances, indices = nbrs.kneighbors(xyz)
        else:
            indices = np.zeros([xyz.shape[0], xyz.shape[0]]).astype(int)
            indices[:] = np.arange(xyz.shape[0])

        for i in range(xyz.shape[0]):
            if self.__k_nearest_neighbours is None or self.__k_nearest_neighbours >= xyz.shape[0]:
                indices[i, 0], indices[i, i] = indices[i, i], indices[i, 0]
            descriptor[i] = self.create_point_descriptor(xyz[i], color[i], normals[i], indices[i])
        return descriptor

    def descriptors_distances(self, original_object_descriptor, compared_object_descriptors):
        descriptors_distances = np.zeros([original_object_descriptor.shape[0], compared_object_descriptors.shape[0]])

        for original_descriptor in original_object_descriptor:
            try:
                original_descriptor = scipy.linalg.logm(original_descriptor)
            except:
                print(original_descriptor)
        for compared_descriptor in compared_object_descriptors:
            compared_descriptor = scipy.linalg.logm(compared_descriptor)

        # for i, original_descriptor in enumerate(original_object_descriptor):
        #     for j, compared_descriptor in enumerate(compared_object_descriptors):
        #         descriptors_distances[i, j] = np.linalg.norm(original_descriptor - compared_descriptor, ord='fro')

        for i, original_descriptor in enumerate(original_object_descriptor):
            descriptors_distances[i] = np.linalg.norm(compared_object_descriptors - original_descriptor, axis=(1, 2),
                                                      ord='fro')
        return descriptors_distances.T

    def compare_objects(self, compared_object_xyz, compared_object_color, compared_object_normals,
                        number_of_random_points=None):
        if number_of_random_points is None or number_of_random_points > compared_object_xyz.shape[0]:
            number_of_random_points = compared_object_xyz.shape[0]
        compared_object_descriptors = self.create_descriptors(compared_object_xyz, compared_object_color,
                                                              compared_object_normals)
        return self.compare_descriptors(compared_object_descriptors, number_of_random_points)

    def compare_descriptors(self, compared_object_descriptors, number_of_random_points):
        rng = np.random.default_rng()
        if number_of_random_points > compared_object_descriptors.shape[0]:
            number_of_random_points = compared_object_descriptors.shape[0]
        idx = rng.choice(compared_object_descriptors.shape[0], size=number_of_random_points, replace=False)
        distances = self.descriptors_distances(self.__object_descriptor, compared_object_descriptors[idx, :, :])

        return np.amin(distances, axis=1)

    @property
    def object_descriptor(self):
        return self.__object_descriptor


class GlobalCovarianceDescriptor:
    def __init__(self, xyz_points, color_points, normals, depth_image, rgb_image, mask_image,
                 use_xyz=False, use_rgb=False, use_normals=False, use_intensity=False, use_depth=False,
                 use_intensity_magnitude=False, use_depth_magnitude=False):
        self.__features = {'xyz': use_xyz, 'rgb': use_rgb, 'normals': use_normals, 'intensity': use_intensity,
                           'depth': use_depth, 'intensity_magnitude': use_intensity_magnitude,
                           'depth_magnitude': use_depth_magnitude}
        self.__number_of_active_parameters = int(use_xyz == True) * 3 + int(use_rgb == True) * 3 + int(
            use_normals == True) * 3 + int(use_intensity == True) * 6 + int(use_depth == True) * 2 + int(
            use_intensity_magnitude == True) + int(use_depth_magnitude == True)

        self.__object_descriptor = self.create_descriptor(xyz_points, color_points, normals, depth_image, rgb_image,
                                                          mask_image)

    def create_descriptor(self, xyz, color, normal, depth_image, color_image, mask):
        feature_vectors = self.create_feature_vectors(xyz, color, normal, depth_image, color_image, mask)
        mean_values = np.mean(feature_vectors, axis=0)
        temp_matrix = feature_vectors - mean_values
        covariance_matrix = (temp_matrix.T).dot(temp_matrix) / (feature_vectors.shape[0])
        return covariance_matrix

    def create_feature_vectors(self, xyz, color, normal, depth_image, color_image, mask):
        current_feature_number = 0
        feature_vectors = np.zeros([xyz.shape[0], self.__number_of_active_parameters])
        if self.__features['xyz']:
            feature_vectors[:, current_feature_number:current_feature_number + 3] = xyz
            current_feature_number += 3
        if self.__features['rgb']:
            feature_vectors[:, current_feature_number:current_feature_number + 3] = color
            current_feature_number += 3
        if self.__features['normals']:
            feature_vectors[:, current_feature_number:current_feature_number + 3] = normal
            current_feature_number += 3
        if self.__features['intensity']:
            flatten_mask = mask.flatten()
            grey_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            Ix = cv2.Sobel(grey_image, cv2.CV_64F, 1, 0, ksize=5)
            Iy = cv2.Sobel(grey_image, cv2.CV_64F, 0, 1, ksize=5)
            Ixx = (cv2.Sobel(Ix, cv2.CV_64F, 1, 0, ksize=5).flatten())[flatten_mask == 1]
            Iyy = (cv2.Sobel(Iy, cv2.CV_64F, 0, 1, ksize=5).flatten())[flatten_mask == 1]
            Ixy = (cv2.Sobel(Ix, cv2.CV_64F, 0, 1, ksize=5).flatten())[flatten_mask == 1]
            I = (grey_image.flatten())[flatten_mask == 1]
            if self.__features['intensity_magnitude']:
                M = np.sqrt(Ix ** 2 + Iy ** 2)
                M = (M.flatten())[flatten_mask == 1]
                feature_vectors[:, current_feature_number] = M[:xyz.shape[0]]
                current_feature_number += 1
            Ix = (Ix.flatten())[flatten_mask == 1]
            Iy = (Iy.flatten())[flatten_mask == 1]
            feature_vectors[:, current_feature_number] = I[:xyz.shape[0]]
            current_feature_number += 1
            feature_vectors[:, current_feature_number] = Ix[:xyz.shape[0]]
            current_feature_number += 1
            feature_vectors[:, current_feature_number] = Iy[:xyz.shape[0]]
            current_feature_number += 1
            feature_vectors[:, current_feature_number] = Ixy[:xyz.shape[0]]
            current_feature_number += 1
            feature_vectors[:, current_feature_number] = Ixx[:xyz.shape[0]]
            current_feature_number += 1
            feature_vectors[:, current_feature_number] = Iyy[:xyz.shape[0]]
            current_feature_number += 1
        if self.__features['depth']:
            flatten_mask = mask.flatten()
            Dx = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=5)
            Dy = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=5)
            if self.__features['depth_magnitude']:
                D = np.sqrt(Dx ** 2 + Dy ** 2)
                D = (D.flatten())[flatten_mask == 1]
                feature_vectors[:, current_feature_number] = D[:xyz.shape[0]]
                current_feature_number += 1
            Dx = (Dx.flatten())[flatten_mask == 1]
            Dy = (Dy.flatten())[flatten_mask == 1]
            feature_vectors[:, current_feature_number] = Dx[:xyz.shape[0]]
            current_feature_number += 1
            feature_vectors[:, current_feature_number] = Dx[:xyz.shape[0]]
            current_feature_number += 1

        # crutch
        # rng = np.random.default_rng()
        # idx = rng.choice(xyz.shape[0], size=int(1000), replace=False)
        # return feature_vectors[idx, :]

        return feature_vectors

    def compare_descriptors(self, compared_descriptor):

        distance = np.linalg.norm(scipy.linalg.logm(self.__object_descriptor) - scipy.linalg.logm(compared_descriptor),
                                  ord='fro')

        return distance

    @property
    def object_descriptor(self):
        return self.__object_descriptor


if __name__ == "__main__":
    from moving_detection import RGBD_MoG
    from moving_detection import region_growing
    import image_processing
    import cv2
    from points_object import PointsObject
    import visualization
    import download_point_cloud
    import time
    import os


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


    def object_from_point_cloud(path, number_of_points):
        xyz_points, rgb_points = download_point_cloud.download_ply(path)
        if number_of_points is not None and number_of_points > xyz_points.shape[0]:
            number_of_points = xyz_points.shape[0]
        object_points = PointsObject()
        object_points.set_points(xyz_points, rgb_points, number_of_points)
        object_points.scale(0.5)
        # visualization.visualize_object([object_points])
        return object_points


    def objects_test_simple_figures():

        # creating descriptor for object
        # object_from_picture()
        orange_sphere = object_from_point_cloud("models/brown cylinder.ply", 1000)
        coordinates, color = orange_sphere.get_points()
        norms = orange_sphere.get_normals()

        start = time.time()
        orange_sphere_descriptor = CovarianceDescriptor(coordinates, color, norms, k_nearest_neighbours=None,
                                                        relevant_distance=0.5, use_alpha=True, use_beta=True,
                                                        use_ro=True,
                                                        use_theta=True, use_psi=True)
        print(time.time() - start)

        number_of_points = 100

        # adding a new object and finding matches
        print("blue conus:")
        compared_object = object_from_point_cloud("models/blue conus.ply", number_of_points)
        coordinates, color = compared_object.get_points()
        norms = compared_object.get_normals()

        start = time.time()
        orange_sphere_descriptor.compare_objects(coordinates, color, norms)
        print(time.time() - start)

        print("brown cylinder:")
        compared_object = object_from_point_cloud("models/brown cylinder.ply", number_of_points)
        coordinates, color = compared_object.get_points()
        norms = compared_object.get_normals()

        start = time.time()
        orange_sphere_descriptor.compare_objects(coordinates, color, norms)
        print(time.time() - start)

        print("green isosphere:")
        compared_object = object_from_point_cloud("models/green isosphere.ply", number_of_points)
        coordinates, color = compared_object.get_points()
        norms = compared_object.get_normals()

        start = time.time()
        orange_sphere_descriptor.compare_objects(coordinates, color, norms)
        print(time.time() - start)

        print("grey plane:")
        compared_object = object_from_point_cloud("models/grey plane.ply", number_of_points)
        coordinates, color = compared_object.get_points()
        norms = compared_object.get_normals()

        start = time.time()
        orange_sphere_descriptor.compare_objects(coordinates, color, norms)
        print(time.time() - start)

        print("orange sphere:")
        compared_object = object_from_point_cloud("models/orange sphere.ply", number_of_points)
        coordinates, color = compared_object.get_points()
        norms = compared_object.get_normals()

        start = time.time()
        orange_sphere_descriptor.compare_objects(coordinates, color, norms)
        print(time.time() - start)

        print("red cube:")
        compared_object = object_from_point_cloud("models/red cube.ply", number_of_points)
        coordinates, color = compared_object.get_points()
        norms = compared_object.get_normals()

        start = time.time()
        orange_sphere_descriptor.compare_objects(coordinates, color, norms)
        print(time.time() - start)

        print("violet thor:")
        compared_object = object_from_point_cloud("models/violet thor.ply", number_of_points)
        coordinates, color = compared_object.get_points()
        norms = compared_object.get_normals()

        start = time.time()
        orange_sphere_descriptor.compare_objects(coordinates, color, norms)
        print(time.time() - start)


    def objects_test_real_figures():

        original_object = object_from_point_cloud("PCDs/real_objects/coffee_mug_5.pcd", 1000)
        # visualization.visualize_object([original_object])

        coordinates, color = original_object.get_points()
        norms = original_object.get_normals()

        start = time.time()
        original_object_descriptor = CovarianceDescriptor(coordinates, color, norms, k_nearest_neighbours=None,
                                                          relevant_distance=0.1, use_alpha=True, use_beta=True,
                                                          use_ro=True, use_theta=True, use_psi=True)
        print(time.time() - start)

        lengths = {}
        for filename in os.listdir("PCDs/real_objects"):
            start = time.time()

            current_object = object_from_point_cloud("PCDs/real_objects/" + filename, 1000)
            coordinates, color = current_object.get_points()
            norms = current_object.get_normals()
            l = original_object_descriptor.compare_objects(coordinates, color, norms, 100)
            lengths[filename[:-6]] = l
            print(time.time() - start)
        for item in lengths:
            print(item, ' : ', lengths[item])


    def objects_test_real_figures_v2():
        descriptors = {}
        number_of_points = 1000
        for filename in os.listdir("PCDs/real_objects"):
            if filename[-4:] == ".pcd":
                start = time.time()

                current_object = object_from_point_cloud("PCDs/real_objects/" + filename, number_of_points)
                coordinates, color = current_object.get_points()
                norms = current_object.get_normals()

                descriptors[filename[:-6]] = CovarianceDescriptor(coordinates, color, norms, k_nearest_neighbours=None,
                                                                  relevant_distance=0.1, use_alpha=True, use_beta=True,
                                                                  use_ro=True, use_theta=True, use_psi=True,
                                                                  use_rgb=True)
                print(time.time() - start)

        for filename in os.listdir("PCDs/real_objects"):
            if filename[-4:] == ".pcd":
                compared_object = object_from_point_cloud("PCDs/real_objects/" + filename, number_of_points)
                coordinates, color = compared_object.get_points()
                norms = compared_object.get_normals()
                compared_object_descriptor = CovarianceDescriptor(coordinates, color, norms, k_nearest_neighbours=None,
                                                                  relevant_distance=0.1, use_alpha=True, use_beta=True,
                                                                  use_ro=True, use_theta=True, use_psi=True,
                                                                  use_rgb=True)

                number_of_comparing_points = 100
                length_values = np.zeros([len(descriptors), number_of_comparing_points])
                for i, object_class in enumerate(descriptors):
                    start = time.time()
                    length_values[i] = descriptors[object_class].compare_descriptors(
                        compared_object_descriptor.object_descriptor,
                        number_of_comparing_points)
                    print(time.time() - start)

                print(filename)
                for i, object_class in enumerate(list(descriptors.keys())):
                    print(object_class, " : ",
                          np.count_nonzero(np.argmin(length_values, axis=0) == i) / number_of_comparing_points * 100)


    def objects_test_real_figures_with_global():

        path = "PCDs/real_objects/"

        filename = "scissors_2.pcd"
        compared_object = object_from_point_cloud(path + filename, None)
        mask_image = cv2.imread(path + filename[:-4] + "_maskcrop.png", cv2.IMREAD_GRAYSCALE)
        mask = cv2.normalize(mask_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        depth_image = cv2.imread(path + filename[:-4] + "_depthcrop.png", cv2.IMREAD_ANYDEPTH) * mask
        mask = np.where(depth_image == 0, 0, 1)
        rgb_image = cv2.imread(path + filename[:-4] + "_crop.png")
        rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2] = rgb_image[:, :, 0] * mask, rgb_image[:, :, 1] * \
                                                                     mask, rgb_image[:, :, 2] * mask
        coordinates, color = compared_object.get_points()
        norms = compared_object.get_normals()
        if np.sum(mask) < coordinates.shape[0]:
            coordinates = coordinates[:np.sum(mask)]
            color = color[:np.sum(mask)]
            norms = norms[:np.sum(mask)]

        start = time.time()
        original_object_descriptor = GlobalCovarianceDescriptor(coordinates, color, norms, depth_image, rgb_image, mask,
                                                                use_xyz=True, use_rgb=True, use_normals=True)
        print(time.time() - start)

        for filename in os.listdir("PCDs/real_objects"):
            if filename[-4:] == ".pcd":
                compared_object = object_from_point_cloud(path + filename, None)
                mask_image = cv2.imread(path + filename[:-4] + "_maskcrop.png", cv2.IMREAD_GRAYSCALE)
                mask = cv2.normalize(mask_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                depth_image = cv2.imread(path + filename[:-4] + "_depthcrop.png", cv2.IMREAD_ANYDEPTH) * mask
                mask = np.where(depth_image == 0, 0, 1)
                rgb_image = cv2.imread(path + filename[:-4] + "_crop.png")
                rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2] = rgb_image[:, :, 0] * mask, rgb_image[:, :,
                                                                                                        1] * \
                                                                             mask, rgb_image[:, :, 2] * mask
                coordinates, color = compared_object.get_points()
                norms = compared_object.get_normals()

                if np.sum(mask) < coordinates.shape[0]:
                    coordinates = coordinates[:np.sum(mask)]
                    color = color[:np.sum(mask)]
                    norms = norms[:np.sum(mask)]

                compared_object_descriptor = GlobalCovarianceDescriptor(coordinates, color, norms, depth_image,
                                                                        rgb_image,
                                                                        mask,
                                                                        use_xyz=True, use_rgb=True, use_normals=True)
                start = time.time()
                print(filename, ':',
                      original_object_descriptor.compare_descriptors(compared_object_descriptor.object_descriptor))
                print(time.time() - start)


    def objects_test_moving_figures_local():
        number_of_comparing_points = 100
        classes = {}

        rgb_im = image_processing.load_image("falling balls and cylinder", "rgb_" + str(0) + ".png")
        depth_im = image_processing.load_image("falling balls and cylinder", "depth_" + str(0) + ".png", "depth")
        mog = RGBD_MoG(rgb_im, depth_im, number_of_gaussians=3)

        for number_of_frame in range(1, 5):

            color_mask = np.zeros([rgb_im.shape[0], rgb_im.shape[1], 3])

            rgb_im = image_processing.load_image("falling balls and cylinder", "rgb_" + str(number_of_frame) + ".png")
            depth_im = image_processing.load_image("falling balls and cylinder",
                                                   "depth_" + str(number_of_frame) + ".png",
                                                   "depth")
            mask = mog.set_mask(rgb_im, depth_im)

            depth_im = depth_im * (mask / 255).astype(int)
            masks = region_growing(mask / 255, depth_im / 255, depth_threshold=0.01, significant_number_of_points=10)
            if len(masks) == 0:
                print("No moving objects in the frame")
            else:
                for mask in masks:
                    xyz_points, rgb_points = image_processing.calculate_point_cloud(rgb_im / 255, depth_im * mask / 255)
                    current_object = PointsObject()
                    current_object.set_points(xyz_points, rgb_points)
                    norms = current_object.get_normals()
                    compared_object_descriptor = CovarianceDescriptor(xyz_points, rgb_points, norms,
                                                                      k_nearest_neighbours=None,
                                                                      relevant_distance=0.1, use_alpha=True,
                                                                      use_beta=True,
                                                                      use_ro=True, use_theta=True, use_psi=True,
                                                                      use_rgb=True)
                    match_found = False
                    lengths = np.zeros([len(classes), np.amin([number_of_comparing_points, xyz_points.shape[0]])])

                    if number_of_frame == 1:
                        match_found = False
                    else:
                        match_found = True
                        for object_number, object_class in enumerate(classes):
                            lengths[object_number] = object_class.compare_descriptors(
                                compared_object_descriptor.object_descriptor, number_of_comparing_points)
                            print(np.sum(mask))
                        min_args = np.argmin(lengths, axis=0)[np.amin(lengths, axis=0) < 0.1]
                        unique, counts = np.unique(min_args, return_counts=True)
                        best_match = unique[np.argmax(counts)]
                        for object_number, object_class in enumerate(classes):
                            if object_number == best_match:
                                color_mask[:, :, 0] += mask * classes[object_class][0]
                                color_mask[:, :, 1] += mask * classes[object_class][1]
                                color_mask[:, :, 2] += mask * classes[object_class][2]

                    if not match_found:
                        classes[compared_object_descriptor] = np.random.rand(3)
                        color_mask[:, :, 0] += mask * classes[compared_object_descriptor][0]
                        color_mask[:, :, 1] += mask * classes[compared_object_descriptor][1]
                        color_mask[:, :, 2] += mask * classes[compared_object_descriptor][2]
                image_processing.save_image(color_mask, "tracking_results", frame_number=number_of_frame,
                                            image_name="local_two_same")

                # visualization.visualize_object([current_object])


    def objects_test_moving_figures_global():
        classes = {}

        rgb_im = image_processing.load_image("falling balls and cylinder", "rgb_" + str(0) + ".png")
        depth_im = image_processing.load_image("falling balls and cylinder", "depth_" + str(0) + ".png", "depth")
        mog = RGBD_MoG(rgb_im, depth_im, number_of_gaussians=3)

        for number_of_frame in range(1, 5):

            color_mask = np.zeros([rgb_im.shape[0], rgb_im.shape[1], 3])

            rgb_im = image_processing.load_image("falling balls and cylinder", "rgb_" + str(number_of_frame) + ".png")
            depth_im = image_processing.load_image("falling balls and cylinder",
                                                   "depth_" + str(number_of_frame) + ".png",
                                                   "depth")
            mask = mog.set_mask(rgb_im, depth_im)

            depth_im = depth_im * (mask / 255).astype(int)
            masks = region_growing(mask / 255, depth_im / 255, depth_threshold=0.01, significant_number_of_points=10)
            if len(masks) == 0:
                print("No moving objects in the frame")
            else:
                for mask in masks:
                    xyz_points, rgb_points = image_processing.calculate_point_cloud(rgb_im / 255, depth_im * mask / 255)
                    current_object = PointsObject()
                    current_object.set_points(xyz_points, rgb_points)
                    norms = current_object.get_normals()
                    compared_object_descriptor = GlobalCovarianceDescriptor(xyz_points, rgb_points, norms, depth_im,
                                                                            rgb_im, mask, use_xyz=True, use_rgb=True,
                                                                            use_normals=True)
                    match_found = False
                    lengths = np.zeros([len(classes)])

                    if number_of_frame == 1:
                        match_found = False
                    else:
                        match_found = True
                        for object_number, object_class in enumerate(classes):
                            lengths[object_number] = object_class.compare_descriptors(
                                compared_object_descriptor.object_descriptor)
                        min_arg = np.argmin(lengths)
                        print(lengths)
                        for object_number, object_class in enumerate(classes):
                            if object_number == min_arg:
                                color_mask[:, :, 0] += mask * classes[object_class][0]
                                color_mask[:, :, 1] += mask * classes[object_class][1]
                                color_mask[:, :, 2] += mask * classes[object_class][2]

                    if not match_found:
                        classes[compared_object_descriptor] = np.random.rand(3)
                        color_mask[:, :, 0] += mask * classes[compared_object_descriptor][0]
                        color_mask[:, :, 1] += mask * classes[compared_object_descriptor][1]
                        color_mask[:, :, 2] += mask * classes[compared_object_descriptor][2]
                image_processing.save_image(color_mask, "tracking_results", frame_number=number_of_frame,
                                            image_name="global_two_same")


    def rename_files():
        path = "PCDs/real_objects/"
        for filename in os.listdir("PCDs/real_objects"):
            print(filename)
            dst = filename[:-8] + filename[-4:]
            src = path + filename
            dst = path + dst
            #
            # # rename() function will
            # # rename all the files
            os.rename(src, dst)
            # i += 1


    # object_from_picture()
    objects_test_moving_figures_global()
    # objects_test_real_figures_with_global()
    # rename_files()
    # for filename in os.listdir("PCDs/real_objects"):
    #     if filename[-4:] == ".pcd":
    #         original_object = object_from_point_cloud("PCDs/real_objects/" + filename, 5000)
    #         visualization.visualize_object([original_object])
    # objects_test_real_figures_v2()
    # original_object = object_from_point_cloud("PCDs/apple_5_1_1.pcd", 5000)
    # visualization.visualize_object([original_object])
