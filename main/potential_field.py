from points_object import PointsObject
import numpy as np
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
import time


@dataclass
class MovementGaussian:
    speed_mean: np.ndarray = np.asarray([])
    acceleration_mean: np.ndarray = np.asarray([])
    speed_covariance: np.ndarray = np.asarray([])
    acceleration_covariance: np.ndarray = np.asarray([])
    weight: np.ndarray = np.asarray([])
    ranking: np.ndarray = np.asarray([])


class PotentialFieldObject:

    def __init__(self, object, number_of_gaussians=1, fading_coefficient=0):
        self.__object = object
        x_gaussians = MovementGaussian()
        y_gaussians = MovementGaussian()
        z_gaussians = MovementGaussian()
        angle_x_gaussians = MovementGaussian()
        angle_y_gaussians = MovementGaussian()
        angle_z_gaussians = MovementGaussian()
        self.__current_vector = np.zeros([])
        self.__fading_coefficient = fading_coefficient

    def interaction(self, passive_object, min_radius=0.01):
        center = self.__object.get_center()
        points = self.__object.get_points()[0]
        passive_object_points = passive_object.get_points()[0]
        object_vectors = self.sort_object_points(center, points)
        points_into_object_radius = self.potential_interceptions(center, passive_object_points, object_vectors[0, 1])
        if points_into_object_radius.size == 0:
            print("no interceptions")
            return None
        else:
            vectors = np.empty((0,2), float)
            for index, distance in points_into_object_radius:
                potential_vectors = self.potential_interceptions(passive_object_points[int(index)], points, min_radius)
                if not potential_vectors.size == 0:
                    vectors = np.r_[vectors, potential_vectors]
        print(points[vectors[:,0].astype(int)])

    def sort_object_points(self, center_point, points):
        nbrs = NearestNeighbors(n_neighbors=points.shape[0], algorithm='auto').fit(points)
        distances, indices = nbrs.kneighbors(center_point.reshape(1, -1))
        return np.flip(np.dstack((indices, distances)), 1)[0]

    def potential_interceptions(self, center_point, points, max_dist):
        nbrs = NearestNeighbors(n_neighbors=points.shape[0], algorithm='auto').fit(points)
        distances, indices = nbrs.kneighbors(center_point.reshape(1, -1))
        return np.dstack((indices[distances < max_dist], distances[distances < max_dist]))[0]
