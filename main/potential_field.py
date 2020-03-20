from points_object import PointsObject
import numpy as np
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
import time
import math


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
        self.__current_vector = np.asarray([0, 0, 0])
        self.__gravity = np.asarray([0, -1, 0])
        self.__fading_coefficient = fading_coefficient
        self.__summary_vector_threshold = 0.001

    def interaction(self, passive_object, min_radius=0.01):

        object_is_stable = False
        center = self.__object.get_center()
        passive_object_points = passive_object.get_points()[0]
        passive_object_normals = passive_object.get_normals()
        object_points = self.__object.get_points()[0]
        object_vectors = self.sort_object_points(center, object_points)

        while not object_is_stable:

            points_into_object_radius_indexes = self.potential_interceptions(center, passive_object_points,
                                                                             object_vectors[0, 1])[:, 0].astype(int)

            if points_into_object_radius_indexes.size == 0:
                print("no interceptions")
                self.visualize(stable_object=passive_object)
            else:
                more_indexes = self.potential_interceptions(center, passive_object_points, object_vectors[0, 1])[
                               :, 0].astype(int)
                second_object_points, second_object_normals = self.find_positive_vectors(
                    (passive_object_points[more_indexes]), passive_object_normals[more_indexes] / 50, center)

                drown_points_indexes = self.find_drown_points(object_points, second_object_points,
                                                              second_object_normals)

                crutch = 0
                while not drown_points_indexes.size == 0:

                    normals = center - self.__object.get_normals()[drown_points_indexes] / 50
                    normal_vector = np.mean(normals, axis=0)
                    summary_vector = self.__gravity / 50 + self.__current_vector + normal_vector

                    print(summary_vector)

                    extreme_point = self.find_extreme_point(object_points[drown_points_indexes], summary_vector, center)
                    rotation_point_index = np.where(np.all(object_points == extreme_point, axis=1))[0]

                    self.make_it_not_drown(object_points[rotation_point_index], second_object_points,
                                           second_object_normals)
                    object_points = self.__object.get_points()[0]

                    self.rotate(summary_vector, object_points[rotation_point_index][0], 1)
                    center = self.__object.get_center()
                    object_points = self.__object.get_points()[0]
                    more_indexes = self.potential_interceptions(center, passive_object_points,
                                                                object_vectors[0, 1])[:, 0].astype(int)
                    second_object_points, second_object_normals = self.find_positive_vectors(
                        (passive_object_points[more_indexes]), passive_object_normals[more_indexes] / 50, center)
                    if second_object_points.size == 0:
                        drown_points_indexes = np.empty(0)
                    else:
                        drown_points_indexes = self.find_drown_points(object_points, second_object_points,
                                                                      second_object_normals)

                    crutch +=1
                    if crutch > 300:
                        drown_points_indexes = np.empty(0)

                points_into_object_radius = self.potential_interceptions(center, passive_object_points,
                                                                         object_vectors[0, 1])
                second_object_points, second_object_normals = self.find_positive_vectors(
                    (passive_object_points[more_indexes]), passive_object_normals[more_indexes] / 50, center)
                if not points_into_object_radius.size == 0:
                    print("what")
                    vectors = np.empty((0, 2), float)
                    for index, distance in points_into_object_radius:
                        potential_vectors = self.potential_interceptions(passive_object_points[int(index)],
                                                                         object_points,
                                                                         min_radius)
                        if not potential_vectors.size == 0:
                            vectors = np.r_[vectors, potential_vectors]
                    #
                    if vectors.size == 0:
                        print("no intersections")
                    #                     self.visualize(stable_object=passive_object, center_point=center)
                    else:
                        normals = center - self.__object.get_normals()[vectors[:, 0].astype(int)] / 50
                        normal_vector = np.mean(normals, axis=0)
                        summary_vector = self.__gravity / 50 + normal_vector
                        self.__current_vector = (summary_vector - center) / np.linalg.norm(summary_vector - center)
                    if np.sqrt((summary_vector - center).dot(
                            summary_vector - center)) < self.__summary_vector_threshold:
                        print(np.sqrt((summary_vector - center).dot(summary_vector - center)))
                        print("object is stable")
                        object_is_stable = True
                    else:
                        print(np.sqrt((summary_vector - center).dot(summary_vector - center)))
                        print("time to move, baby!")

                    self.visualize(stable_object=passive_object, center_point=center,
                                   normal_numbers=vectors[:, 0].astype(int),
                                   movement_vector=self.__current_vector,
                                   points_form_center=[normal_vector, summary_vector],
                                   point_point_vectors_points=np.c_[
                                       passive_object_points, passive_object_points - passive_object_normals/50])

            # print("current_vector", self.__current_vector)
            self.__object.shift(self.__current_vector/100 + self.__gravity/100)
            center = self.__object.get_center()
            object_points = self.__object.get_points()[0]

    # def interaction(self, passive_object, min_radius=0.01):
    #
    #     object_is_stable = False
    #
    #     while not object_is_stable:
    #         center = self.__object.get_center()
    #         object_points = self.__object.get_points()[0]
    #         passive_object_points = passive_object.get_points()[0]
    #
    #         object_vectors = self.sort_object_points(center, object_points)
    #         points_into_object_radius = self.potential_interceptions(center, passive_object_points,
    #                                                                  object_vectors[0, 1])
    #
    #         if points_into_object_radius.size == 0:
    #             print("no interceptions")
    #             self.visualize(stable_object=passive_object, center_point=center)
    #         else:
    #             indexes = self.potential_interceptions(center, passive_object_points, object_vectors[0, 1])[:,
    #                       0].astype(int)
    #             passive_object_points, passive_object_normals = self.find_positive_vectors(
    #                 (passive_object.get_points()[0][indexes]), passive_object.get_normals()[indexes] / 50, center)
    #             positive_vectors = passive_object_normals - passive_object_points
    #
    #             self.visualize(stable_object=passive_object, center_point=center,
    #                            just_points=passive_object_points - positive_vectors)
    #
    #             drown_points_indexes = self.find_drown_points(object_points, passive_object_points, positive_vectors)
    #
    #
    #             if not drown_points_indexes.size == 0:
    #                 passive_object_points, passive_object_normals = self.find_positive_vectors(
    #                     (passive_object.get_points()[0]), passive_object.get_normals() / 50, center)
    #             while not drown_points_indexes.size == 0:
    #                 extreme_point = self.find_extreme_point(object_points[drown_points_indexes], summary_vector, center)
    #                 rotation_point_index = np.where(np.all(object_points == extreme_point, axis=1))[0]
    #
    #                 self.make_it_not_drown(object_points[rotation_point_index], passive_object_points,
    #                                        passive_object_normals)
    #                 object_points = self.__object.get_points()[0]
    #                 center = self.__object.get_center()
    #                 self.rotate(summary_vector, object_points[rotation_point_index][0], 0.1)
    #
    #                 drown_points_indexes = self.find_drown_points(object_points, passive_object_normals)
    #
    #             normals = center - self.__object.get_normals()[drown_points_indexes] / 50
    #             normal_vector = np.mean(normals, axis=0)
    #             summary_vector = self.__gravity / 50 + self.__current_vector + normal_vector
    #
    #             vectors = np.empty((0, 2), float)
    #             if not points_into_object_radius.size == 0:
    #
    #                 for index, distance in points_into_object_radius:
    #                     potential_vectors = self.potential_interceptions(passive_object_points[int(index)],
    #                                                                      object_points,
    #                                                                      min_radius)
    #                     if not potential_vectors.size == 0:
    #                         vectors = np.r_[vectors, potential_vectors]
    #
    #                 if vectors.size == 0:
    #                     print("no intersections")
    #                     self.visualize(stable_object=passive_object, center_point=center)
    #                 else:
    #                     normals = center - self.__object.get_normals()[vectors[:, 0].astype(int)] / 50
    #                     normal_vector = np.mean(normals, axis=0)
    #                     summary_vector = self.__gravity / 50 + self.__current_vector / 50 + normal_vector
    #                     self.__current_vector = (summary_vector - center) / np.linalg.norm(summary_vector - center)
    #                     if np.sqrt((summary_vector - center).dot(
    #                             summary_vector - center)) < self.__summary_vector_threshold:
    #                         print(np.sqrt((summary_vector - center).dot(summary_vector - center)))
    #                         print("object is stable")
    #                         object_is_stable = True
    #                     else:
    #                         print(np.sqrt((summary_vector - center).dot(summary_vector - center)))
    #                         print("time to move, baby!")
    #
    #             self.visualize(stable_object=passive_object, center_point=center,
    #                            normal_numbers=vectors[:, 0].astype(int),
    #                            movement_vector=self.__current_vector,
    #                            # points_form_center=[normal_vector, summary_vector, object_points[rotation_point_index][0]],
    #                            points_form_center=[normal_vector, summary_vector],
    #                            point_point_vectors_points=np.c_[passive_object_points, passive_object_normals],
    #                            just_points=object_points[drown_points_indexes])
    #
    #         self.__object.shift(self.__current_vector / 50 + self.__gravity / 50)

    def sort_object_points(self, center_point, points):
        nbrs = NearestNeighbors(n_neighbors=points.shape[0], algorithm='auto').fit(points)
        distances, indices = nbrs.kneighbors(center_point.reshape(1, -1))
        return np.flip(np.dstack((indices, distances)), 1)[0]

    def potential_interceptions(self, center_point, points, max_dist):
        nbrs = NearestNeighbors(n_neighbors=points.shape[0], algorithm='auto').fit(points)
        distances, indices = nbrs.kneighbors(center_point.reshape(1, -1))
        return np.dstack((indices[distances < max_dist], distances[distances < max_dist]))[0]

    def find_positive_vectors(self, points, normals, center_point):

        positive_vectors = np.zeros_like(normals)

        positive_vectors[:, 0] = np.where(
            np.linalg.norm((center_point - (normals + points)), axis=1) < np.linalg.norm(
                center_point - (- normals + + points), axis=1), normals[:, 0], -normals[:, 0])
        positive_vectors[:, 1] = np.where(
            np.linalg.norm((center_point - (normals + points)), axis=1) < np.linalg.norm(
                center_point - (- normals + points), axis=1), normals[:, 1], -normals[:, 1])
        positive_vectors[:, 2] = np.where(
            np.linalg.norm((center_point - (normals + points)), axis=1) < np.linalg.norm(
                center_point - (- normals + points), axis=1), normals[:, 2], -normals[:, 2])

        return points, positive_vectors

    def find_drown_points(self, points, object_points, object_normals):
        pos_nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(object_points + object_normals)
        neg_nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(object_points - object_normals)

        drown_points = np.where(pos_nbrs.kneighbors(points)[0] > neg_nbrs.kneighbors(points)[0],
                                points, np.nan)

        return (np.argwhere(~np.isnan(drown_points).any(axis=1))).flatten()

    def find_extreme_point(self, points_, vector, center, dimensions=[0, 2]):
        center = center[dimensions]
        points = points_[:, dimensions] - center
        vector = vector[dimensions] - center

        points_projection = np.zeros_like(points)
        temp = np.dot(points, vector) / np.dot(vector, vector)
        points_projection[:, 0] = vector[0] * temp
        points_projection[:, 1] = vector[1] * temp

        projection_lengths = np.zeros(points.shape[0])
        for p, p_p in enumerate(points_projection):
            projection_lengths[p] = np.sqrt(p_p.dot(p_p))
            if projection_lengths[p] < np.sqrt((p_p - vector).dot(p_p - vector)):
                projection_lengths[p] = 0
        return points_[np.argmax(projection_lengths)]

    def make_it_not_drown(self, point, surface_points, surface_normals):

        pos_nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(surface_points + surface_normals)
        neg_nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(surface_points - surface_normals)
        if point.size != 3:
            pnt = np.asarray([np.copy(point[0])])
        else:
            pnt = np.copy(point)
        crutch = 0
        while pos_nbrs.kneighbors(pnt.reshape(1, -1))[0][0, 0] > neg_nbrs.kneighbors(pnt.reshape(1, -1))[0][0, 0]:
            nearest_normal_index = [pos_nbrs.kneighbors(pnt.reshape(1, -1))[1][0, 0]]
            pnt += (surface_normals[nearest_normal_index] - surface_points[nearest_normal_index]) / 100
            crutch += 1
            if crutch > 50:
                self.__object.shift(pnt - np.asarray([point[0]]))
                print("there is a problem, boss")
                return
        self.__object.shift(pnt - np.asarray([point[0]]))

    def rotate(self, vector, rotation_point, angle=0.1, dimensions=[0, 2]):

        rotation_vector = np.flip(vector[dimensions])
        rotation_vector = rotation_vector / np.linalg.norm(rotation_vector)
        rotation_vector_3d = np.zeros(3)
        for i, d in enumerate(dimensions):
            rotation_vector_3d[d] = rotation_vector[i]
        rotation_vector_3d[dimensions[0]] *= -1

        self.__object.shift(-rotation_point)
        self.__object.rotate(rotation_vector_3d, math.radians(angle))
        self.__object.shift(rotation_point)

    def visualize(self, stable_object=None, center_point=None, normal_numbers=None, movement_vector=None,
                  points_form_center=None, point_point_vectors_points=None, just_points=None, visualize_object=True):
        import visualization

        if stable_object is not None:
            if visualize_object:
                objects = [stable_object, self.__object]
            else:
                objects = [stable_object]
        else:
            objects = None

        if center_point is not None:
            vector_points = []
            vector_lines = []
            vector_colors = []
            vector_points.append(center_point)
            imp_points = [center_point]
            imp_points_color = [[1, 0, 0]]
        else:
            vector_points = None
            vector_lines = None
            vector_colors = None
            imp_points = []
            imp_points_color = []

        if movement_vector is not None:
            vector_points.append(center_point + movement_vector / 50)
            vector_lines.append([0, len(vector_points) - 1])
            vector_colors.append([0, 255 / 150, 255 / 190])

        if normal_numbers is not None:
            normals = self.__object.get_normals()[normal_numbers]
            for normal in normals:
                vector_points.append(center_point - normal / 50)
                vector_lines.append([0, len(vector_points) - 1])
                vector_colors.append([0, 1, 0])

            points = (self.__object.get_points()[0])[normal_numbers]
            for point in points:
                vector_points.append(point)
                vector_lines.append([0, len(vector_points) - 1])
                vector_colors.append([1, 0, 0])

        if points_form_center is not None:
            for point in points_form_center:
                vector_points.append(point)
                vector_lines.append([0, len(vector_points) - 1])
                vector_colors.append([0, 0, 1])
        else:
            vector_points = []
            vector_lines = []
            vector_colors = []

        object_normals_lines = []
        object_normals_points = []
        object_normals_colors = []
        if point_point_vectors_points is not None:
            for i, row in enumerate(point_point_vectors_points):
                object_normals_lines.append([i * 2, i * 2 + 1])
                # object_normals_points.extend([row[:3], row[3:] / 50 + row[:3]])
                object_normals_points.extend([row[:3], row[3:]])
                object_normals_colors.append([1, 0, 0])
        else:
            point_point_vectors_points = []

        if just_points is not None:
            for p in just_points:
                imp_points.append(p)
                imp_points_color.append([1, 0, 0])

        visualization.visualize(objects=objects, points=imp_points, points_color=imp_points_color,
                                lines=[vector_lines, object_normals_lines],
                                lines_points=[vector_points, object_normals_points],
                                lines_color=[vector_colors, object_normals_colors])
