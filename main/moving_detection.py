import numpy as np
import math
import random
from collections import deque
from dataclasses import dataclass


@dataclass
class RGB:
    r: float
    g: float
    b: float


class FrameDifference:
    """Class for finding moving objects by frame difference method

    Attributes:
        current_depth (numpy array): current depth map
        current_rgb (numpy array): current rgb image
        previous_depth (numpy array): previous depth map
        previous_rgb (numpy array): previous rgb image
        mask (numpy array): mask, that displays the area of moving object
    """

    def __init__(self, depth_im, rgb_im, rgb_threshold=0.3, depth_threshold=0.1):
        self.__current_depth = depth_im
        self.__current_rgb = rgb_im
        self.__previous_depth = np.empty_like(self.__current_depth)
        self.__previous_rgb = np.empty_like(self.__current_rgb)
        self.__mask = []
        self.__rgb_threshold = rgb_threshold
        self.__depth_threshold = depth_threshold

    @property
    def current_depth(self):
        return self.__current_depth

    @current_depth.setter
    def current_depth(self, current_depth):
        self.__previous_depth = np.copy(self.__current_depth)
        self.__current_depth = current_depth

    @property
    def current_rgb(self):
        return self.__current_rgb

    @current_depth.setter
    def current_rgb(self, current_rgb):
        self.__previous_rgb = np.copy(self.__current_rgb)
        self.__current_rgb = current_rgb

    def subtraction_mask(self):
        """Creating a mask for detecting changed pixels

        First step is subtraction rgb images for detecting color changes. If difference is more than the threshold
        value, pixel is checking for a depth changing. If it is more, than zero, it is a moving part.

        Return:
            mask (np.array): mask of image, where 0 is for standing and 1 is for moving
        """
        rgb_subtraction = self.__current_rgb - self.__previous_rgb

        mask = np.empty_like(self.__current_depth)
        for i in range(rgb_subtraction.shape[0]):
            for j in range(rgb_subtraction.shape[1]):
                if rgb_subtraction[i, j, 0] * rgb_subtraction[i, j, 0] + rgb_subtraction[i, j, 1] * rgb_subtraction[
                    i, j, 1] + rgb_subtraction[i, j, 2] * rgb_subtraction[i, j, 2] > self.__rgb_threshold and \
                        self.__previous_depth[i, j] - self.__current_depth[i, j] > 0:
                    mask[i, j] = 1
                else:
                    mask[i, j] = 0
        return mask

    def region_growing(self, movement_mask):
        """Region growing realization

        Pixel, that was checked as "moving", with the largest value is choosing as seed. From that seed performing a
        region growing - if value of neighbour pixels is close to the seed value they are adding to the region and the
        next growing will be performing from these values.
        Algorithm works until there are seeds.

        Arguments:
            movement_mask (np.array): a mask for seeds
        Return:
            mask (np.array): a mask of moving objects
        """

        seeds = movement_mask * self.__current_depth
        done = np.zeros_like(seeds)

        while not np.sum(seeds) == 0:
            ind_i = math.floor(np.argmax(seeds) / self.__current_rgb.shape[1])
            ind_j = np.argmax(seeds) - ind_i * self.__current_rgb.shape[1]
            mask = np.zeros_like(self.__current_depth)

            q = deque()

            q.append([ind_i, ind_j])
            seeds[ind_i, ind_j] = 0
            done[ind_i, ind_j] = 1
            mask[ind_i, ind_j] = 1

            while q:
                ind = q.popleft()
                for i in range(ind[0] - 1, ind[0] + 2):
                    for j in range(ind[1] - 1, ind[1] + 2):
                        if -1 < i < self.__current_depth.shape[0] and -1 < j < self.__current_depth.shape[1]:
                            if not done[i, j] == 1 and not self.__current_depth[i, j] > 1:
                                if math.fabs(self.__current_depth[i, j] - self.current_depth[
                                    ind[0], ind[1]]) < self.__depth_threshold:
                                    q.append([i, j])
                                    seeds[i, j] = 0
                                    done[i, j] = 1
                                    mask[i, j] = 1
            if np.sum(mask) > 100:
                self.__mask.append(mask)
        return self.__mask


class ViBЕ:

    def __init__(self, rgb_im, number_of_samples=20, threshold_lambda=2, threshold_r=20, time_factor=16,
                 neighbourhood_area=4):
        self.__current_rgb = rgb_im
        self.__previous_rgb = np.empty_like(rgb_im)
        self.__background = np.empty([rgb_im.shape[0], rgb_im.shape[1], number_of_samples, 3])
        self.__mask = np.empty([rgb_im.shape[0], rgb_im.shape[1]])
        self.__potential_neighbours = np.arange(-neighbourhood_area, neighbourhood_area + 1)
        self.__potential_neighbours = self.__potential_neighbours[self.__potential_neighbours != 0]
        self.__number_of_samples = number_of_samples
        self.__threshold_r = threshold_r
        self.__threshold_lambda = threshold_lambda
        self.__time_factor = time_factor
        self.initial_background()
        self.set_mask()

    def initial_background(self):
        resolution_i = self.__current_rgb.shape[0]
        resolution_j = self.__current_rgb.shape[1]
        self.__previous_rgb = np.copy(self.__current_rgb)
        for i in range(self.__current_rgb.shape[0]):
            for j in range(self.__previous_rgb.shape[1]):
                self.__background[i, j, 0] = self.__current_rgb[i, j]
                for k in range(1, self.__number_of_samples):
                    rand_i = get_random_neighbour(i, resolution_i, self.__potential_neighbours)
                    rand_j = get_random_neighbour(j, resolution_j, self.__potential_neighbours)
                    self.__background[i, j, k] = self.__current_rgb[rand_i, rand_j]

    def set_mask(self):
        for i in range(self.__current_rgb.shape[0]):
            for j in range(self.__current_rgb.shape[1]):
                self.set_pixel(i, j)

    def set_pixel(self, i, j):

        if self.in_background(i, j):
            self.__mask[i, j] = 0

            if self.time_factor_chance(self.__time_factor):
                self.update_sample(i, j, i, j)

            if self.time_factor_chance(self.__time_factor):
                neighbour_i = get_random_neighbour(i, self.__current_rgb.shape[0], self.__potential_neighbours)
                neighbour_j = get_random_neighbour(j, self.__current_rgb.shape[1], self.__potential_neighbours)
                self.update_sample(neighbour_i, neighbour_j, i, j)
            elif self.time_factor_chance(self.__time_factor):
                area_radius = 2
                area = np.arange(-area_radius, area_radius + 1)
                area = area[area != 0]

                neighbour_i = get_random_neighbour(i, self.__current_rgb.shape[0], area)
                neighbour_j = get_random_neighbour(j, self.__current_rgb.shape[1], area)

                self.update_sample(neighbour_i, neighbour_j, i, j)

            return 0
        else:
            self.__mask[i, j] = 1
            if self.time_factor_chance(self.__time_factor) and \
                    color_distance(self.__current_rgb[i, j], self.__previous_rgb[i, j]) < self.__threshold_r and \
                    self.no_foreground_neighbours(i, j):
                self.update_sample(i, j, i, j)
            return 1

    def no_foreground_neighbours(self, i, j):
        if not i - 1 < 0 and not self.__mask[i - 1, j] == 1:
            return True
        if not i + 1 >= self.__current_rgb.shape[0] and not self.__mask[i + 1, j] == 1:
            return True
        if not j - 1 < 0 and not self.__mask[i, j - 1] == 1:
            return True
        if not j + 1 >= self.__current_rgb.shape[1] and not self.__mask[i, j + 1] == 1:
            return True
        return False

    def update_sample(self, goal_i, goal_j, set_i, set_j):
        self.__background[goal_i, goal_j, random.randrange(self.__number_of_samples)] = \
            self.__current_rgb[set_i, set_j]

    def time_factor_chance(self, chance_factor):
        return np.random.choice([True, False], 1, p=[1 / chance_factor, 1 - 1 / chance_factor])

    def in_background(self, i, j):
        """Checking for belonging to background

        Arguments:
            i, j (int): indexes of current pixel

        Return:
            bool: true if belongs to background, else false
        """

        close_pixels = 0

        for k in range(self.__number_of_samples):

            if color_distance(self.__current_rgb[i, j], self.__background[i, j, k]) < self.__threshold_r:
                close_pixels += 1

            if close_pixels >= self.__threshold_lambda:
                return True

        return False

    @property
    def current_rgb(self):
        return self.__current_rgb

    @current_rgb.setter
    def current_rgb(self, current_rgb):
        self.__previous_rgb = np.copy(self.__current_rgb)
        self.__current_rgb = current_rgb

    @property
    def mask(self):
        return self.__mask


def color_distance(current_pixel, sample_pixel):
    """Calculation of distance between colors in rgb space

    Arguments:
        current_pixel(np.array): RGB of first pixel
        sample_pixel(np.array): RGB of second pixel
    """
    difference = current_pixel - sample_pixel
    return np.sum(difference ** 2)


def get_random_neighbour(index, resolution, area):
    neighbour_index = index + np.random.choice(area)
    while neighbour_index < 0 or neighbour_index >= resolution:
        neighbour_index = index + np.random.choice(area)
    return neighbour_index
