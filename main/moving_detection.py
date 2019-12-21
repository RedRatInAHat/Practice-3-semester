import numpy as np
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import List


@dataclass
class RGB:
    r: float
    g: float
    b: float


@dataclass
class Gaussian:
    index: List[int]
    mean: List[List[float]]
    luminance_variance: List[float]
    color_variance: List[float]
    depth_variance: List[float]
    weight: List[float]


class FrameDifference:
    """Class for finding moving objects by frame difference method

    Attributes:
        __current_depth (numpy array): current depth map
        __current_rgb (numpy array): current rgb image
        __previous_depth (numpy array): previous depth map
        __previous_rgb (numpy array): previous rgb image
        __mask (numpy array): mask, that displays the area of moving object
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

    @current_rgb.setter
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
    """Class for finding moving objects by Vision Background Extractor (ViBE)

    Attributes:
        __current rgb (numpy.array): current rgb image
        __previous rgb (numpy.array): previous rgb image
        __background (numpy.array): a model which contains its own rgb-value and values of neighbours
        __mask (numpy.array): mask, that displays the area of moving object
        __potential_neighbours (numpy.array): array which represents area of neighbour value searching
        __number_of_samples (int): number of values which every background pixel
        __threshold_r (float): threshold value of color vector in color space
        __threshold_lambda (int): threshold value for number of neighbours
        __time_factor (int): value representing probability
    """

    def __init__(self, rgb_im, number_of_samples=20, threshold_lambda=2, threshold_r=20 / 255, time_factor=16,
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
        # self.set_mask()

    def initial_background(self):
        """Filling background

        For every "pixel" in background, which contains __number_of_samples samples, a value of neighbour is choosing;
        the first value in samples is its own.
        """
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
        """Going through all pixels in mask"""
        for i in range(self.__current_rgb.shape[0]):
            for j in range(self.__current_rgb.shape[1]):
                self.set_pixel(i, j)

    def set_pixel(self, i, j):
        """Choosing status of pixel: background or foreground

        If pixel belongs to background pixel of mask is setting to 0. With probability 1/time_factor sample is updating,
        the same probability the neighbour of sample will be upgraded.

        Arguments:
            i, j (int): indexes of pixel
        """
        if self.in_background(i, j):
            self.__mask[i, j] = 0

            if time_factor_chance(self.__time_factor):
                self.update_sample(i, j, i, j)

            if time_factor_chance(self.__time_factor):
                neighbour_i = get_random_neighbour(i, self.__current_rgb.shape[0], self.__potential_neighbours)
                neighbour_j = get_random_neighbour(j, self.__current_rgb.shape[1], self.__potential_neighbours)
                self.update_sample(neighbour_i, neighbour_j, i, j)

        else:
            self.__mask[i, j] = 1

    def update_sample(self, goal_i, goal_j, set_i, set_j):
        """Updating sample.

        Random position of sample will be upgraded with a new value.

        Arguments:
            goal_i, goal_j (int): indexes of sample to change
            set_i, set_j (int): indexes of pixel which values will be assigned to sample
        """
        self.__background[goal_i, goal_j, random.randrange(self.__number_of_samples)] = \
            self.__current_rgb[set_i, set_j]

    def in_background(self, i, j):
        """Checking for belonging to background

        If color distance between current image and background samples is more than threshold_r, it is appointed to
        background, otherwise - foreground.

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

    @mask.setter
    def mask(self, mask):
        self.__mask = mask


# that part kills Python.exe for some reason
# class DEVB(ViBЕ):
#
#     def __init__(self, rgb_im, depth_im, number_of_samples=20, threshold_lambda=2, threshold_r=20 / 255,
#                  threshold_theta=3 / 255, time_factor=16, neighbourhood_area=4):
#
#         ViBЕ.__init__(self, rgb_im, number_of_samples, threshold_lambda, threshold_r, time_factor, neighbourhood_area)
#
#         self.__current_depth = depth_im
#         self.__threshold_theta = threshold_theta
#
#         self.__depth_background = depth_im
#
#         self.initial_background()
#
#     def set_pixel(self, i, j):
#
#         if self.in_background(i, j):
#
#             self.mask[i, j] = 0
#
#             if time_factor_chance(self.time_factor):
#                 self.update_sample(i, j, i, j)
#
#             if time_factor_chance(self.time_factor):
#                 neighbour_i = get_random_neighbour(i, self.current_rgb.shape[0], self.potential_neighbours)
#                 neighbour_j = get_random_neighbour(j, self.current_rgb.shape[1], self.potential_neighbours)
#                 self.update_sample(neighbour_i, neighbour_j, i, j)
#                 self.__depth_background[i, j] = self.__current_depth[i, j]
#         else:
#
#             self.mask[i, j] = 1
#             if color_distance(self.__depth_background, self.__current_depth) > self.__threshold_theta:
#                 self.mask[i, j] = 0
#
#                 if time_factor_chance(self.time_factor):
#                     self.update_sample(i, j, i, j)
#
#                 if time_factor_chance(self.time_factor):
#                     neighbour_i = get_random_neighbour(i, self.current_rgb.shape[0], self.potential_neighbours)
#                     neighbour_j = get_random_neighbour(j, self.current_rgb.shape[1], self.potential_neighbours)
#                     self.update_sample(neighbour_i, neighbour_j, i, j)
#                     self.__depth_background[i, j] = self.__current_depth[i, j]
#
#     def set_images(self, current_rgb, current_depth):
#         self.current_rgb = current_rgb
#         self.__current_depth = current_depth
#
#     @property
#     def current_depth(self):
#         return self.__current_depth
#
#     @current_depth.setter
#     def current_depth(self, current_depth):
#         self.__current_depth = current_depth

class DEVB:
    """Class for finding moving objects by Depth-Extended ViBe (DEVB)

    Attributes:
        __current rgb (numpy.array): current rgb image
        __current_depth (numpy.array): current depth image
        __previous rgb (numpy.array): previous rgb image
        __number_of_samples (int): number of values which every background pixel
        __threshold_theta (float): threshold value of depth vector in depth space
        __threshold_r (float): threshold value of color vector in color space
        __threshold_lambda (int): threshold value for number of neighbours
        __time_factor (int): value representing probability
        __background (numpy.array): a model which contains its own rgb-value and values of neighbours
        __depth_background (numpy.array): a model of background in depth format
        __mask (numpy.array): mask, that displays the area of moving object
        __potential_neighbours (numpy.array): array which represents area of neighbour value searching
    """

    def __init__(self, rgb_im, depth_im, number_of_samples=20, threshold_lambda=2, threshold_r=20 / 255,
                 threshold_theta=3 / 255, time_factor=16, neighbourhood_area=4):

        self.__current_rgb = rgb_im
        self.__current_depth = depth_im
        self.__previous_rgb = np.empty_like(rgb_im)

        self.__number_of_samples = number_of_samples
        self.__threshold_theta = threshold_theta
        self.__threshold_r = threshold_r
        self.__threshold_lambda = threshold_lambda
        self.__time_factor = time_factor

        self.__background = np.empty([rgb_im.shape[0], rgb_im.shape[1], number_of_samples, 3])
        self.__depth_background = depth_im
        self.__mask = np.empty([rgb_im.shape[0], rgb_im.shape[1]])
        self.__potential_neighbours = np.arange(-neighbourhood_area, neighbourhood_area + 1)
        self.__potential_neighbours = self.__potential_neighbours[self.__potential_neighbours != 0]

        self.initial_background()

    def initial_background(self):
        """Filling background

        For every "pixel" in background, which contains __number_of_samples samples, a value of neighbour is choosing;
        the first value in samples is its own.
        """
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
        """Going through all pixels in mask"""
        for i in range(self.__current_rgb.shape[0]):
            for j in range(self.__current_rgb.shape[1]):
                self.set_pixel(i, j)

    def set_pixel(self, i, j):
        """Choosing status of pixel: background or foreground

        If pixel belongs to background pixel of mask is setting to 0. With probability 1/time_factor sample is updating,
        the same probability the neighbour of sample and depth_background will be upgraded.
        Otherwise it is checking for depth_background matching. If pixel became closer for threshold_theta value or more
        it is setting to 1; it is updating almost the same as for previous if, but depth is updating with 1 chance. Else
        it is set to 0.

        Arguments:
            i, j (int): indexes of pixel
        """
        if self.in_background(i, j):

            self.__mask[i, j] = 0

            if time_factor_chance(self.__time_factor):
                self.update_sample(i, j, i, j)

            if time_factor_chance(self.__time_factor):
                neighbour_i = get_random_neighbour(i, self.__current_rgb.shape[0], self.__potential_neighbours)
                neighbour_j = get_random_neighbour(j, self.__current_rgb.shape[1], self.__potential_neighbours)
                self.update_sample(neighbour_i, neighbour_j, i, j)
                self.__depth_background[i, j] = self.__current_depth[i, j]
        else:
            if self.__depth_background[i, j] - self.__current_depth[i, j] > self.__threshold_theta:
                self.__mask[i, j] = 1

                if time_factor_chance(self.__time_factor):
                    self.update_sample(i, j, i, j)

                if time_factor_chance(self.__time_factor):
                    neighbour_i = get_random_neighbour(i, self.__current_rgb.shape[0], self.__potential_neighbours)
                    neighbour_j = get_random_neighbour(j, self.__current_rgb.shape[1], self.__potential_neighbours)
                    self.update_sample(neighbour_i, neighbour_j, i, j)
                self.__depth_background[i, j] = self.__current_depth[i, j]
            else:
                self.__mask[i, j] = 0

    def update_sample(self, goal_i, goal_j, set_i, set_j):
        """Updating sample.

        Random position of sample will be upgraded with a new value.

        Arguments:
            goal_i, goal_j (int): indexes of sample to change
            set_i, set_j (int): indexes of pixel which values will be assigned to sample
        """
        self.__background[goal_i, goal_j, random.randrange(self.__number_of_samples)] = \
            self.__current_rgb[set_i, set_j]

    def in_background(self, i, j):
        """Checking for belonging to background

        If color distance between current image and background samples is more than threshold_r, it is appointed to
        background, otherwise - foreground.

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

    def set_images(self, current_rgb, current_depth):
        self.__previous_rgb = self.__current_rgb
        self.__current_rgb = current_rgb
        self.__current_depth = current_depth

    @property
    def mask(self):
        return self.__mask

    @mask.setter
    def mask(self, mask):
        self.__mask = mask


class MoG:

    def __init__(self, rgb_im, depth_im, number_of_gaussians=3, learning_rate=.025):

        self.__number_of_gaussians = number_of_gaussians
        self.__learning_rate = learning_rate

        self.__current_yuv = self.RGB_to_YUV(rgb_im * 255)
        self.__current_rgb = rgb_im * 255
        self.__gaussians = []
        self.__mask = np.zeros_like(depth_im)

        self.initialization()

    def RGB_to_YUV(self, rgb):
        """
        T-REC-T.871 recommendation
        code from https://gist.github.com/Quasimondo/c3590226c924a06b276d606f4f189639
        """

        # for row in self.__current_rgb:
        #     for column in row:
        #         r, g, b = column[0], column[1], column[2]
        #         y = 0.299*r + 0.587*g+0.114*b
        #         c_b = 0.5*(b-y)/0.886 + 128
        #         c_r = 0.5*(r-y)/0.701 + 128

        m = np.array([[0.29900, -0.16874, 0.50000],
                      [0.58700, -0.33126, -0.41869],
                      [0.11400, 0.50000, -0.08131]])

        yuv = np.dot(rgb, m)
        yuv[:, :, 1:] += 128.0
        return yuv

    def initialization(self):
        for i in range(self.__current_yuv.shape[0]):
            for j in range(self.__current_yuv.shape[1]):
                self.__gaussians.append(
                    Gaussian([i, j], [[100, 100, 100]] * self.__number_of_gaussians, [36] * self.__number_of_gaussians,
                             [36] * self.__number_of_gaussians, [36] * self.__number_of_gaussians,
                             [1 / self.__number_of_gaussians] * self.__number_of_gaussians))
        print(self.__gaussians)


def color_distance(current_pixel, sample_pixel):
    """Calculation of distance between colors in rgb space

    Arguments:
        current_pixel(np.array): RGB of first pixel
        sample_pixel(np.array): RGB of second pixel
    """
    difference = current_pixel - sample_pixel
    dist = 0
    for i in range(current_pixel.shape[0]):
        dist += difference[i] * difference[i]
    return dist


def time_factor_chance(chance_factor):
    """With 1/chance_factor returns True; otherwise false"""
    return np.random.choice([True, False], 1, p=[1 / chance_factor, 1 - 1 / chance_factor])


def get_random_neighbour(index, resolution, area):
    """Get random index in neighbourhood area

    If index is not less or more then picture resolution, choose random index in area.

    Arguments:
        index (int): index which neighbour is chosen for
        resolution (int): resolution of image
        area (numpy.array): array which represents the area in which the neighbour will be chosen
    """
    neighbour_index = index + np.random.choice(area)
    while neighbour_index < 0 or neighbour_index >= resolution:
        neighbour_index = index + np.random.choice(area)
    return neighbour_index
