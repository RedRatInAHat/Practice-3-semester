import numpy as np
import math
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
            ind_i = math.floor(np.argmax(seeds) / 640)
            ind_j = np.argmax(seeds) - ind_i * 640
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

    def __init__(self, rgb_im, number_of_samples=20, threshold_lambda=2, threshold_r=20, time_factor=16):
        self.__current_rgb = rgb_im
        self.__previous_rgb = np.empty_like(rgb_im)
        self.__background = np.empty(rgb_im.shape[0], rgb_im.shape[1], number_of_samples)
        self.__mask = np.empty_like(rgb_im)
        self.initial_background()

    def initial_background(self):
        self.__previous_rgb = np.copy(self.__current_rgb)
        for i in range(self.__current_rgb.shape[0]):
            for j in range(self.__previous_rgb.shape[1]):
                pass

    def get_random_neighbour(self, i, j):
        pass
