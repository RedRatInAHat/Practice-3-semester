import numpy as np
import math
import random
from collections import deque
from dataclasses import dataclass
import warnings


@dataclass
class RGB:
    r: float
    g: float
    b: float


@dataclass
class Gaussian:
    index: np.ndarray = np.asarray([])
    luminance_mean: np.ndarray = np.asarray([])
    color_mean: np.ndarray = np.asarray([])
    depth_mean: np.ndarray = np.asarray([])
    luminance_variance: np.ndarray = np.asarray([])
    color_variance: np.ndarray = np.asarray([])
    depth_variance: np.ndarray = np.asarray([])
    weight: np.ndarray = np.asarray([])
    ranking: np.ndarray = np.asarray([])
    depth_observations: np.ndarray = np.asarray([])


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


class RGB_MoG:
    """Class for finding moving objects by Mixture of Gaussians with RGB image

    Attributes:
        __number_of_gaussians (int): number of gaussians for each color channel in each pixel
        __learning_rate_alfa (float): coefficient representing speed of changing of gaussians
        __height (int): height of image
        __width (int): width of image
        __number_of_channels (int): number of color channels of image
        __current_rgb (np.ndarray): current rgb image
        __gaussians (list): list with Gaussian() objects, describing gaussians of pixel
        __mask (np.ndarray): mask, that displays the area of moving object
    """

    def __init__(self, rgb_im, number_of_gaussians=2, learning_rate_alfa=0.025):
        self.__number_of_gaussians = number_of_gaussians
        self.__learning_rate_alfa = learning_rate_alfa

        self.__height = rgb_im.shape[0]
        self.__width = rgb_im.shape[1]

        try:
            self.__number_of_channels = rgb_im.shape[2]
        except:
            self.__number_of_channels = 1

        self.__current_rgb = rgb_im
        self.__gaussians = []
        self.__mask = np.zeros([self.__height, self.__width])

        self.initialization()

    def initialization(self):
        """Creating of gaussians

        For each channel of the pixel mean is a color of image; variances are ones; weights are 1/number_of_gaussians;
        ranking is a sequence.
        """
        variance = np.ones([self.__number_of_gaussians])
        weight = np.ones([self.__number_of_gaussians]) / self.__number_of_gaussians
        initial_ranking = np.arange(self.__number_of_gaussians)

        for h in range(self.__height):
            for w in range(self.__width):
                mean = np.zeros([self.__number_of_gaussians, self.__number_of_channels]) + self.__current_rgb[h, w]
                self.__gaussians.append(
                    Gaussian(index=np.asarray([h, w]), color_mean=mean, color_variance=variance, weight=weight,
                             ranking=initial_ranking))

    def set_raking(self, gauss):
        """Sets rank

        Calculating rating for each gaussian

        Arguments:
            gauss (Gaussian): gaussians of current pixel
        """
        gauss.ranking = np.argsort(-gauss.weight / gauss.color_variance)

    def probability(self, gauss):
        """Gaussian probability density

        Calculating probability of each gaussian and their matching (possibility of matching current color to gaussian.

        Arguments:
            gauss (Gaussian): gaussians of current pixel
        """
        dist_square = np.sum((self.__current_rgb[gauss.index[0], gauss.index[1]] - gauss.color_mean) ** 2,
                             axis=1) / gauss.color_variance
        dist = np.sqrt(dist_square)

        probability = np.exp(dist_square / (-2)) / (np.sqrt((2 * np.pi) ** 3) * gauss.color_variance)

        matching_criterion = (dist < 2.5 * gauss.color_variance)
        return matching_criterion, probability

    def update(self, gauss, matching_criterion, probability):
        """Updating gaussian parameters

        Updating of mean, variance and weight of gaussian if color matches to them; else mean is current color, variance
        is a list of 4 and weight is becoming less.

        Arguments:
            gauss (Gaussian): gaussians of current pixel
            matching_criterion (np.ndarray): array of matching colors to gaussians
            probability (np.ndarray): array of probability densities for every gaussian
        """

        learning_rate_ro = self.__learning_rate_alfa * probability

        update_status = np.bitwise_or.reduce(matching_criterion, axis=0)
        update_mask = np.where(update_status, matching_criterion, -1)

        gauss.weight = np.where(update_mask == 1,
                                (1 - self.__learning_rate_alfa) * gauss.weight + self.__learning_rate_alfa,
                                (1 - self.__learning_rate_alfa) * gauss.weight)

        gauss.color_variance = np.where(update_mask == 1, np.sqrt(
            (1 - learning_rate_ro) * (gauss.color_variance ** 2) + learning_rate_ro * (
                np.sum(np.subtract(self.__current_rgb[gauss.index[0], gauss.index[1]], gauss.color_mean) ** 2))),
                                        3 + np.ones(self.__number_of_gaussians))

        for i in range(self.__number_of_gaussians):
            if update_mask[i] == 1:
                gauss.color_mean[i] = (1 - learning_rate_ro[i]) * gauss.color_mean[i] + learning_rate_ro[i] * \
                                      self.__current_rgb[gauss.index[0], gauss.index[1]]
            else:
                gauss.color_mean[i] = self.__current_rgb[gauss.index[0], gauss.index[1]]

    def make_mask(self, matching_criterion):
        """Setting a pixel for mask

        Set pixel white if pixel is for foreground

        Arguments:
            matching_criterion (np.ndarray): mask for moving objects
        """
        matching = np.bitwise_or.reduce(matching_criterion, axis=0)
        res = np.where(matching, 0, 255)

        return res

    def set_mask(self, rgb_im):
        """Making mask of moving object

        For each pixel of rgb image calculate if it foreground or background

        Arguments:
            rgb_im (np.ndarray): array of image
        """
        self.__current_rgb = rgb_im
        for gauss in self.__gaussians:
            self.set_raking(gauss)
            matching_criterion, probability = self.probability(gauss)
            self.update(gauss, matching_criterion, probability)
            self.__mask[gauss.index[0], gauss.index[1]] = self.make_mask(matching_criterion)
        return self.__mask


class RGBD_MoG:

    def __init__(self, rgb_im, depth_im, number_of_gaussians=3, learning_rate_alfa=.025, depth_reliability_ro=0.2,
                 matching_rate_beta=2.5, luminance_min=16, depth_threshold=0.01, reliability_threshold=.4):

        self.__number_of_gaussians = number_of_gaussians
        self.__learning_rate_alfa = learning_rate_alfa
        self.__depth_reliability_ro = depth_reliability_ro
        self.__matching_rate_beta = matching_rate_beta
        self.__luminance_min = luminance_min
        self.__depth_threshold = depth_threshold
        self.__reliability_threshold = reliability_threshold

        self.__height = rgb_im.shape[0]
        self.__width = rgb_im.shape[1]
        try:
            self.__number_of_channels = rgb_im.shape[2] + 1
        except:
            print("There must be RGB image, not GreyScale")

        self.__current_yuv = self.RGB_to_YUV(rgb_im)
        self.__current_depth = depth_im
        self.__gaussians = []
        self.__mask = np.zeros_like(depth_im)

        self.initialization()

    def RGB_to_YUV(self, rgb):
        """
        T-REC-T.871 recommendation
        code from https://gist.github.com/Quasimondo/c3590226c924a06b276d606f4f189639
        """
        m = np.array([[0.29900, -0.16874, 0.50000],
                      [0.58700, -0.33126, -0.41869],
                      [0.11400, 0.50000, -0.08131]])

        yuv = np.dot(rgb, m)
        yuv[:, :, 1:] += 128.0
        return yuv

    def initialization(self):

        variance = np.ones([self.__number_of_gaussians])
        weight = np.ones([self.__number_of_gaussians]) / self.__number_of_gaussians
        initial_ranking = np.arange(self.__number_of_gaussians)

        for h in range(self.__height):
            for w in range(self.__width):
                luminance_mean, color_mean, depth_mean = np.zeros([self.__number_of_gaussians]) + self.__current_yuv[
                    h, w, 0], np.zeros([self.__number_of_gaussians, 2]) + self.__current_yuv[h, w, 1:], np.zeros(
                    [self.__number_of_gaussians]) + self.__current_depth[h, w]
                depth_observations = np.where(self.__current_depth[h, w] == 255, np.asarray([[0, 1]] * 3),
                                              np.asarray([[1, 1]] * 3))
                self.__gaussians.append(
                    Gaussian(index=np.asarray([h, w]),
                             luminance_mean=luminance_mean, color_mean=color_mean, depth_mean=depth_mean,
                             luminance_variance=np.copy(variance), color_variance=np.copy(variance),
                             depth_variance=np.copy(variance),
                             weight=np.copy(weight), ranking=np.copy(initial_ranking),
                             depth_observations=depth_observations))

    def set_ranking(self, gauss):
        gauss.ranking = np.argsort(-gauss.weight / gauss.luminance_variance)

    def matching(self, gauss):
        yuv_pixel = self.__current_yuv[gauss.index[0], gauss.index[1]]
        depth_pixel = self.__current_depth[gauss.index[0], gauss.index[1]]
        beta = self.__matching_rate_beta ** 2

        depth_matching = np.bitwise_or((depth_pixel - gauss.depth_mean) ** 2 < beta * gauss.depth_variance,
                                       np.bitwise_or(depth_pixel == 255, np.bitwise_not(
                                           gauss.depth_observations[:, 0] / gauss.depth_observations[:,
                                                                            1] > self.__depth_reliability_ro)))

        # for i in range(self.__number_of_gaussians):
        #     if (depth_pixel - gauss.depth_mean[i]) ** 2 > 0:
        #         print(depth_pixel, gauss.depth_mean[i], depth_pixel - gauss.depth_mean[i], depth_pixel - gauss.depth_mean)
        # depth_matching = (depth_pixel - gauss.depth_mean) ** 2 > beta/255 * gauss.depth_variance

        color_condition_1 = np.bitwise_and(
            gauss.luminance_mean > self.__luminance_min, yuv_pixel[0] > self.__luminance_min)
        color_condition_2 = (yuv_pixel[0] - gauss.luminance_mean) ** 2 < beta * gauss.luminance_variance
        color_condition_3 = np.sum((yuv_pixel[1:] - gauss.color_mean), axis=1) ** 2 < beta * gauss.color_variance
        color_matching = np.where(color_condition_1, np.add(color_condition_2, color_condition_3),
                                  color_condition_2)

        matching_criterion = np.bitwise_and(depth_matching, color_matching)

        return matching_criterion

    def update(self, gauss, matching_criterion, number_of_observations):
        yuv_pixel = self.__current_yuv[gauss.index[0], gauss.index[1]]
        depth_pixel = self.__current_depth[gauss.index[0], gauss.index[1]]

        if np.bitwise_or.reduce(matching_criterion):

            gauss.luminance_variance = np.where(matching_criterion, (
                    1 - self.__learning_rate_alfa) * gauss.luminance_variance + self.__learning_rate_alfa * (
                                                        yuv_pixel[0] - gauss.luminance_mean) ** 2,
                                                gauss.luminance_variance)

            gauss.luminance_mean = np.where(matching_criterion,
                                            (1 - self.__learning_rate_alfa) * gauss.luminance_mean +
                                            self.__learning_rate_alfa * yuv_pixel[0], gauss.luminance_mean)

            for i in range(self.__number_of_gaussians):
                if matching_criterion[i]:
                    gauss.color_variance[i] = (1 - self.__learning_rate_alfa) * gauss.color_variance[
                        i] + self.__learning_rate_alfa * np.sum((yuv_pixel[1:] - gauss.color_mean[i]) ** 2)
                    gauss.color_mean[i] = (1 - self.__learning_rate_alfa) * gauss.color_mean[
                        i] + self.__learning_rate_alfa * yuv_pixel[1:]
                    gauss.depth_observations[i] = [(1 - self.__learning_rate_alfa) * gauss.depth_observations[i, 0] +
                                                   self.__learning_rate_alfa * (depth_pixel < 255),
                                                   number_of_observations]

            gauss.depth_variance = np.where(matching_criterion, (
                    1 - self.__learning_rate_alfa) * gauss.depth_variance + self.__learning_rate_alfa * (
                                                    depth_pixel - gauss.depth_mean) ** 2, gauss.depth_variance)
            gauss.depth_mean = np.where(matching_criterion,
                                        (1 - self.__learning_rate_alfa) * gauss.depth_mean +
                                        self.__learning_rate_alfa * depth_pixel, gauss.depth_mean)

            gauss.weight = np.where(matching_criterion, (
                    1 - self.__learning_rate_alfa) * gauss.weight + self.__learning_rate_alfa * matching_criterion,
                                    gauss.weight)
        else:
            index = gauss.ranking[-1]

            gauss.luminance_mean[index] = yuv_pixel[0]
            gauss.color_mean[index] = yuv_pixel[1:]
            gauss.depth_mean[index] = depth_pixel

            gauss.luminance_variance[index] = 1
            gauss.color_variance[index] = 1
            gauss.depth_variance[index] = 1

            gauss.weight[index] = self.__learning_rate_alfa

            gauss.depth_observations[index] = [self.__learning_rate_alfa * (depth_pixel < 255), number_of_observations]

    def pixel_mask(self, gauss, matching_criterion):

        max_depth = 0
        depth_reliability = np.zeros(self.__number_of_gaussians)
        for index in np.argsort(-gauss.depth_mean):
            if self.__current_depth[gauss.index[0], gauss.index[1]] == 255:
                depth_reliability = np.ones(self.__number_of_gaussians)
                break
            elif gauss.depth_observations[index, 0] / gauss.depth_observations[index, 1] > \
                    self.__depth_reliability_ro or gauss.weight[index] > self.__depth_threshold:
                if gauss.depth_mean[index] >= max_depth:
                    depth_reliability[index] = 1
                    max_depth = gauss.depth_mean[index]
        depth_reliability = depth_reliability.astype(int)

        threshold = 0
        color_reliability = np.zeros(self.__number_of_gaussians)
        for index in gauss.ranking:
            while threshold < self.__reliability_threshold:
                threshold += gauss.weight[index]
                color_reliability[index] = 1
        color_reliability = color_reliability.astype(int)

        background = np.bitwise_or.reduce(np.bitwise_and(matching_criterion, color_reliability))
        background = np.bitwise_or.reduce(np.bitwise_or(background, np.bitwise_and.reduce(depth_reliability)))
        if background:
            self.__mask[gauss.index[0], gauss.index[1]] = 0
        else:
            self.__mask[gauss.index[0], gauss.index[1]] = 255

    def set_mask(self, rgb_im, depth_im):
        self.__current_yuv = self.RGB_to_YUV(rgb_im)
        self.__current_depth = depth_im
        number_of_observations = self.__gaussians[0].depth_observations[0, 1] + 1

        for gauss in self.__gaussians:
            self.set_ranking(gauss)
            matching_criterion = self.matching(gauss)
            self.update(gauss, matching_criterion, number_of_observations)
            self.pixel_mask(gauss, matching_criterion)

        return self.__mask


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
