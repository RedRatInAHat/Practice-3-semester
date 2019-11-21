import numpy as np
import math


class PointsObject:
    """Point clouds type objects

    Attributes:
        xyz (numpy.array): an array for xyz coordinates of the object
        rgb (numpy.array): an array for rgb value (.0, 1.0) of the points of the object
        visible (bool): shows should the object be using or not
        moving (bool): shows should the object move
    """

    def __init__(self):
        self.__xyz = np.empty([0, 3])
        self.__rgb = np.empty([0, 3])
        self.__visible = True
        self._moving = False

    def add_points(self, xyz, rgb=None):
        """Adding points to xyz and rgb

        If user didn't send rgb, it is made grey by default

        Args:
            xyz (numpy.array): an array for xyz coordinates of the object to add
            rgb(numpy.array): an array for rgb value (.0, 1.0) of the points of the object to add
        """
        try:
            self.__xyz = np.append(self.__xyz, xyz, axis=0)
            if rgb is None or not rgb.shape[0] == xyz.shape[0]:
                rgb = np.empty([xyz.shape[0], 3])
                rgb.fill(0.5)
            self.__rgb = np.append(self.__rgb, rgb, axis=0)
        except ValueError as e:
            print("Error in PointObject.add_points:", e)

    def set_points(self, xyz, rgb=None):
        """Setting xyz and rgb points

        If user didn't send rgb, it is made grey by default

        Args:
            xyz (numpy.array): an array for xyz coordinates of the object
            rgb(numpy.array): an array for rgb value (.0, 1.0) of the points of the object
        """
        try:
            self.__xyz = xyz
            if rgb is None or not rgb.shape[0] == xyz.shape[0]:
                rgb = np.empty([xyz.shape[0], 3])
                rgb.fill(0.5)
            self.__rgb = rgb
        except ValueError as e:
            print("Error in PointObject.set_points:", e)

    def get_points(self):
        """Returns coordinates and colors of object's points"""
        return self.__xyz, self.__rgb

    @property
    def visible(self):
        return self.__visible

    @visible.setter
    def visible(self, visible):
        self.__visible = visible

    @property
    def moving(self):
        return self.__moving

    @moving.setter
    def moving(self, moving):
        self.__moving = moving

    def rotate(self, axis, angle):
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

        A = np.zeros((self.__xyz.shape[0], self.__xyz.shape[1] + 1))
        A[:, :-1] = self.__xyz[:, :]

        A = np.dot(R, A.T).T

        self.__xyz = A[:, :-1]

    def shift(self, distance):
        """Linear moving of points of the object

        Arguments:
            distance (numpy.array): distance in xyz format according to which points must be moved
        """
        for i in range(self.__xyz.shape[0]):
            self.__xyz[i] = self.__xyz[i] + distance


if __name__ == "__main__":
    test = PointsObject()
    test.set_points(np.zeros([2, 3]))
    test.shift(np.array([2, 3, 1]))
    print("here: ", test.get_points())
    test.rotate([0, 0, 1], math.pi)
    print(test.get_points())