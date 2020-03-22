import numpy as np
import math


class PointsObject:
    """Point clouds type objects

    Attributes:
        xyz (numpy.array): an array for active xyz coordinates of the object
        rgb (numpy.array): an array for active rgb value (.0, 1.0) of the points of the object
        visible (bool): shows should the object be using or not
        moving (bool): shows should the object move
        active_points (numpy.array): shows which points will be active
    """

    def __init__(self):
        self.__xyz = np.zeros([0, 3])
        self.__rgb = np.zeros([0, 3])
        self.__visible = True
        self.__moving = False
        self.__active_points = np.empty([0])

    def add_points(self, xyz, rgb=None, number=None):
        """Adding points to xyz and rgb

        If user didn't send rgb, it is made grey by default. If user doesn't point the number of active points they all
        are active.

        Args:
            xyz (numpy.array): an array for xyz coordinates of the object to add
            rgb (numpy.array): an array for rgb value (.0, 1.0) of the points of the object to add
            number (int): number of active points
        """
        try:
            self.__xyz = np.append(self.__xyz, xyz, axis=0)

            if rgb is None or not rgb.shape[0] == xyz.shape[0]:
                rgb = np.empty([xyz.shape[0], 3])
                rgb.fill(0.5)
            self.__rgb = np.append(self.__rgb, rgb, axis=0)

            if number is not None and number > xyz.shape[0]:
                print("Number of active points is more, than number of points. Do something with it.")
                number = None

            self.__active_points = np.append(self.__active_points, self.choose_random_active(xyz.shape[0], number),
                                             axis=0)

        except ValueError as e:
            print("Error in PointObject.add_points:", e)

    def set_points(self, xyz, rgb=None, number=None):
        """Setting xyz and rgb points

        If user didn't send rgb, it is made grey by default

        Args:
            xyz (numpy.array): an array for xyz coordinates of the object
            rgb (numpy.array): an array for rgb value (.0, 1.0) of the points of the object
            number (int): number of active points
        """
        try:
            self.__xyz = xyz
            if rgb is None or not rgb.shape[0] == xyz.shape[0]:
                rgb = np.empty([xyz.shape[0], 3])
                rgb.fill(0.5)
            self.__rgb = rgb
        except ValueError as e:
            print("Error in PointObject.set_points:", e)

        self.__active_points = self.choose_random_active(xyz.shape[0], number)

    def get_points(self):
        """Returns coordinates and colors of active object's points"""
        xyz = self.__xyz[self.__active_points == True]
        rgb = self.__rgb[self.__active_points == True]
        return xyz, rgb

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
        self.__xyz = self.__xyz + distance

    def scale(self, S):
        """Scaling of point cloud

        Arguments:
            S (float): scaling coefficient
        """

        matrix = np.array([[S, 0, 0, 0],
                           [0, S, 0, 0],
                           [0, 0, S, 0],
                           [0, 0, 0, 1]])

        A = np.zeros((self.__xyz.shape[0], self.__xyz.shape[1] + 1))
        A[:, :-1] = self.__xyz[:, :]

        A = np.dot(matrix, A.T).T

        self.__xyz = A[:, :-1]

    def number_of_active_points(self):
        return np.sum(self.__active_points)

    def number_of_all_points(self):
        return self.__xyz.shape[0]

    def choose_random_active(self, array_len, number):
        """Choosing the points, which will be static

        Arguments:
            array_len (int): length of array for which random indexes will be chosen
            number (int): number of points which will be chosen as active

        Returns:
            new_active (np.array): array of indicators which show is an element active or not
        """
        new_active = np.empty(array_len)
        if number is None:
            new_active.fill(True)
        else:
            true_active = np.empty([number])
            true_active.fill(True)
            false_active = np.empty([array_len - number])
            false_active.fill(False)
            new_active = np.append(true_active, false_active)
            np.random.shuffle(new_active)
        return new_active

    def clear(self):
        """Erases points"""
        self.__xyz = np.zeros([0, 3])
        self.__rgb = np.zeros([0, 3])
        self.__active_points = np.zeros([0])

    def set_number_of_active_points(self, number):
        self.__active_points = self.choose_random_active(self.number_of_all_points(), number)

    def return_n_last_points(self, number):
        return self.__xyz[-(number + 1):-1], self.__rgb[-(number + 1):-1]

    def save_all_points(self, path, name):
        """Saving all points cloud's points in .pcd format

        Arguments:
            path (string): path to the file. Folders have to exist
            name (string): name of the file
        """
        import open3d as o3d
        from pathlib import Path

        Path(path).mkdir(parents=True, exist_ok=True)
        full_path = path + "/" + name + ".pcd"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.__xyz)
        pcd.colors = o3d.utility.Vector3dVector(self.__rgb)
        o3d.io.write_point_cloud(full_path, pcd)

    def save_active_points(self, path, name):
        """Saving only active points cloud's points in .pcd format

        Arguments:
            path (string): path to the file. Folders have to exist
            name (string): name of the file
        """
        import open3d as o3d
        from pathlib import Path

        Path(path).mkdir(parents=True, exist_ok=True)
        xyz, rgb = self.get_points()

        full_path = path + "/" + name + ".pcd"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.io.write_point_cloud(full_path, pcd)

    def get_normals(self):
        import open3d as o3d

        xyz, rgb = self.get_points()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        return np.asarray(pcd.normals)

    def get_center(self):
        return np.mean(self.__xyz, axis=0)


if __name__ == "__main__":
    test = PointsObject()
    test.add_points(np.asarray([[0, 1, 2], [3, 5, 9], [6, 6, 6], [8, 4, 3]]))
    print(test.get_points())
    test.set_number_of_active_points(2)
    print(test.get_points())
    test.save_all_points("PCDs", "test")
    test.save_active_points("PCDs", "test1")
