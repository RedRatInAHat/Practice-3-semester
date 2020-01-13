from points_object import PointsObject
from potential_field import PotentialFieldObject
import download_point_cloud
import visualization
import math


def load_many_objects():
    models_list = []

    models_list.append(download_point_cloud.download_to_object("models/blue conus.ply"))
    models_list.append(download_point_cloud.download_to_object("models/grey plane.ply"))
    models_list.append(download_point_cloud.download_to_object("models/red cube.ply"))

    models_list[0].scale(0.1)
    models_list[0].clear()
    visualization.visualize(models_list[0].get_points()[0], models_list[0].get_points()[1])
    visualization.visualize_object(models_list)


def try_two_objects_interaction():
    orange_sphere = download_point_cloud.download_to_object("models/orange sphere.ply", 1000)
    orange_sphere.scale(0.3)
    orange_sphere.shift([0, 0.145, 0])
    # visualization.visualize_object([orange_sphere])

    grey_plane = download_point_cloud.download_to_object("models/grey plane.ply", 1000)
    grey_plane.scale(0.1)
    grey_plane.rotate([1, 0, 0], math.radians(90))
    visualization.visualize(objects=[grey_plane, orange_sphere])

    moving_orange_sphere = PotentialFieldObject(orange_sphere)

    moving_orange_sphere.interaction(grey_plane)


if __name__ == "__main__":
    # load_many_objects()
    try_two_objects_interaction()
