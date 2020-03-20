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
    orange_sphere.shift([0, 0.18, 0])
    # visualization.visualize_object([orange_sphere])
    # moving_orange_sphere = PotentialFieldObject(orange_sphere)

    grey_plane = download_point_cloud.download_to_object("models/grey plane.ply", 6000)
    grey_plane.scale(0.1)
    grey_plane.rotate([1, 0, 0], math.radians(90))
    # visualization.visualize(objects=[grey_plane, orange_sphere])
    # moving_orange_sphere.interaction(grey_plane)

    # blue_conus = download_point_cloud.download_to_object("models/blue conus.ply", 3000)
    # blue_conus.scale(0.4)
    # blue_conus.rotate([1, 0, 0], math.radians(30))
    # blue_conus.rotate([0, 1, 0], math.radians(60))
    # blue_conus.shift([0, -0.3, 0])
    # visualization.visualize(objects=[blue_conus, orange_sphere])
    # moving_orange_sphere.interaction(blue_conus)

    # brown_cylinder = download_point_cloud.download_to_object("models/brown cylinder.ply", 3000)
    # brown_cylinder.scale(0.4)
    # brown_cylinder.rotate([1, 0, 0], math.radians(60))
    # brown_cylinder.rotate([0, 1, 0], math.radians(30))
    # brown_cylinder.shift([-0.3, -0.6, 0])
    # visualization.visualize(objects=[brown_cylinder, orange_sphere])
    # moving_orange_sphere.interaction(brown_cylinder)

    violet_thor = download_point_cloud.download_to_object("models/violet thor.ply")
    visualization.visualize(objects=[violet_thor])
    # violet_thor.scale(0.1)
    # violet_thor.rotate([1, 0, 0], math.radians(90))
    # violet_thor.shift([-0.05, -0.1, 0])
    # visualization.visualize(objects=[violet_thor, orange_sphere])
    # moving_orange_sphere.interaction(violet_thor)


if __name__ == "__main__":
    # load_many_objects()
    try_two_objects_interaction()
