from points_object import PointsObject
import download_point_cloud
import visualization

if __name__ == "__main__":
    models_list = []

    models_list.append(download_point_cloud.download_to_object("models/blue conus.ply"))
    # models_list.append(download_point_cloud.download_to_object("models/grey plane.ply"))
    # models_list.append(download_point_cloud.download_to_object("models/red cube.ply"))

    models_list[0].scale(0.1)
    models_list[0].clear()
    # visualization.visualize(models_list[0].get_points()[0], models_list[0].get_points()[1])
    visualization.visualize_object(models_list)