from points_object import PointsObject
import download_point_cloud
import visualization

if __name__ == "__main__":
    models_list = []

    models_list.append(download_point_cloud.download_to_object("models/blue conus.ply"))
    visualization.visualize(models_list[0].get_points()[0], models_list[0].get_points()[1])