from points_object import PointsObject
import download_point_cloud


if __name__ == "__main__":
    models_list = []

    points, color = download_point_cloud.download_ply("models/blue conus.ply")
    models_list.append(PointsObject)
    models_list[0].set_points(points, color)

    points, color = download_point_cloud.download_ply("models/brown cylinder.ply")
    models_list.append(PointsObject)
    models_list[1].set_points(points, color)
