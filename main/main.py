from points_object import PointsObject
import image_processing
import visualization
import download_point_cloud
import shape_recognition
import moving_prediction
import open3d_icp

if __name__ == "__main__":
    # parameters


    # load the model
    stable_object = download_point_cloud.download_to_object("models/grey plane.ply", 3000)
    stable_object.scale(0.3)
    stable_object.rotate([90, 0, 0])

    falling_object = download_point_cloud.download_to_object("models/red cube.ply", 3000)
    falling_object.scale(0.3)
    falling_object.shift([0, 3, 0])

    shapes = [falling_object]
