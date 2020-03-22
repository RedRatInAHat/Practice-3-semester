from points_object import PointsObject
import image_processing
import numpy as np
import visualization
import download_point_cloud
import shape_recognition
import time
import random


def create_mask():
    color_mask = image_processing.load_image("tracking_results", "global_two_different3.png")
    binary_mask = np.where(np.sum(color_mask, axis=2), 1, 0)
    image_processing.save_image(binary_mask, "Mask", "mask")


def apply_mask(rgb, depth, mask):
    print(np.max(rgb), np.max(depth), np.max(mask))
    return rgb * mask, depth * mask[:, :, 0]


def create_points_cloud():
    rgb_im = image_processing.load_image("falling ball and cube", "rgb_3.png")
    depth_im = image_processing.load_image("falling ball and cube", "depth_3.png", "depth")
    mask_im = image_processing.load_image("Mask", "mask.png")
    rgb_im, depth_im = apply_mask(rgb_im, depth_im, mask_im / 255)
    return image_processing.calculate_point_cloud(rgb_im / 255, depth_im / 255)


def save_points_cloud():
    points_cloud, points_color = create_points_cloud()
    object = PointsObject()
    object.set_points(points_cloud, points_color)
    object.save_all_points("Test", "ball")


if __name__ == "__main__":
    # save_points_cloud()
    # ball = PointsObject()
    # ball = download_point_cloud.download_to_object("preDiploma_PC/ball.pcd")
    # visualization.visualize_object([ball])
    grey_plane = download_point_cloud.download_to_object("models/red cube.ply", 5000)
    grey_plane.scale(0.1)
    grey_plane.shift([0.1, 0.5, 0.3])
    grey_plane.rotate([1, 0, 1], 2)

    # visualization.visualize_object([grey_plane])

    start = time.time()
    found_shapes = shape_recognition.RANSAC(grey_plane.get_points()[0], grey_plane.get_normals())
    print(time.time() - start)
    shapes = [grey_plane]
    for _, s in enumerate(found_shapes):
        new_shape = PointsObject()
        new_shape.add_points(s, np.asarray([[random.random(), random.random(), random.random()]] * s.shape[0]))
        shapes.append(new_shape)
    visualization.visualize_object(shapes)
