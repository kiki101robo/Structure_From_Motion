import numpy as np
import cv2

def triangulate_3d_points(image_points1, image_points2, rotation_matrix1, translation_vector1, rotation_matrix2, translation_vector2, camera_matrix):
    """ Triangulates 3D points from their corresponding 2D projections in two different camera views.

    Args:
        image_points1 (np.array): Array of 2D points in the first image.
        image_points2 (np.array): Array of 2D points in the second image.
        rotation_matrix1 (np.array): Rotation matrix for the first camera.
        translation_vector1 (np.array): Translation vector for the first camera.
        rotation_matrix2 (np.array): Rotation matrix for the second camera.
        translation_vector2 (np.array): Translation vector for the second camera.
        camera_matrix (np.array): Intrinsic camera matrix.

    Returns:
        np.array: Array of 3D points triangulated from the input 2D points.
    """
    # Construct projection matrices for the two views
    projection_matrix1 = camera_matrix @ np.hstack((rotation_matrix1, translation_vector1))
    projection_matrix2 = camera_matrix @ np.hstack((rotation_matrix2, translation_vector2))

    # Reshape image points to conform with OpenCV's function requirements (2xN)
    reshaped_points1 = image_points1.T  # Transpose to shape (2, N)
    reshaped_points2 = image_points2.T

    # Perform triangulation using OpenCV's triangulatePoints function
    points_4d_homogeneous = cv2.triangulatePoints(projection_matrix1, projection_matrix2, reshaped_points1, reshaped_points2)
    # Convert homogeneous coordinates to 3D Euclidean coordinates
    points_3d = points_4d_homogeneous[:3, :] / points_4d_homogeneous[3, :]

    return points_3d
