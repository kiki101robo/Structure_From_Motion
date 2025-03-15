import numpy as np
import cv2
import matplotlib.pyplot as plt

def recover_relative_camera_pose(source_points, destination_points, camera_matrix):
    """ Recovers the relative camera pose using the Essential Matrix from point correspondences.
    
    Args:
        source_points (np.array): Coordinates of matched keypoints in the first image.
        destination_points (np.array): Coordinates of matched keypoints in the second image.
        camera_matrix (np.array): The camera intrinsic matrix.

    Returns:
        R (np.array): The 3x3 relative rotation matrix.
        t (np.array): The 3x1 relative translation vector.
        pose_mask (np.array): Mask array used to distinguish inliers from outliers during recovery.

    Raises:
        ValueError: If the essential matrix computation fails.
    """
    # Calculate the Essential Matrix using RANSAC for robustness
    essential_matrix, inlier_mask = cv2.findEssentialMat(source_points, destination_points, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    if essential_matrix is None:
        raise ValueError("Essential matrix computation failed. Check the input points and camera matrix.")

    # Decompose the Essential Matrix to extract possible relative rotation and translation
    _, rotation_matrix, translation_vector, pose_mask = cv2.recoverPose(essential_matrix, source_points, destination_points, camera_matrix)

    return rotation_matrix, translation_vector, pose_mask
