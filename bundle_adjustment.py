import numpy as np
import cv2

def refine_camera_pose_and_3d_points(initial_points_3d, source_image_points, destination_image_points, intrinsic_matrix, initial_rotation, initial_translation):
    """ Refines the camera pose and 3D point coordinates using a simple bundle adjustment approach.
    
    Args:
        initial_points_3d (np.array): Initial 3D world points.
        source_image_points (np.array): 2D image points in the source image.
        destination_image_points (np.array): 2D image points in the destination image.
        intrinsic_matrix (np.array): Camera's intrinsic matrix.
        initial_rotation (np.array): Initial rotation matrix of the camera.
        initial_translation (np.array): Initial translation vector of the camera.

    Returns:
        np.array: Refined 3D points.
        np.array: Refined rotation matrix.
        np.array: Refined translation vector.
    """
    # Convert data types for OpenCV compatibility
    object_points = initial_points_3d.T.astype(np.float64)
    image_points = destination_image_points.astype(np.float64)
    camera_matrix = intrinsic_matrix.astype(np.float64)
    distortion_coefficients = np.zeros((4, 1))  # Assuming no lens distortion

    # Prepare initial rotation and translation vectors for solvePnP
    if initial_rotation is not None and initial_translation is not None:
        rotation_vector, _ = cv2.Rodrigues(initial_rotation)
        translation_vector = initial_translation.astype(np.float64)
    else:
        rotation_vector = np.zeros((3, 1))
        translation_vector = np.zeros((3, 1))

    # Check if there are sufficient points for the pose estimation
    if object_points.shape[0] < 4 or image_points.shape[0] < 4:
        print("Insufficient points for pose estimation with solvePnP.")
        return None, None, None

    # Execute pose refinement using solvePnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points, image_points, camera_matrix, distortion_coefficients, 
        rotation_vector, translation_vector, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print("Pose refinement failed using solvePnP.")
        return None, None, None

    # Convert the rotation vector back to a matrix after refinement
    refined_rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    refined_translation_vector = translation_vector

    # Recalculate the 3D points using the refined pose
    projection_matrix2 = np.hstack((refined_rotation_matrix, refined_translation_vector))
    projection_matrix2 = intrinsic_matrix @ projection_matrix2
    projection_matrix1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    projection_matrix1 = intrinsic_matrix @ projection_matrix1
    homogeneous_4d_points = cv2.triangulatePoints(projection_matrix1, projection_matrix2, source_image_points.T, destination_image_points.T)
    refined_points_3d = homogeneous_4d_points[:3, :] / homogeneous_4d_points[3, :]

    return refined_points_3d, refined_rotation_matrix, refined_translation_vector
