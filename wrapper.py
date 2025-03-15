import glob
import numpy as np
from point_cloud import *
from triangulate_pose import * 
from helper import *
from feature_matching import *
from camera_pose_recovery import *
from feature_matching import *
from bundle_adjustment import *

def process_image_sequence():
    """ Processes a sequence of images to reconstruct a 3D scene using Structure from Motion. """
    # File paths configuration
    image_directory = r'C:\Users\Kirti\Desktop\data\images\*.jpg'
    image_paths = glob.glob(image_directory)
    image_paths.sort()
    calibration_file = r'C:\Users\Kirti\Desktop\data\images\K.txt'

    if len(image_paths) < 2:
        print("Insufficient images in the directory. At least two images are required.")
        return

    # Load the images and camera calibration matrix
    try:
        images = load_images(image_paths)
        camera_matrix = read_calibration(calibration_file)
    except FileNotFoundError as error:
        print(error)
        return
    except Exception as error:
        print(f"Error reading calibration data: {error}")
        return

    print(f"Intrinsic Matrix K:\n{camera_matrix}\n")

    all_3d_points = []
    all_colors = []

    # Initialize transformation matrices
    cumulative_rotation = np.eye(3)
    cumulative_translation = np.zeros((3, 1))

    previous_rotation = cumulative_rotation.copy()
    previous_translation = cumulative_translation.copy()

    for i in range(len(images) - 1):
        current_image = images[i]
        next_image = images[i + 1]

        print(f"Processing image pair {i + 1} and {i + 2}...")

        # Match features between the current and next image
        try:
            source_points, destination_points, _ = match_features_and_visualize(current_image, next_image)
            print("Feature matching completed.\n")
        except ValueError as error:
            print(f"Feature matching error: {error}")
            continue

        # Recover the relative camera pose
        try:
            relative_rotation, relative_translation, _ = recover_relative_camera_pose(source_points, destination_points, camera_matrix)
            print("Camera pose recovery completed.\n")
        except ValueError as error:
            print(f"Camera pose recovery error: {error}")
            continue

        # Normalize the translation vector
        relative_translation /= np.linalg.norm(relative_translation)

        # Update global transformations
        cumulative_rotation = relative_rotation @ previous_rotation
        cumulative_translation = previous_translation + previous_rotation @ relative_translation

        # Triangulate 3D points
        try:
            points_3d = triangulate_3d_points(source_points, destination_points, previous_rotation, previous_translation, cumulative_rotation, cumulative_translation, camera_matrix)
            print("3D point triangulation completed.\n")
        except Exception as error:
            print(f"Triangulation error: {error}")
            continue
        
        """
        try:
            refined_points_3d, refined_rotation, refined_translation = refine_camera_pose_and_3d_points(points_3d, source_points, destination_points, camera_matrix, cumulative_rotation, cumulative_translation)
            if refined_points_3d is not None:
                points_3d = refined_points_3d
                cumulative_rotation = refined_rotation
                cumulative_translation = refined_translation
                print("Bundle adjustment completed.\n")
        except Exception as error:
            print(f"Bundle adjustment error: {error}")
            continue
        """
        # Extract colors from the current image
        colors = extract_colors_from_image(current_image, source_points.reshape(-1, 2))

        # Store the results
        all_3d_points.append(points_3d.T)
        all_colors.append(colors)

        # Update the previous transformations
        previous_rotation = cumulative_rotation
        previous_translation = cumulative_translation

    if not all_3d_points:
        print("No 3D points were reconstructed. Exiting.")
        return

    # Combine all collected 3D points and colors
    all_3d_points = np.concatenate(all_3d_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    print(f"Total 3D points reconstructed: {all_3d_points.shape[0]}")
    print(f"3D Point Cloud Shape: {all_3d_points.shape}")
    print(f"Color Data Shape: {all_colors.shape}\n")

    # Visualize and save the reconstructed point cloud
    display_point_cloud_with_normals(all_3d_points)
    export_point_cloud_to_ply(all_3d_points, all_colors, output_filename="output_with_colorswithoutbundle.ply")

    print("Structure from Motion pipeline executed successfully.")

if __name__ == '__main__':
    process_image_sequence()
