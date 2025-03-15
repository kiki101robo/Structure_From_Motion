import open3d as o3d
import numpy as np

def display_point_cloud_with_normals(point_cloud_data):
    """ Visualizes a 3D point cloud with estimated normals in a viewing window.
    
    Args:
        point_cloud_data (np.array): Array of 3D coordinates representing the point cloud.
    """
    # Create a point cloud object from the input data
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)

    # Automatically estimate normals for the point cloud
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Ensure normals are oriented in a consistent direction, assuming a general viewpoint
    point_cloud.orient_normals_towards_camera_location(np.array([0, 0, 0]))

    # Render the point cloud in a window with specific properties
    o3d.visualization.draw_geometries([point_cloud], window_name="3D Reconstruction View", width=800, height=600, left=50, top=50)

def export_point_cloud_to_ply(point_cloud_data, point_cloud_colors=None, output_filename="output.ply"):
    """ Saves a colored 3D point cloud to a PLY file, suitable for external visualization tools like Meshlab.
    
    Args:
        point_cloud_data (np.array): Array of 3D coordinates for the point cloud.
        point_cloud_colors (np.array, optional): Array of colors associated with each point.
        output_filename (str): The filename to save the PLY file as.
    """
    # Create a point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)

    # If colors are provided, attach them to the point cloud
    if point_cloud_colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(point_cloud_colors)

    # Write the point cloud data to a PLY file
    o3d.io.write_point_cloud(output_filename, point_cloud)
    print(f"Point cloud successfully saved to {output_filename}")
