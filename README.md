# Structure_From_Motion

## Overview
This project builds on the "Building_Built_In_Minutes" initiative, enhancing the process of reconstructing 3D scenes from multiple 2D images. By utilizing advanced Structure from Motion (SfM) techniques and integrating the OpenCV library with point cloud visualization tools, this project aims to refine previous methodologies, enhancing both accuracy and efficiency.

## Features
- **Camera Calibration:** Automatically reads camera intrinsic parameters from a calibration file.
- **Feature Matching:** Utilizes SIFT and FLANN to match features across image sequences.
- **Camera Pose Recovery:** Recovers camera poses using the essential matrix and relative pose estimation.
- **Point Triangulation:** Triangulates 3D points from matched 2D image points.
- **Bundle Adjustment:** Refines camera poses and 3D point cloud using solvePnP.
- **Point Cloud Visualization:** Displays and exports refined point clouds with associated color data.

## Installation
**Prerequisites:**
   - Python 3.6+
   - OpenCV
   - Open3D
   - NumPy
   - Matplotlib


## Usage
To process an image sequence and reconstruct the 3D scene:
1. Ensure your images and calibration file are correctly placed in the designated folder.
2. Run the main script: wrapper.py/wrapper1.py


## Configuration
- Images should be stored in `./data/images/`.
- Calibration file `K.txt` should be in the same directory as the images.

## Output
- Outputs a PLY file with the 3D point cloud which can be viewed in visualization tools like MeshLab.
## Demo 
![Demo GIF](data/images/SFM-ezgif.com-optimize)

