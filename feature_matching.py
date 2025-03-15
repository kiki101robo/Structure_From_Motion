import numpy as np
import cv2
import matplotlib.pyplot as plt

def match_features_and_visualize(image1, image2):
    """ Matches features between two images using SIFT and FLANN and visualizes the results. """
    # Convert images to grayscale to prepare for feature detection
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT feature detector with an increased number of features for detailed analysis
    sift_detector = cv2.SIFT_create(nfeatures=5000)

    # Detect keypoints and their descriptors in both images
    keypoints1, descriptors1 = sift_detector.detectAndCompute(gray_image1, None)
    keypoints2, descriptors2 = sift_detector.detectAndCompute(gray_image2, None)

    # Validate that descriptors were found in both images
    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Descriptors not found in one or both images.")

    # Set up the FLANN matcher parameters using a KDTree index
    flann_index_kdtree = 1
    index_parameters = dict(algorithm=flann_index_kdtree, trees=5)
    search_parameters = dict(checks=100)  # More checks for higher accuracy

    # Create the FLANN matcher and perform matching between descriptors
    flann_matcher = cv2.FlannBasedMatcher(index_parameters, search_parameters)
    raw_matches = flann_matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Filter matches using Lowe's ratio test to find high-quality matches
    quality_matches = []
    for match1, match2 in raw_matches:
        if match1.distance < 0.7 * match2.distance:
            quality_matches.append(match1)

    # Output the total count of matches and the number of high-quality matches
    print(f"Total matches found: {len(raw_matches)}")
    print(f"High-quality matches after Lowe's ratio test: {len(quality_matches)}")

    # Verify that a sufficient number of good matches exists
    if len(quality_matches) < 10:
        raise ValueError("Not enough good matches. Try using images with more distinct features.")

    # Visualize the matches for debugging or presentation purposes
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, quality_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Extract the coordinates of the good matches
    source_points = np.float32([keypoints1[m.queryIdx].pt for m in quality_matches])
    destination_points = np.float32([keypoints2[m.trainIdx].pt for m in quality_matches])

    return source_points, destination_points, matched_image
