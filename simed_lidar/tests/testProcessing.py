import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from lidar_functions import keypoints_to_global, calculate_global_position, sift_lidar_simulation, remove_outliers_zscore, compute_relative_position, remove_outliers
from relative_pose_tracker import RelativePoseTracker

def convert_to_keypoints(coordinates):
    """
    Converts a NumPy array of shape (N, 1, 2) to cv2.KeyPoint objects.

    :param coordinates: NumPy array of shape (N, 1, 2).
    :return: List of cv2.KeyPoint objects.
    """
    keypoints = []
    for coord in coordinates:
        if coord.shape == (1, 2):  # Ensure it's a valid (1, 2) array
            x, y = coord[0]  # Extract (x, y) coordinates
            keypoints.append(cv.KeyPoint(x=float(x), y=float(y), size=5))
        else:
            print("Invalid keypoint format:", coord)
    return keypoints



def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess the image to reduce background interference by converting it
    to grayscale, binarizing, and extracting contours.

    :param image_path: Path to the input image.
    :return: Preprocessed image with reduced background noise.
    """
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret1,th1 = cv.threshold(gray,105,255,cv.THRESH_BINARY)

    th2 = cv.adaptiveThreshold(th1, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 19, 0)

    #th3 = cv.adaptiveThreshold(th2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 111, 0)
    #_, binary = cv.threshold(th2, 153, 255, cv.THRESH_OTSU)

    #th4 = cv.adaptiveThreshold(th3, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_TOZERO, 911, 0)

    #th4 = cv.threshold(th3, 119, 255, cv.THRESH_BINARY)





    contours, _ = cv.findContours(th2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv.drawContours(mask, contours, -1, (255), thickness=cv.FILLED)
    preprocessed_image = cv.bitwise_and(gray, mask)

    blur = cv.GaussianBlur(preprocessed_image,(19, 19),0)
    ret3,th3 = cv.threshold(blur, 0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)



    contours, _ = cv.findContours(th3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv.drawContours(mask, contours, -1, (255), thickness=cv.FILLED)
    preprocessed_image2 = cv.bitwise_and(gray, mask)




    return preprocessed_image, preprocessed_image2


folder_path = "/home/jesus/ros2_ws/src/simed_lidar/tests/realsense_images"

# Get a sorted list of image filenames
image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
robot_pos = RelativePoseTracker("assets/box.png", "assets/robot.png")
currNum = 0


# Traverse the images from a.png to h.png
for image_file in image_files:
    if currNum == 4:
        break
    # Replace "input_image.png" with the actual path to your image
    sample_image_path = folder_path + "/" + image_file  # Replace with the path to your PNG file
    preprocessed1, preprocessed2 = preprocess_image(sample_image_path)

    robot_pose = robot_pos.get_relative_pose(preprocessed2)
    print("robot: ",robot_pose)

    robot_to_ref_keypoints = robot_pos.get_keypoints(preprocessed2)
    robot_to_ref_keypoints = remove_outliers_zscore(robot_to_ref_keypoints.reshape(-1,2))
    print("Robot to reference:", len(robot_to_ref_keypoints))

    #cleaned_robot_keypoints = remove_outliers_zscore(robot_to_ref_keypoints.shape(-1,2))
    #cleaned_robot = remove_outliers(robot_to_ref_keypoints)

    #mean_x, mean_y = np.mean(robot_to_ref_keypoints, axis=0)

    
    #newArr = np.array([mean_x, mean_y])

    robot_to_ref_keypoints = robot_to_ref_keypoints.reshape(-1, 1, 2)


    if isinstance(robot_to_ref_keypoints, np.ndarray) and robot_to_ref_keypoints.shape[1:] == (1, 2):
                robot_to_ref_keypoints = convert_to_keypoints(robot_to_ref_keypoints)

    keypoint_image = cv.drawKeypoints(
        preprocessed2,
        robot_to_ref_keypoints,
        None,
        color=(0, 255, 0),  # Green color for keypoints
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )


    # Display the original and preprocessed image side by side
    original_image = cv.imread(sample_image_path, cv.IMREAD_GRAYSCALE)
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(preprocessed2, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Preprocessed Image")
    plt.imshow(keypoint_image, cmap='gray')
    plt.axis('off')

    # plt.subplot(1, 2, 2 )
    # plt.title("Preprocessed Image")
    # plt.imshow(preprocessed2, cmap='gray')
    # plt.axis('off')

    plt.tight_layout()
    plt.show()

    currNum += 1
