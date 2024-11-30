#!/usr/bin/env python3

"""
Very simple test node, opens a mono image from the assets directory and
prints relative transform

"""

from relative_pose_tracker import RelativePoseTracker
from lidar_functions import keypoints_to_global, calculate_global_position, sift_lidar_simulation, remove_outliers_zscore, compute_relative_position
import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import time
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess the image to reduce background interference by converting it
    to grayscale, binarizing, and extracting contours.

    :param image_path: Path to the input image.
    :return: Preprocessed image with reduced background noise.
    """
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    th2 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 855, 0)

    th3 = cv.adaptiveThreshold(th2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 41, 0)
    #_, binary = cv.threshold(th3, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)


    contours, _ = cv.findContours(th3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv.drawContours(mask, contours, -1, (255), thickness=cv.FILLED)
    preprocessed_image = cv.bitwise_and(gray, mask)


    # cv.imshow("Preprocessed Image", preprocessed_image)

    # # Keep the window open until the user closes it
    # while True:
    #     # Wait for 1 millisecond for a key event
    #     key = cv.waitKey(1)
    #     # If the 'Esc' key is pressed (key code 27), break the loop
    #     if key == 27:  # ASCII code for 'Esc'
    #         break

    # # Destroy the window and release resources
    # cv.destroyAllWindows()

    return preprocessed_image

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
            keypoints.append(cv.KeyPoint(x=float(x), y=float(y), size=1))
        else:
            print("Invalid keypoint format:", coord)
    return keypoints


def main():
    cv_bridge = CvBridge()

    rclpy.init()
    node = rclpy.create_node("test_relative_pose_tracker_static_movement")
    
    try:
        #gives the rotation and translation of some x to the reference image 
        #therefore giving us the position of x within the enviornment
        robot_pos = RelativePoseTracker("assets/box.png", "assets/robot.png")
        #box_pos = RelativePoseTracker("assets/reference.png", "assets/box.png")
        #goal_pos = RelativePoseTracker("assets/reference.png", "assets/bot.png")

        # Path to the folder containing the images
        folder_path = "/home/jesus/ros2_ws/src/simed_lidar/tests/realsense_images"

        # Get a sorted list of image filenames
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

        # Traverse the images from a.png to h.png
        for image_file in image_files:
            file_path = folder_path + "/" + image_file
            #print("File Path:", file_path)

    
            preprocessed_image = preprocess_image(file_path)

            cv.imshow("Preprocessed Image", preprocessed_image)

            # # Keep the window open until the user closes it
            # while True:
            #     # Wait for 1 millisecond for a key event
            #     key = cv.waitKey(1)
            #     # If the 'Esc' key is pressed (key code 27), break the loop
            #     if key == 27:  # ASCII code for 'Esc'
            #         break

            # # Destroy the window and release resources
            # cv.destroyAllWindows()



            time_start = time.time()
            
            #GETS THE POSITION AND ORIENTATION OF THE ROBOT BOX AND GOAL
            #gloablized 
            robot_pose = robot_pos.get_relative_pose(preprocessed_image)
            #box_pose = box_pos.get_relative_pose(cv_frame)
            #goal_pose = goal_pos.get_relative_pose(cv_frame)

            # print(robot_pose)
            # print(box_pose)
            # print(goal_pose)

            robot_global_position = calculate_global_position((0,0,0), robot_pose)
            #box_global_position = calculate_global_position((0,0,0), box_pose)
            #goal_global_position = calculate_global_position((0,0,0), goal_pose)
            print("robot: ",robot_global_position)
            #print("box: ",box_global_position)
            #print("goal: ",goal_global_position)
            #GETS THE POSITION AND ORIENTATION OF ROBOT BOX AND GOAL
        

            #USED TO SIMULATE LIDAR
            #goal_to_ref_keypoints = goal_pos.get_keypoints(cv_frame)
            #box_to_ref_keypoints = box_pos.get_keypoints(cv_frame)
            robot_to_ref_keypoints = robot_pos.get_keypoints(preprocessed_image)
            print("Robot to reference:", len(robot_to_ref_keypoints))

            print("Keypoints data type:", type(robot_to_ref_keypoints))
            print("Keypoints shape:", np.shape(robot_to_ref_keypoints))
            #print("Keypoints data:", robot_to_ref_keypoints)


            if isinstance(robot_to_ref_keypoints, np.ndarray) and robot_to_ref_keypoints.shape[1:] == (1, 2):
                robot_to_ref_keypoints = convert_to_keypoints(robot_to_ref_keypoints)




            keypoint_image = cv.drawKeypoints(
                preprocessed_image,
                robot_to_ref_keypoints,
                None,
                color=(0, 255, 0),  # Green color for keypoints
                flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

            # Display the image with keypoints
            cv.imshow("Keypoints", keypoint_image)

            # Keep the window open until the user closes it
            while True:
                # Wait for 1 millisecond for a key event
                key = cv.waitKey(1)
                # If the 'Esc' key is pressed (key code 27), break the loop
                if key == 27:  # ASCII code for 'Esc'
                    break
                
            #goal_keypoints_global = keypoints_to_global((0,0,0), goal_to_ref_keypoints.reshape(-1,2), 0)
            #box_keypoints_global = keypoints_to_global((0,0,0), box_to_ref_keypoints.reshape(-1,2), 0)
            #robot_keypoints_global = keypoints_to_global((0,0,0), robot_to_ref_keypoints.reshape(-1,2), 0)

            #cleaned_goal_keypoints = remove_outliers_zscore(goal_keypoints_global)
            #cleaned_box_keypoints = remove_outliers_zscore(box_keypoints_global)
            #cleaned_robot_keypoints = remove_outliers_zscore(robot_keypoints_global)

            #all_global_keypoints = np.concatenate((cleaned_box_keypoints, cleaned_goal_keypoints))
            #position_of_robot = compute_relative_position(cleaned_robot_keypoints).tolist()
            #position_of_robot = position_of_robot.tolist()
            #position_of_robot.append(robot_pose[2])

            
            #lidar_data = sift_lidar_simulation(position_of_robot, all_global_keypoints) 
            #print(lidar_data)
            ####USED TO SIMULATE LIDAR

            time_taken = time.time() - time_start

            node.get_logger().info(f"time taken: {time_taken}")
            #node.get_logger().info(f"LIDAR: {lidar_data}")

    except Exception as e:
        node.get_logger().error(f"An error occurred: {e}")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()