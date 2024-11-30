#!/usr/bin/env python3

"""
Very simple test node, opens a mono image from the assets directory and
prints relative transform

"""

from relative_pose_tracker import RelativePoseTracker
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

def keypoints_to_global(reference_pose, keypoints, relative_theta):
    """
    Transform a list of keypoints to global coordinates based on the reference pose.
    
    :param reference_pose: (x_ref, y_ref, theta_ref) - Reference global pose.
    :param keypoints: List of SIFT keypoints relative to the reference pose.
    :param relative_theta: Orientation of the frame relative to the reference.
    :return: List of global coordinates for the keypoints.
    """
    x_ref, y_ref, theta_ref = reference_pose
    global_keypoints = []

    for kp in keypoints:
        # Relative coordinates of the keypoint
        kp_x, kp_y = kp[0], kp[1]

        # Rotate and translate to global coordinates
        global_x = x_ref + kp_x * math.cos(theta_ref) - kp_y * math.sin(theta_ref)
        global_y = y_ref + kp_x * math.sin(theta_ref) + kp_y * math.cos(theta_ref)
        
        global_keypoints.append((global_x, global_y))

    return global_keypoints

def calculate_global_robot_position(reference_pose, robot_relative_pose):
    """
    Calculate the robot's global position based on its relative pose to the reference frame.

    :param reference_pose: Tuple (x_ref, y_ref, theta_ref) - Reference frame position and orientation.
    :param robot_relative_pose: Tuple (dx_r, dy_r, theta_r) - Robot's relative position and orientation.
    :return: Tuple (x_r, y_r, theta_r) - Robot's global position and orientation.
    """
    # Extract reference pose and robot relative pose
    x_ref, y_ref, theta_ref = reference_pose
    dx_r, dy_r, theta_r = robot_relative_pose

    # Calculate robot's global position
    x_r = x_ref + dx_r * math.cos(theta_ref) - dy_r * math.sin(theta_ref)
    y_r = y_ref + dx_r * math.sin(theta_ref) + dy_r * math.cos(theta_ref)
    theta_r_global = theta_ref + theta_r

    return x_r, y_r, theta_r_global

# Example reference pose (global origin)
#reference_pose = (0, 0, 0)  # Reference at origin, no rotation

# Example robot pose relative to the reference
#robot_relative_pose = (2.0, 3.0, math.radians(45))  # Robot 2 meters right and 3 meters up, rotated 45 degrees

# Calculate the robot's global position
#robot_global_position = calculate_global_robot_position(reference_pose, robot_relative_pose)
#print("Robot's Global Position:", robot_global_position)
def sift_lidar_simulation(robot_pose, global_keypoints, num_beams=720, max_range=1000.0):
    x_r, y_r, theta_r = robot_pose

    # Initialize LIDAR ranges
    lidar_ranges = np.full(num_beams, max_range)

    # Define beam angles
    angles = np.linspace(0, 2 * np.pi, num_beams, endpoint=False)

    # Simulate each LIDAR beam
    for i, beam_angle in enumerate(angles):
        beam_angle_global = theta_r + beam_angle

        for obj_x, obj_y in global_keypoints:
            # Calculate distance and angle to the object
            distance = math.sqrt((obj_x - x_r)**2 + (obj_y - y_r)**2)
            angle_to_obj = math.atan2(obj_y - y_r, obj_x - x_r)
            angle_diff = (angle_to_obj - beam_angle_global + np.pi) % (2 * np.pi) - np.pi

            # Check if the object is within the beam's angular sector and in range
            if abs(angle_diff) < math.radians(0.5) and distance < lidar_ranges[i]:
                lidar_ranges[i] = distance

    return lidar_ranges


def remove_outliers_zscore(data, threshold=1):
    """
    Removes outliers from 2D points based on the Z-score method.

    Parameters:
    - data (array-like): An array of shape (n, 2) representing [x, y] points.
    - threshold (float): Z-score threshold for identifying outliers. Default is 3.

    Returns:
    - filtered_data (numpy.ndarray): Array with outliers removed.
    """
    data = np.array(data)  # Ensure input is a NumPy array

    # Calculate mean and standard deviation for each axis
    mean_x, mean_y = np.mean(data, axis=0)
    std_x, std_y = np.std(data, axis=0)

    # Z-score for each point
    z_scores = np.abs((data - [mean_x, mean_y]) / [std_x, std_y])

    # Filter out points with Z > threshold
    filtered_data = data[(z_scores[:, 0] < threshold) & (z_scores[:, 1] < threshold)]
    return filtered_data


def compute_relative_positions(points, reference_point=None):
    """
    Computes the relative positions of points in a 2D cluster.

    Parameters:
    - points (array-like): An array of shape (n, 2) representing [x, y] points.
    - reference_point (array-like, optional): The reference [x, y] point. 
      If None, the centroid of the points is used as the reference.

    Returns:
    - relative_positions (numpy.ndarray): Array of relative positions [dx, dy].
    - reference_point (numpy.ndarray): The reference point used for the calculation.
    """
    points = np.array(points)  # Ensure input is a NumPy array

    # Use the centroid as the reference point if none is provided
    if reference_point is None:
        reference_point = np.mean(points, axis=0)

    # Compute relative positions
    relative_positions = points - reference_point

    return reference_point


def main():
    cv_bridge = CvBridge()

    rclpy.init()
    node = rclpy.create_node("test_relative_pose_tracker_static_movement")
    
    try:
        #gives the rotation and translation of some x to the reference image 
        #therefore giving us the position of x within the enviornment
        robot_pos = RelativePoseTracker("assets/square.png", "assets/cozmopng")
        box_pos = RelativePoseTracker("assets/square.png", "assets/box2.png")
        goal_pos = RelativePoseTracker("assets/square.png", "assets/goal2.png")

        # Path to the folder containing the images
        folder_path = "/home/jesus/ros2_ws/src/simed_lidar/tests/simMovement"

        # Get a sorted list of image filenames
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

        # Traverse the images from a.png to h.png
        for image_file in image_files:
    
            cv_frame = cv.imread(folder_path + "/" + image_file, cv.IMREAD_GRAYSCALE)

            time_start = time.time()
            robot_pose = robot_pos.get_relative_pose(cv_frame)
            #print(robot_pose)

            box_pose = box_pos.get_relative_pose(cv_frame)
            #print(box_pose)

            goal_pose = goal_pos.get_relative_pose(cv_frame)
            #print(goal_pose)

            goal_to_ref_keypoints = goal_pos.get_keypoints(cv_frame)
            #print(len(goal_to_ref_keypoints))
            box_to_ref_keypoints = box_pos.get_keypoints(cv_frame)
            #print(len(box_to_ref_keypoints))
            robot_to_ref_keypoints = robot_pos.get_keypoints(cv_frame)

            #print(keypoints.shape)

            reshaped_keypoints_gtr = goal_to_ref_keypoints.reshape(-1, 2)
            reshaped_keypoints_btr = box_to_ref_keypoints.reshape(-1, 2)
            reshaped_keypoints_rtr = robot_to_ref_keypoints.reshape(-1, 2)

            ##TESTING
            # if reshaped_keypoints_btr == reshaped_keypoints_gtr:
            #     print("SAME SET OF KEYPOINTS")
            ##TESTING

            goal_keypoints_global = keypoints_to_global((0,0,0), reshaped_keypoints_gtr, 0)
            box_keypoints_global = keypoints_to_global((0,0,0), reshaped_keypoints_btr, 0)
            robot_keypoints_global = keypoints_to_global((0,0,0), reshaped_keypoints_rtr, 0)

            cleaned_goal_keypoints = remove_outliers_zscore(goal_keypoints_global)
            cleaned_box_keypoints = remove_outliers_zscore(box_keypoints_global)
            cleaned_robot_keypoints = remove_outliers_zscore(robot_keypoints_global)

            # print(len(cleaned_goal_keypoints))
            # print(len(cleaned_box_keypoints))
            # print(len(cleaned_robot_keypoints))


            ##TESTING 
            # if goal_keypoints_global == box_keypoints_global:
            #     print("SAME SET OF KEYPOINTS Global")
            # else:
            #     print("NOT SAME SET OF KEYPOINTS Global")


            # # Extract coordinates for plotting
            # goal_x, goal_y = zip(*cleaned_goal_keypoints)
            # box_x, box_y = zip(*cleaned_box_keypoints)
            # robot_x, robot_y = zip(*cleaned_robot_keypoints)


            # # Plot
            # plt.figure(figsize=(8, 8))
            # plt.scatter(goal_x, goal_y, color='green', label='Goal Keypoints', s=50)
            # plt.scatter(box_x, box_y, color='blue', label='Box Keypoints', s=50)
            # plt.scatter(robot_x, robot_y, color='red', label='Box Keypoints', s=50)


            # # Configure graph
            # plt.axhline(0, color='black', linewidth=0.5)
            # plt.axvline(0, color='black', linewidth=0.5)
            # plt.grid(color='gray', linestyle='--', linewidth=0.5)
            # plt.legend()
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.title('Global Positions of Goal, Box, and Robot')
            # plt.axis('equal')  # Equal scaling for x and y axes
            # plt.show()
            #TESTING


            all_global_keypoints = np.concatenate((cleaned_box_keypoints, cleaned_goal_keypoints))
            position_of_robot = compute_relative_positions(cleaned_robot_keypoints).tolist()
            #position_of_robot = position_of_robot.tolist()
            position_of_robot.append(robot_pose[2])

            #print(position_of_robot)


            # ##TESTING 
            # # if goal_keypoints_global == box_keypoints_global:
            # #     print("SAME SET OF KEYPOINTS Global")
            # # else:
            # #     print("NOT SAME SET OF KEYPOINTS Global")


            # # # Extract coordinates for plotting
            # goal_x, goal_y = zip(*all_global_keypoints)
            # # box_x, box_y = zip(*cleaned_box_keypoints)
            # # robot_x, robot_y = zip(*cleaned_robot_keypoints)
            # print()

            # robot_x, robot_y = position_of_robot




            # # # Plot
            # # plt.figure(figsize=(8, 8))
            # plt.scatter(goal_x, goal_y, color='green', label='Goal Keypoints', s=50)
            # # plt.scatter(box_x, box_y, color='blue', label='Box Keypoints', s=50)
            # plt.scatter(robot_x, robot_y, color='red', label='Box Keypoints', s=50)


            # # Configure graph
            # plt.axhline(0, color='black', linewidth=0.5)
            # plt.axvline(0, color='black', linewidth=0.5)
            # plt.grid(color='gray', linestyle='--', linewidth=0.5)
            # plt.legend()
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.title('Global Positions of Goal, Box, and Robot')
            # plt.axis('equal')  # Equal scaling for x and y axes
            # plt.show()
            # #TESTING

            lidar_data = sift_lidar_simulation(position_of_robot, all_global_keypoints) 
            #print(lidar_data)

            time_taken = time.time() - time_start

            node.get_logger().info(f"time taken: {time_taken}")
            #node.get_logger().info(f"LIDAR: {lidar_data}")

    except Exception as e:
        node.get_logger().error(f"An error occurred: {e}")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()