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

def calculate_global_position(reference_pose, robot_relative_pose):
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

def remove_outliers_zscore(data, threshold=2):
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


def compute_relative_position(points, reference_point=None):
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

import numpy as np

def remove_outliers(points, threshold_factor=2):
    """
    Removes outliers from an array of points using the Euclidean distance from the mean.
    
    Parameters:
        points (list or np.ndarray): Array of [x, y] points, shape (n, 2).
        threshold_factor (float): Factor for standard deviation to define the outlier threshold. Default is 2.
    
    Returns:
        np.ndarray: Array of [x, y] points with outliers removed.
    """
    points = np.array(points)  # Ensure input is a NumPy array
    if points.shape[1] != 2:
        raise ValueError("Input array must have shape (n, 2).")
    
    # Calculate the mean of the points
    mean = np.mean(points, axis=0)
    
    # Calculate the Euclidean distances from the mean
    distances = np.linalg.norm(points - mean, axis=1)
    
    # Determine the threshold based on the standard deviation
    threshold = threshold_factor * np.std(distances)
    
    # Filter points within the threshold
    filtered_points = points[distances < threshold]
    
    return filtered_points
