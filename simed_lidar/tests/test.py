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
        kp_x, kp_y = kp.pt

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

def sift_lidar_simulation(robot_pose, global_keypoints, num_beams=360, max_range=10.0):
    """
    Simulate LIDAR using global SIFT keypoints as features in the environment.
    
    :param robot_pose: (x_r, y_r, theta_r) - Robot's global position and orientation.
    :param global_keypoints: List of global keypoint coordinates.
    :param num_beams: Number of LIDAR beams.
    :param max_range: Maximum range of the LIDAR.
    :return: LIDAR distances for each beam.
    """
    x_r, y_r, theta_r = robot_pose

    # Initialize LIDAR ranges
    lidar_ranges = np.full(num_beams, max_range)

    # Define beam angles
    angles = np.linspace(0, 2 * np.pi, num_beams, endpoint=False)

    # Simulate each LIDAR beam
    for i, beam_angle in enumerate(angles):
        # Beam direction in global frame
        beam_angle_global = theta_r + beam_angle

        for obj_x, obj_y in global_keypoints:
            # Calculate distance and angle to the object
            distance = math.sqrt((obj_x - x_r)**2 + (obj_y - y_r)**2)
            angle_to_obj = math.atan2(obj_y - y_r, obj_x - x_r)

            # Check if the object is within the beam's angular sector
            if abs(math.sin(angle_to_obj - beam_angle_global)) < 1e-6:
                # Update LIDAR range if the object is closer
                lidar_ranges[i] = min(lidar_ranges[i], distance)

    return lidar_ranges




cv_bridge = CvBridge()

def main():
    rclpy.init()
    node = rclpy.create_node("test_relative_pose_tracker_static_movement")
    
    try:
        #gives the rotation and translation of some x to the reference image 
        #therefore giving us the position of x within the enviornment
        robot_pos = RelativePoseTracker("assets/reference.png", "assets/robot.png")
        box_pos = RelativePoseTracker("assets/reference.png", "assets/box.png")
        goal_pos = RelativePoseTracker("assets/reference.png", "assets/goal.png")

        #this tracks the rotation and translation of x relative to y (x_to_y)
        box_to_robot =  RelativePoseTracker("assets/robot.png", "assets/box.png")
        goal_to_robot =  RelativePoseTracker("assets/robot.png", "assets/goal.png")
        box_to_goal =  RelativePoseTracker("assets/goal.png", "assets/box.png")

        # Path to the folder containing the images
        folder_path = "/home/jesus/ros2_ws/src/simed_lidar/tests/simMovement"

        # Get a sorted list of image filenames
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

        # Traverse the images from a.png to h.png
        for image_file in image_files:
    
            cv_frame = cv.imread(folder_path + "/" + image_file, cv.IMREAD_GRAYSCALE)

            time_start = time.time()
            robot_pose = robot_pos.get_relative_pose(cv_frame)
            box_pose = box_pos.get_relative_pose(cv_frame)
            goal_pose = goal_pos.get_relative_pose(cv_frame)
            #btr = box_to_robot.get_relative_pose(cv_frame)
            #gtr = goal_to_robot.get_relative_pose(cv_frame)
            #btg = box_to_goal.get_relative_pose(cv_frame)

            #robot_to_ref_keypoints = robot_pos.get_keypoints(cv_frame)
            goal_to_ref_keypoints = goal_pos.get_keypoints(cv_frame)
            box_to_ref_keypoints = box_pos.get_keypoints(cv_frame)
            #goal_to_robot_keypoints = goal_to_robot.get_keypoints(cv_frame)
            #box_to_robot_keypoints = box_to_robot.get_keypoints(cv_frame)
            #box_to_goal_keypoints = box_to_goal.get_keypoints(cv_frame)

            goal_keypoints_global = keypoints_to_global(goal_pose, goal_to_ref_keypoints, goal_pose[2])
            box_keypoints_global = keypoints_to_global(box_pose, box_to_ref_keypoints, box_pose[2])

            all_global_keypoints = (goal_keypoints_global + box_keypoints_global)

            robots_global_pos = calculate_global_robot_position((0, 0, 0), robot_pose)

            lidar_data = sift_lidar_simulation(robots_global_pos, all_global_keypoints) 

            time_taken = time.time() - time_start

            node.get_logger().info(f"time taken: {time_taken}")
            node.get_logger().info(f"LIDAR: {lidar_data}")

    except Exception as e:
        node.get_logger().error(f"An error occurred: {e}")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()