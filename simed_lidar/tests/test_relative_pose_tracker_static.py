#!/usr/bin/env python3

"""
Very simple test node: opens a mono image from the assets directory and
prints the relative transform.
"""

from relative_pose_tracker import RelativePoseTracker
import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import time
import os

cv_bridge = CvBridge()

def main():
    rclpy.init()
    node = rclpy.create_node("test_relative_pose_tracker_static")
    
    try:
        # Initialize the RelativePoseTracker
        corner_path = "assets/corner.png"
        bot_path = "assets/bot.png"
        frame_path = "assets/color_frame.png"
        
        if not (os.path.exists(corner_path) and os.path.exists(bot_path) and os.path.exists(frame_path)):
            node.get_logger().error("One or more input images do not exist.")
            return

        pose_tracker = RelativePoseTracker(corner_path, bot_path)
        
        # Load the grayscale image
        cv_frame = cv.imread(frame_path, cv.IMREAD_GRAYSCALE)
        if cv_frame is None:
            node.get_logger().error("Failed to load mono_frame.png.")
            return

        # Compute relative pose
        time_start = time.time()
        relative_pose = pose_tracker.get_relative_pose(cv_frame)
        time_taken = time.time() - time_start

        relative_pos_arr = [relative_pose[0][0][0][0], relative_pose[0][0][0][1], relative_pose[1]]
        
        # Log the results
        node.get_logger().info(f"Relative pose: {relative_pos_arr}")
        node.get_logger().info(f"Time taken: {time_taken:.2f}s")

    except Exception as e:
        node.get_logger().error(f"An error occurred: {e}")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
