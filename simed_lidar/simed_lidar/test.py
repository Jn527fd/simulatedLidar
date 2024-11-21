#!/usr/bin/env python3

"""
Very simple test node, opens a mono image from the assets directory and
prints relative transform

"""

from ar_bot_tabletop.relative_pose_tracker import RelativePoseTracker
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import time

cv_bridge = CvBridge()

rospy.init_node("test_relative_pose_tracker_static")

#gives the rotation and translation of some x to the reference image 
#therefore giving us the position of x within the enviornment
robot_pos = RelativePoseTracker("assets/reference.png", "assets/robot.png")
box_pos = RelativePoseTracker("assets/reference.png", "assets/box.png")
goal_pos = RelativePoseTracker("assets/reference.png", "assets/goal.png")

#this tracks the rotation and translation of x relative to y (x_to_y)
box_to_robot =  RelativePoseTracker("assets/robot.png", "assets/box.png")
goal_to_robot =  RelativePoseTracker("assets/robot.png", "assets/goal.png")
box_to_goal =  RelativePoseTracker("assets/goal.png", "assets/box.png")

cv_frame = cv.imread("assets/mono_frame.png", cv.IMREAD_GRAYSCALE)




time_start = time.time()
print(pose_tracker.get_relative_pose(cv_frame))
print(f"time taken: {time.time() - time_start}s")