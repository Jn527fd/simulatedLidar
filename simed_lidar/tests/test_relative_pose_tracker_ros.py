#!/usr/bin/env python3

"""
Waits for a color message to be recieved, prints the transform

"""

from ar_bot_tabletop.relative_pose_tracker import RelativePoseTracker
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

cv_bridge = CvBridge()

rospy.init_node("test_relative_pose_tracker_ros")

pose_tracker = RelativePoseTracker("assets/corner.png", "assets/bot.png")

color_frame = rospy.wait_for_message("camera/color/image_raw", Image)

cv_frame = cv_bridge.imgmsg_to_cv2(color_frame, "mono8")

print(pose_tracker.get_relative_pose(cv_frame))
