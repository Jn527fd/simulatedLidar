import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
from relative_pose_tracker import RelativePoseTracker
from lidar_functions import keypoints_to_global, calculate_global_position, sift_lidar_simulation, remove_outliers_zscore, compute_relative_position, remove_outliers
import matplotlib.pyplot as plt


cv_bridge = CvBridge()

# Configure the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

##use these configs
##minimum_distance: int = 0.9,
##minimum_matches: int = 95,
robot_pos = RelativePoseTracker("assets/square.png", "assets/box.png")


##use these configs
##
##
#box_pos = RelativePoseTracker("assets/square.png", "assets/box3.png")


##use these configs
##
##
#goal_pos = RelativePoseTracker("assets/square.png", "assets/goal3.png")

try:
    print("Press 'q' to exit the camera feed.")

    while True:
        # Wait for frames from the camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue  # Skip iteration if no frame is available

        # Convert frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        #cv_frame = cv_bridge.imgmsg_to_cv2(color_image, "mono8")


        # Perform OpenCV processing on the image
        # Example: Convert to grayscale
        processed_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        #processed_image = color_image
        # Show the processed image
        
        #cv2.imshow('Processed Image', processed_image)
        #cv2.waitKey(10)
        #cv2.destroyAllWindows()

        # #GETS THE POSITION AND ORIENTATION OF THE ROBOT BOX AND GOAL
        # #gloablized 
        robot_pose = robot_pos.get_relative_pose(processed_image)
        #box_pose = box_pos.get_relative_pose(processed_image)
        #goal_pose = goal_pos.get_relative_pose(processed_image)

        # print(robot_pose)
        # print(box_pose)
        # print(goal_pose)

        #robot_global_position = calculate_global_position((0,0,0), robot_pose)
        #box_global_position = calculate_global_position((0,0,0), box_pose)
        #goal_global_position = calculate_global_position((0,0,0), goal_pose)
        #print("robot: ",robot_global_position)
        #print("box: ",box_global_position)
        #print("goal: ",goal_global_position)
        # #GETS THE POSITION AND ORIENTATION OF ROBOT BOX AND GOAL
    

        # #USED TO SIMULATE LIDAR
        #goal_to_ref_keypoints = goal_pos.get_keypoints(processed_image)
        # #print(goal_to_ref_keypoints)
        # box_to_ref_keypoints = box_pos.get_keypoints(processed_image)
        robot_to_ref_keypoints = robot_pos.get_keypoints(processed_image)


        #print("goal: ", len(goal_to_ref_keypoints))
        # print("box: ", len(box_to_ref_keypoints))
        print("cozmo: ", len(robot_to_ref_keypoints))
        
        #goal_keypoints_global = keypoints_to_global((0,0,0), goal_to_ref_keypoints.reshape(-1,2), 0)
        # box_keypoints_global = keypoints_to_global((0,0,0), box_to_ref_keypoints.reshape(-1,2), 0)
        robot_keypoints_global = keypoints_to_global((0,0,0), robot_to_ref_keypoints.reshape(-1,2), 0)

        #cleaned_goal_keypoints = remove_outliers_zscore(goal_keypoints_global)
        # cleaned_box_keypoints = remove_outliers_zscore(box_keypoints_global)
        cleaned_robot_keypoints = remove_outliers_zscore(robot_keypoints_global)
        cleaned_robot = remove_outliers(cleaned_robot_keypoints)

        #all_global_keypoints = np.concatenate((cleaned_box_keypoints, cleaned_goal_keypoints))
        all_global_keypoints = cleaned_robot_keypoints
        position_of_robot = compute_relative_position(cleaned_robot).tolist()
        position_of_robot.append(robot_pose[2])
        #position_of_goal = compute_relative_position(cleaned_goal_keypoints).tolist()
        #position_of_goal.append(goal_pose[2])
        #position_of_box = compute_relative_position(cleaned_box_keypoints).tolist()
        #position_of_box.append(box_pose[2])

        goal_x, goal_y = zip(*all_global_keypoints)
        # box_x, box_y = zip(*cleaned_box_keypoints)
        robot_x, robot_y, w = position_of_robot
        #goal_xp, goal_yp, w = position_of_goal
        #box_x, box_y, w = position_of_box

        # # Plot
        plt.figure(figsize=(8, 8))
        plt.scatter(goal_x, goal_y, color='green', label='Goal Keypoints', s=50)
        #plt.scatter(goal_xp, goal_yp, color='purple', label='Goal estimated positon', s=50)
        #plt.scatter(box_x, box_y, color='blue', label='Box estimated postion', s=50)
        plt.scatter(robot_x, robot_y, color='red', label='robot estimated position', s=50)



        # Configure graph
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Global Positions of Goal, Box, and Robot')
        plt.axis('equal')  # Equal scaling for x and y axes
        plt.show()



        # print("position of robot: ",position_of_robot)
        # print("positon of goal: ",position_of_goal)
        # print("position of box: ", position_of_box)

        
        # lidar_data = sift_lidar_simulation(position_of_robot, all_global_keypoints) 

        # ##TODO implement a particle filter to estimate the positon of the robot, goal and box
        # ##given the new position [x,y,w]
        
        
        # ##use the lidar data and filtered relative position of robot, goal, and box 
        # #to move the cozmo robot and push the box to the goal

finally:
    # Stop the pipeline and close the window
    pipeline.stop()
    cv2.destroyAllWindows()
