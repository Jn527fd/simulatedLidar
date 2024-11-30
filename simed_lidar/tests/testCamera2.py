import time
import cv2
import pyrealsense2 as rs
import os
import numpy as np

# Specify the folder where you want to save the images
output_directory = './realsense_images/'

# Create the directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the pipeline to stream color data
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

# Create a function to capture and save images
def capture_and_save_image():
    # Wait for a new frame
    frames = pipeline.wait_for_frames()

    # Get the color frame
    color_frame = frames.get_color_frame()

    # Convert to numpy array
    color_image = np.asanyarray(color_frame.get_data())

    # Get the current time for filename
    current_time = time.strftime("%Y%m%d-%H%M%S")

    # Save the image as a PNG file
    image_filename = os.path.join(output_directory, f"image_{current_time}.png")
    cv2.imwrite(image_filename, color_image)
    print(f"Image saved as {image_filename}")

# Capture and save an image every minute
try:
    while True:
        capture_and_save_image()
        time.sleep(10)  # Wait for one minute before capturing again
except KeyboardInterrupt:
    print("Capture stopped by user.")
finally:
    # Stop the pipeline when done
    pipeline.stop()
