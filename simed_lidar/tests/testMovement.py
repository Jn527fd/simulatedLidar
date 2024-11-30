import pycozmo

cli = pycozmo.Client()
cli.start()
cli.connect()
cli.wait_for_robot()

# Immediately set the head angle to 0.0 to avoid any movement during initialization
#cli.set_head_angle(angle=0.0)

# Drive the robot's wheels without moving the head
cli.drive_wheels(lwheel_speed=100.0, rwheel_speed=100.0, duration=5.0)

# Stop the robot (optional)
cli.stop()

cli.disconnect()
