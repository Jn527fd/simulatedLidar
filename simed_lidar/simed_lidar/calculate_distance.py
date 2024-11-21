"""Responsible for calculating the distance between two points using camera
intrinsics

"""

import numpy as np
import pyrealsense2 as rs


class CalculateDistance:
    def __init__(self) -> None:
        self.pipeline = rs.pipeline()

    def calculate_distance(
        self, image: np.ndarray, first_point: tuple, second_point: tuple
    ) -> float:
        color_intrin = self.color_intrin

        udist = self.depth_frame.get_distance(first_point[0], first_point[1])
        vdist = self.depth_frame.get_distance(second_point[0], second_point[1])

        point1 = rs.rs2_deproject_pixel_to_point(
            color_intrin, [first_point[0], first_point[1]], udist
        )
        point2 = rs.rs2_deproject_pixel_to_point(
            color_intrin, [second_point[0], second_point[1]], vdist
        )

        dist = np.sqrt(
            np.pow(point1[0] - point2[0], 2)
            + np.pow(point1[1] - point2[1], 2)
            + np.pow(point1[2] - point2[2], 2)
        )

        return dist