#!/usr/bin/env python3

"""
Responsible for tracking robot and obstacle posistions relative to the env.

Sources:
    https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    https://medium.com/analytics-vidhya/opencv-feature-matching-sift-algorithm-scale-invariant-feature-transform-16672eafb253
"""


import cv2 as cv
import numpy as np
from typing import Tuple
from typing import Union


class RelativePoseTracker:
    """
    Tracks a marker using SIFT, used to find target image pose
    relative to referance image

    :param reference_image_path: file path referance image
    :param transform_coordinates: coordinates to get location of in frame
    :param k: k for KNN
    :param minimum_distance: min distance of knn considered as valid match
    :param minimum_matches: how many matches need to be made to be considered as valid
    :param flann_trees: number of flann trees to use
    :param flann_checks: number of flann checks
    """

    def __init__(
        self,
        reference_image_path: str,
        target_image_path: str,
        k: int = 2,
        minimum_distance: int = 0.7,
        minimum_matches: int = 10,
        flann_trees: int = 5,
        flann_checks: int = 50,
    ) -> None:
        self._k = k
        self._minimum_distance = minimum_distance
        self._minimum_matches = minimum_matches

        self._sift = cv.SIFT_create()

        reference_image = cv.imread(reference_image_path, cv.IMREAD_GRAYSCALE)
        target_image = cv.imread(target_image_path, cv.IMREAD_GRAYSCALE)

        (
            self._referance_key_points,
            self._referance_description,
        ) = self._sift.detectAndCompute(reference_image, None)

        (
            self._target_key_points,
            self._target_description,
        ) = self._sift.detectAndCompute(target_image, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=flann_trees)
        search_params = dict(checks=flann_checks)

        self._flann = cv.FlannBasedMatcher(index_params, search_params)

    def get_relative_pose(
        self, image: np.ndarray
    ) -> Union[Tuple[np.array, float], None]:
        """
        returns translation and orientation of the target image relative to
        the referance image.

        :param image: camera image to perform keypoint matching on
        """
        frame_key_points, frame_description = self._sift.detectAndCompute(image, None)

        referance_matches = self._flann.knnMatch(
            self._referance_description, frame_description, k=self._k
        )

        target_matches = self._flann.knnMatch(
            self._target_description, frame_description, k=self._k
        )

        referance_filtered_matches = [
            m
            for m, n in referance_matches
            if m.distance < self._minimum_distance * n.distance
        ]

        target_filtered_matches = [
            m
            for m, n in target_matches
            if m.distance < self._minimum_distance * n.distance
        ]

        if (
            len(referance_filtered_matches) > self._minimum_matches
            and len(target_filtered_matches) > self._minimum_matches
        ):
            referance_template_matched_keypoints = np.float32(
                [
                    self._referance_key_points[m.queryIdx].pt
                    for m in referance_filtered_matches
                ]
            ).reshape(-1, 1, 2)

            referance_matched_keypoints = np.float32(
                [frame_key_points[m.trainIdx].pt for m in referance_filtered_matches]
            ).reshape(-1, 1, 2)

            target_template_matched_keypoints = np.float32(
                [
                    self._target_key_points[m.queryIdx].pt
                    for m in target_filtered_matches
                ]
            ).reshape(-1, 1, 2)

            target_matched_keypoints = np.float32(
                [frame_key_points[m.trainIdx].pt for m in target_filtered_matches]
            ).reshape(-1, 1, 2)

            referance_transformation_matrix, _ = cv.estimateAffinePartial2D(
                referance_template_matched_keypoints, referance_matched_keypoints
            )
            target_transformation_matrix, _ = cv.estimateAffinePartial2D(
                target_template_matched_keypoints, target_matched_keypoints
            )

            referance_homography_matrix, _ = cv.findHomography(
                referance_template_matched_keypoints,
                referance_matched_keypoints,
                cv.RANSAC,
                5.0,
            )
            target_homography_matrix, _ = cv.findHomography(
                target_template_matched_keypoints,
                target_matched_keypoints,
                cv.RANSAC,
                5.0,
            )

            destination = np.float32([[0, 0]]).reshape(-1, 1, 2)

            referance_translation = cv.perspectiveTransform(
                destination, referance_homography_matrix
            )
            target_translation = cv.perspectiveTransform(
                destination, target_homography_matrix
            )

            translation = target_translation - referance_translation

            rotation_target = np.arctan2(
                target_transformation_matrix[1, 0], target_transformation_matrix[0, 0]
            )
            rotation_referance = np.arctan2(
                referance_transformation_matrix[1, 0],
                referance_transformation_matrix[0, 0],
            )

            rotation = rotation_target - rotation_referance

            return (translation[0][0][0], translation[0][0][1], rotation)
        return None
    
    def get_keypoints(self, image: np.ndarray) -> list:
        """
        Returns keypoints detected in the given image using SIFT.
        
        :param image: Input camera image.
        :return: List of keypoints (pixel coordinates).
        """
        frame_keypoints, _ = self._sift.detectAndCompute(image, None)
        return frame_keypoints
