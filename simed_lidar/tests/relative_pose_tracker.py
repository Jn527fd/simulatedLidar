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
        minimum_distance: int = .7, #orgionally 0.7
        minimum_matches: int = 1, #orgionally 10
        flann_trees: int = 5, #orgionally 5
        flann_checks: int = 50,
    ) -> None:
        self._k = k
        self._minimum_distance = minimum_distance
        self._minimum_matches = minimum_matches

        self._sift = cv.SIFT_create()  

        reference_image = cv.imread(reference_image_path, cv.IMREAD_GRAYSCALE)
        target_image = cv.imread(target_image_path, cv.IMREAD_GRAYSCALE)

        # # Apply CLAHE for contrast enhancement
        #clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
        #reference_image = clahe.apply(reference_image)
        #target_image = clahe.apply(target_image)


        reference_image = cv.GaussianBlur(reference_image, (7, 7), 0)
        target_image = cv.GaussianBlur(target_image, (7, 7), 0)

        (
            self._referance_key_points,
            self._referance_description,
        ) = self._sift.detectAndCompute(reference_image, None)

        (
            self._target_key_points,
            self._target_description,
        ) = self._sift.detectAndCompute(target_image, None)

        FLANN_INDEX_KDTREE = 5
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=flann_trees)
        search_params = dict(checks=flann_checks)

        self._flann = cv.FlannBasedMatcher(index_params, search_params)
        self.target_matched_keypoints = None

    def filter_keypoints_even_odd(self, keypoints: list, image_size: Tuple[int, int]) -> list:
        """
        Filter keypoints using the Even-Odd algorithm to focus on relevant features.
        
        :param keypoints: List of detected keypoints.
        :param image_size: Dimensions of the image (width, height).
        :return: Filtered list of keypoints.
        """
        width, height = image_size
        filtered_keypoints = [
            kp for kp in keypoints if int(kp.pt[0]) % 2 == 0 and int(kp.pt[1]) % 2 == 1
        ]
        return filtered_keypoints


    def get_relative_pose(
        self, image: np.ndarray
    ) -> Union[Tuple[np.array, float], None]:
        """
        returns translation and orientation of the target image relative to
        the referance image.

        :param image: camera image to perform keypoint matching on
        """
        frame_key_points, frame_description = self._sift.detectAndCompute(image, None)
        print("Unfilterd Frame Key Points:",len(frame_key_points))
        #frame_key_points = self.filter_keypoints_even_odd(frame_key_points, image.shape[::-1])
        #print("Filtered Frame Key Points:",len(frame_key_points))

        #match keypoints using FLANN
        referance_matches = self._flann.knnMatch(
            self._referance_description, frame_description, k=self._k
        )

        print("Reference Matches:",len(referance_matches))

        target_matches = self._flann.knnMatch(
            self._target_description, frame_description, k=self._k
        )

        print("Target Matches:",len(target_matches))

        #Apply ratio test to filter matches
        referance_filtered_matches = [
            m
            for m, n in referance_matches
            if m.distance < self._minimum_distance * n.distance
        ]

        print("Filtered Reference Matches:",len(referance_filtered_matches))


        target_filtered_matches = [
            m
            for m, n in target_matches
            if m.distance < self._minimum_distance * n.distance
        ]

        print("Filtered Target Matches:",len(target_filtered_matches))


        if (
            len(referance_filtered_matches) > self._minimum_matches
            and len(target_filtered_matches) > self._minimum_matches
        ):
            print("PASSED THE IF STATEMENT")
            #Extract keypoint coordinates for refrerence and target matches
            referance_template_matched_keypoints = np.float32(
                [
                    self._referance_key_points[m.queryIdx].pt
                    for m in referance_filtered_matches
                ]
            ).reshape(-1, 1, 2)

            print("Reference Template Matched Keypoints:", len(referance_template_matched_keypoints))

            #[ERROR] [1732978702.589503485] [test_relative_pose_tracker_static_movement]: An error occurred: list index out of range

            referance_matched_keypoints = np.float32(
                [frame_key_points[m.trainIdx].pt for m in referance_filtered_matches]
            ).reshape(-1, 1, 2)

            print("Referance Matched Keypoints:", len(referance_matched_keypoints))

            target_template_matched_keypoints = np.float32(
                [
                    self._target_key_points[m.queryIdx].pt
                    for m in target_filtered_matches
                ]
            ).reshape(-1, 1, 2)

            self.target_matched_keypoints = np.float32(
                [frame_key_points[m.trainIdx].pt for m in target_filtered_matches]
            ).reshape(-1, 1, 2)
            

            referance_transformation_matrix, _ = cv.estimateAffinePartial2D(
                referance_template_matched_keypoints, referance_matched_keypoints
            )
            target_transformation_matrix, _ = cv.estimateAffinePartial2D(
                target_template_matched_keypoints, self.target_matched_keypoints
            )

            referance_homography_matrix, _ = cv.findHomography(
                referance_template_matched_keypoints,
                referance_matched_keypoints,
                cv.RANSAC,
                3.0,
            )
            target_homography_matrix, _ = cv.findHomography(
                target_template_matched_keypoints,
                self.target_matched_keypoints,
                cv.RANSAC,
                3.0,
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
        return self.target_matched_keypoints
