import cv2 as cv
import numpy as np
import torch
import typing as t
from jaxtyping import Float
from torch import Tensor

from ._base import _MatchInterface


class OpenCVMatching(_MatchInterface):
    @staticmethod
    def key_features_in_image(
        image: np.ndarray,
    ) -> t.Tuple[t.List[cv.KeyPoint], np.ndarray]:
        """
        Detects key features in an image and calculates their descriptors.

        Args:
            image (np.ndarray): The input image.

        Returns:
            Tuple[List[cv2.KeyPoint], np.ndarray]: A tuple containing the keypoints and descriptors.
        """

        # image to grayscale and numpy
        image_gray = cv.cvtColor(src=image, code=cv.COLOR_RGB2GRAY)
        image_gray = np.array(image_gray)

        # detect feature in the image and calculate their descriptor
        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(image_gray, None)

        return kp, des

    @staticmethod
    def match_features_in_two_image(image_1_des, image_2_des):

        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

        # Match descriptors
        matches = bf.match(image_1_des, image_2_des)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        return matches

    @staticmethod
    def predict_correspondence(
        prev_image: Float[Tensor, "3 H W"],
        next_image: Float[Tensor, "3 H W"],
        **kwargs: t.Dict[str, t.Any]
    ) -> t.Tuple[
        Float[Tensor, "*batch 2"],
        Float[Tensor, "*batch 2"],
        Float[Tensor, "*batch"],
    ]:
        assert prev_image.max() <= 1.0
        assert prev_image.min() >= 0.0
        # convert to np.uint8 and opencv format
        pred_image_cv = (
            (prev_image.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        )
        next_image_cv = (
            (next_image.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        )
        img1_key, img1_des = OpenCVMatching.key_features_in_image(pred_image_cv)
        img2_key, img2_des = OpenCVMatching.key_features_in_image(next_image_cv)
        matches = OpenCVMatching.match_features_in_two_image(img1_des, img2_des)

        # matched points
        matched_points_1 = []
        matched_points_2 = []
        distances = []
        for match in matches[:100]:
            matched_points_1.append(
                [img1_key[match.queryIdx].pt[0], img1_key[match.queryIdx].pt[1]]
            )
            matched_points_2.append(
                [img2_key[match.trainIdx].pt[0], img2_key[match.trainIdx].pt[1]]
            )
            distances.append(match.distance)
        matched_points_1 = torch.from_numpy(np.array(matched_points_1)).float()
        matched_points_2 = torch.from_numpy(np.array(matched_points_2)).float()
        distances = torch.from_numpy(np.array(distances)).float()

        return matched_points_1, matched_points_2, distances
