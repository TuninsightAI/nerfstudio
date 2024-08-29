import typing as t
from functools import lru_cache

from jaxtyping import Float, Bool
from torch import Tensor
from torchvision.transforms import Grayscale

from .loftr import QuadTreeLoFTR
from .._base import _MatchInterface


class LoFTRMatching(_MatchInterface):
    @lru_cache()
    @staticmethod
    def _loftr_model(
        setting: t.Literal["", "indoor", "outdoor"] = "outdoor"
    ) -> QuadTreeLoFTR:
        return QuadTreeLoFTR(setting=setting).eval().cuda()

    @staticmethod
    def predict_correspondence(
        prev_image: Float[Tensor, "3 H W"],
        next_image: Float[Tensor, "3 H W"],
        prev_mask: Float[Tensor, "H W"] | None = None,
        next_mask: Float[Tensor, "H W"] | None = None,
        *,
        threshold=0.0,
        normalize=True,
        setting: t.Literal["", "indoor", "outdoor"] = "outdoor",
    ):
        assert prev_image.shape == next_image.shape
        assert prev_image.max() <= 1.0
        assert prev_image.min() >= 0.0

        model = LoFTRMatching._loftr_model(setting=setting)
        size = prev_image.shape[1:]
        matching = model(
            {
                "image0": Grayscale()(prev_image[None, ...]),
                "image1": Grayscale()(next_image[None, ...]),
            }
        )
        mask = matching["confidence"] > threshold
        keypoints0 = matching["keypoints0"]
        keypoints1 = matching["keypoints1"]

        if prev_mask is not None:
            # check the image size and mask size
            assert prev_mask.shape == size
            mask = mask & LoFTRMatching._filter_correspondences_based_on_mask(
                keypoints0, prev_mask
            )
        if next_mask is not None:
            mask = mask & LoFTRMatching._filter_correspondences_based_on_mask(
                keypoints1, next_mask
            )
        keypoints0 = keypoints0[mask]
        keypoints1 = keypoints1[mask]

        if normalize:
            keypoints0[:, 0] = 2 * keypoints0[:, 0] / (size[1] - 1) - 1
            keypoints0[:, 1] = 2 * keypoints0[:, 1] / (size[0] - 1) - 1

            keypoints1[:, 0] = 2 * keypoints1[:, 0] / (size[1] - 1) - 1
            keypoints1[:, 1] = 2 * keypoints1[:, 1] / (size[0] - 1) - 1

        return (
            keypoints0.contiguous(),
            keypoints1.contiguous(),
            matching["confidence"][mask],
        )

    @staticmethod
    def _filter_correspondences_based_on_mask(
        keypoints: Float[Tensor, "N 2"], mask: Float[Tensor, "H W"]
    ) -> Bool[Tensor, "N 2"]:
        mask = mask > 0.5
        keypoints = keypoints.round().long()
        return mask[keypoints[:, 1], keypoints[:, 0]].bool()
