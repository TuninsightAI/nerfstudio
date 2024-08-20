# sys.path.append(os.path.join())

import torch
from torch import nn
from torch.nn import functional as F

from nerfstudio import THIRD_PARTY_PATH, CHECKPOINT_PATH

feature_extractor_package_path = (
    THIRD_PARTY_PATH / "QuadTreeAttention/FeatureMatching/src"
)


class QuadTreeLoFTR(nn.Module):
    def __init__(self, setting="outdoor"):
        super().__init__()

        import importlib.util
        import importlib.machinery
        import sys
        import os

        package_name = "src"

        # Create a spec for the package
        spec = importlib.util.spec_from_file_location(
            package_name, os.path.join(feature_extractor_package_path, "__init__.py")
        )

        # Create a module from the spec
        mypackage = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = mypackage

        # Execute the module
        spec.loader.exec_module(mypackage)
        get_cfg_defaults = mypackage.config.default.get_cfg_defaults
        LoFTR = mypackage.loftr.LoFTR
        lower_config = mypackage.utils.misc.lower_config

        config = get_cfg_defaults()
        config.merge_from_file(
            os.path.join(
                os.path.dirname(__file__),
                f"{THIRD_PARTY_PATH}/QuadTreeAttention/FeatureMatching/configs/loftr/{setting}/loftr_ds_quadtree.py",
            )
        )
        config = lower_config(config)

        matcher = LoFTR(config=config["loftr"])
        state_dict = torch.load(
            os.path.join(f"{CHECKPOINT_PATH}/feature_matching/{setting}.ckpt"),
            map_location="cpu",
        )["state_dict"]
        matcher.load_state_dict(state_dict, strict=True)

        self.new_shape = (480, 640)

        self.matcher = matcher

    def forward(self, x):
        image0, image1 = x["image0"], x["image1"]
        original_shape = image0.shape[-2:]

        image0 = F.interpolate(
            image0, self.new_shape, mode="bilinear", align_corners=False, antialias=True
        )
        image1 = F.interpolate(
            image1, self.new_shape, mode="bilinear", align_corners=False, antialias=True
        )

        batch = {
            "image0": image0,
            "image1": image1,
        }

        self.matcher(batch)

        keypoints0 = batch["mkpts0_f"]
        keypoints1 = batch["mkpts1_f"]
        confidence = batch["mconf"]

        keypoints0[..., 0] *= original_shape[-1] / self.new_shape[-1]
        keypoints0[..., 1] *= original_shape[-2] / self.new_shape[-2]

        keypoints1[..., 0] *= original_shape[-1] / self.new_shape[-1]
        keypoints1[..., 1] *= original_shape[-2] / self.new_shape[-2]

        return {
            "keypoints0": keypoints0,
            "keypoints1": keypoints1,
            "confidence": confidence,
        }
