import typing as t
from pathlib import Path

import cv2
import numpy as np
import torch

from dctoolbox.depth_anything_v2.dpt import DepthAnythingV2

__default_model_path = lambda x: (
    Path(__file__).parent / "checkpoints" / f"depth_anything_v2_{x}.pth"  # noqa
)


def run(
    input_dir: Path,
    output_dir: Path,
    model_type: t.Literal["vits", "vitb", "vitl", "vitg"] = "vitl",
    image_extension: t.Literal["png", "jpg", "jpeg"] = "png",
):
    """Run Depth Anything V2 to compute depth maps.

    Args:
        input_dir (str): path to input folder
        output_dir (str): path to output folder
        model_path (str): path to saved model
    """
    assert input_dir.exists() and input_dir.is_dir(), input_dir
    assert output_dir.exists() and output_dir.is_dir(), output_dir

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }
    model_path = __default_model_path(model_type)

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(model_path, map_location="cpu"))
    depth_anything = depth_anything.to(device).eval()

    # get input
    image_paths = sorted(Path(input_dir).rglob(f"*.{image_extension}"))

    for k, cur_image_path in enumerate(image_paths):
        raw_image = cv2.imread(cur_image_path.as_posix())

        disparity = depth_anything.infer_image(raw_image, args.input_size)

        disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min())

        disparity = disparity.astype(np.float32)

        depth = 1 / (10.0 + disparity)

        filename = (
            output_dir
            / cur_image_path.relative_to(input_dir).parent
            / cur_image_path.stem
        )

        filename.parent.mkdir(parents=True, exist_ok=True)
        np.savez(filename, pred=depth)
