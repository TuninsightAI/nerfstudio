"""Compute depth maps for images in the input folder.
"""
import glob
import imageio
import numpy as np
import os
import torch
from matplotlib import pyplot as plt
from pathlib import Path
from torchvision.transforms import Compose
from tqdm import tqdm

from dctoolbox.dpt.io import read_image
from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

__default_model_path = Path(__file__).parent / "dpt/dpt_hybrid-midas-501f0c75.pt"


@torch.no_grad()
def run(input_dir: Path, output_dir: Path, model_path: Path = __default_model_path):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_dir (str): path to input folder
        output_dir (str): path to output folder
        model_path (str): path to saved model
    """
    assert input_dir.exists() and input_dir.is_dir(), input_dir

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_w = net_h = 384
    model = DPTDepthModel(
        scale=0.000305,
        shift=0.1378,
        path=model_path,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
        invert=True,
        freeze=True,
    )
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()
    model = model.half()
    model.to(device)

    # get input
    img_names = glob.glob(os.path.join(input_dir, "*"))

    # create output folder
    os.makedirs(output_dir, exist_ok=True)

    for ind, img_name in tqdm(
        enumerate(img_names), total=len(img_names), dynamic_ncols=True
    ):
        if os.path.isdir(img_name):
            continue

        # input

        img = read_image(img_name)

        img_input = transform({"image": img})["image"]

        # compute
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        filename = os.path.join(
            output_dir, os.path.splitext(os.path.basename(img_name))[0]
        )
        np.savez(filename, pred=prediction)
        normalize_pred = np.clip(
            np.clip(
                255.0 / prediction.max() * (prediction - prediction.min()), 0, 255
            ).astype(np.uint8),
            0,
            255,
        ).astype(np.uint8)
        colormap = plt.cm.get_cmap("plasma")
        colored_pred = colormap(normalize_pred)[:, :, :3]

        imageio.imwrite(
            os.path.join(f"{filename}.png"), (colored_pred * 255).astype(np.uint8)
        )


if __name__ == "__main__":
    import tyro

    tyro.cli(run)
