# this file create image matching points library for a list of image pairs

import typing as t
import warnings
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
import rich
import torch
from PIL import Image
from jaxtyping import Float
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from nerfstudio.utils.epipolar import LoFTRMatching

warnings.filterwarnings("ignore")


@dataclass
class ImagePairMatchingConfig:
    image_folder: Path
    """image directory including cameras/images"""
    image_extension: t.Literal["png", "jpeg", "jpg"] = "jpeg"
    """image extension, choose from png, jpeg, jpg"""
    mask_folder: Path | None = None
    """ mask directory including camera/masks"""
    mask_extension: t.Literal["png", "jpeg", "jpg", "jpeg.png"] = "jpeg.png"
    """mask extension, choose from png, jpeg, jpg, jpeg.png"""
    save_path: Path | None = None
    """save path for the matching points, as a pandas file"""
    match_interval: int = 1
    """interval between images"""

    @torch.no_grad()
    def _main(self, *, image_folder: Path, mask_folder: Path | None):
        assert image_folder.exists() and image_folder.is_dir(), "image folder not found"

        image_paths = sorted(image_folder.glob(f"*.{self.image_extension}"))
        if len(image_paths) < 2:
            raise ValueError(f"At least 2 images are required in {image_folder}")

        mask_paths = zip(repeat(None), repeat(None))
        if mask_folder is not None:
            assert (
                mask_folder.exists() and mask_folder.is_dir()
            ), "mask folder not found"
            mask_paths = sorted(mask_folder.glob(f"*.{self.mask_extension}"))
            if len(mask_paths) != len(image_paths):
                raise ValueError(
                    f"Number of masks and images do not match in {mask_folder}"
                )

        results = []
        for image_pair, mask_pair in tqdm(
            zip(self.pairwise(image_paths), self.pairwise(mask_paths)),
            total=len(image_paths) - 1,
        ):
            *kps, confidence = LoFTRMatching.predict_correspondence(
                prev_image=self._read_image_to_tensor(
                    image_pair[0], device=torch.device("cuda")
                ),
                prev_mask=self._read_mask_to_tensor(
                    mask_pair[0], device=torch.device("cuda")
                )
                if mask_pair[0] is not None
                else None,
                next_image=self._read_image_to_tensor(
                    image_pair[1], device=torch.device("cuda")
                ),
                next_mask=self._read_mask_to_tensor(
                    mask_pair[1], device=torch.device("cuda")
                )
                if mask_pair[1] is not None
                else None,
                setting="indoor",
                threshold=0.5,
                normalize=False,
            )
            kps = torch.cat(kps, dim=1).cpu().numpy()
            camera_name = image_folder.name
            first_image_name = image_pair[0].stem
            second_image_name = image_pair[1].stem

            results.append(
                {
                    "from": f"{camera_name}/{first_image_name}",
                    "to": f"{camera_name}/{second_image_name}",
                    "kps": kps,
                }
            )
        results_pd = pd.DataFrame(results)
        return results_pd

    def main(self):
        assert (
            self.image_folder.exists() and self.image_folder.is_dir()
        ), "image folder not found"
        cameras = sorted([x.name for x in self.image_folder.glob("*") if x.is_dir()])
        logger.debug(f"Found {len(cameras)} cameras in {self.image_folder}")

        if self.mask_folder is not None:
            assert (
                self.mask_folder.exists() and self.mask_folder.is_dir()
            ), "mask folder not found"
            mask_cameras = sorted(
                [x.name for x in self.mask_folder.glob("*") if x.is_dir()]
            )
            assert set(cameras) == set(mask_cameras), "Camera names do not match"

        folder_results = []
        for cur_camera in cameras:
            logger.debug(f"Processing {cur_camera}")
            image_folder = self.image_folder / cur_camera
            mask_folder = None
            if self.mask_folder is not None:
                mask_folder = self.mask_folder / cur_camera
            folder_results.append(
                self._main(image_folder=image_folder, mask_folder=mask_folder)
            )
        results_pd = pd.concat(folder_results, axis=0).reset_index(drop=True)
        if self.save_path is not None:
            results_pd.to_pickle(self.save_path)
        return results_pd

    def pairwise(
        self, iterable: t.Iterable[t.Any]
    ) -> t.Iterator[t.Tuple[t.Any, t.Any]]:
        for a, b in zip(iterable, iterable[self.match_interval :]):
            yield a, b

    @staticmethod
    def _read_image_to_tensor(
        image_path: Path, device: torch.device = torch.device("cpu")
    ) -> Float[Tensor, "3 H W"]:
        with Image.open(image_path) as image:
            image_tensor = (
                torch.tensor(np.array(image)).permute(2, 0, 1).to(device).float()
                / 255.0
            )
        return image_tensor

    @staticmethod
    def _read_mask_to_tensor(
        image_path: Path, device: torch.device = torch.device("cpu")
    ) -> Float[Tensor, "H W"]:
        with Image.open(image_path) as image:
            image = image.convert("L")
            image_tensor = torch.tensor(np.array(image)).to(device).float() / 255.0
        return image_tensor.squeeze()


if __name__ == "__main__":
    pairwise_df = ImagePairMatchingConfig(
        image_folder=Path(
            "/home/jizong/Workspace/dConstruct/data/2024-08-26-fix-intrinisc/subregion/images/"
        ),
        mask_folder=Path(
            "/home/jizong/Workspace/dConstruct/data/2024-08-26-fix-intrinisc/subregion/masks/"
        ),
        mask_extension="jpeg.png",
        save_path=Path(
            "/home/jizong/Workspace/dConstruct/data/2024-08-26-fix-intrinisc/subregion/matching_points.pkl"
        ),
        match_interval=2,
    ).main()
    rich.print(pairwise_df.head(n=100))
