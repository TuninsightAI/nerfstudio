"""
Use the output of visibility to check to create a new subset dataset
"""
import json
import shutil
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from tqdm import tqdm


def save_image(image_path, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(image_path, save_path)


@dataclass
class SubregionConfig:
    image_dir: Path
    visibility_json: Path
    output_dir: Path

    def main(self):
        with open(self.visibility_json, "r") as f:
            visibility = json.load(f)
        cameras = set([vis.split("/")[0] for vis in visibility])
        time_stamps = set([vis.split("/")[1] for vis in visibility])

        visibility = sorted(
            [
                f"{camera}/{time_stamp}"
                for camera, time_stamp in product(cameras, time_stamps)
            ]
        )

        for cur_image in tqdm(visibility):
            image_path = self.image_dir / (cur_image + ".jpeg")

            save_image_path = (
                self.output_dir / "images" / image_path.relative_to(self.image_dir)
            )
            # save_mask_path = save_mask_path.parent / (save_image_path.stem + ".jpeg.png")

            save_image(image_path, save_image_path)
            # save_image(mask_path, save_mask_path2)
