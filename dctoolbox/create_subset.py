"""
Use the output of visibility to check to create a new subset dataset
"""
import json
import shutil
from itertools import product
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass

def save_image(image_path, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(image_path, save_path)


@dataclass
class SubregionConfig:
    input_path: Path
    output_path: Path
    git_hash: str

    def __post_init__(self):
        assert self.input_path.exists(), f"Config path {self.input_path} does not exist"
        assert self.output_path.exists(), f"Config path {self.output_path} does not exist"

    def main(self):

        image_folder = self.output_path / "images"
        mask_folder = self.output_path / "masks"

        visibility_path = Path(self.output_path, "visibility", f"git_{self.git_hash}", "visibility.json")
        with open(visibility_path, "r") as f:
            visibility = json.load(f)
        cameras = set([vis.split("/")[0] for vis in visibility])
        time_stamps = set([vis.split("/")[1] for vis in visibility])

        visibility = sorted(
            [f"{camera}/{time_stamp}" for camera, time_stamp in product(cameras, time_stamps)]
        )
        
        save_folder = Path(self.input_path.parent, "subregion1")

        for cur_image in tqdm(visibility):
            image_path = image_folder / (cur_image + ".jpeg")
            mask_path = mask_folder / (cur_image + ".jpeg.png")

            save_image_path = save_folder / "images" / image_path.relative_to(image_folder)
            save_mask_path = save_folder / "masks" / mask_path.relative_to(mask_folder)
            # save_mask_path = save_mask_path.parent / (save_image_path.stem + ".jpeg.png")

            save_mask_path2 = save_folder / "masks" / mask_path.relative_to(mask_folder)

            save_image(image_path, save_image_path)
            save_image(mask_path, save_mask_path)
            # save_image(mask_path, save_mask_path2)





