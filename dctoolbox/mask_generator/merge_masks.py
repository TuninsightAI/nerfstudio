import typing as t
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from PIL import Image
from tqdm import tqdm


@dataclass
class MergeMasksConfig:
    mask_dirs: t.List[str]
    """ Path to the Image folder """

    output_dir: str
    """ Path to the merged mask folder"""

    def __post_init__(self):
        assert all(Path(x).exists() for x in self.mask_dirs), f"{self.mask_dirs} does not exist"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def main(self):
        num_masks = [len(sorted(Path(x).rglob("*.png"))) for x in self.mask_dirs]
        assert num_masks, "No masks found"
        assert set(num_masks) == {num_masks[0]}, "Number of masks in each directory should be the same"

        def _get_image_array(image_path):
            with Image.open(image_path) as img:
                return np.array(img.convert("L")).astype(float)

        for cur_masks in tqdm(zip(*[sorted(Path(x).rglob("*.png")) for x in self.mask_dirs])):
            assert len(set([x.stem for x in cur_masks])) == 1, "Mismatched mask names"
            cur_masks: t.List[Path]

            masks = [_get_image_array(x) for x in cur_masks]
            merged_mask = 1 - np.array(sum(masks)).astype(np.uint8) / 255
            Path(self.output_dir, cur_masks[0].relative_to(self.mask_dirs[0]).parent).mkdir(parents=True, exist_ok=True)

            Image.fromarray((merged_mask * 255).astype(np.uint8)).save(
                Path(self.output_dir, cur_masks[0].relative_to(self.mask_dirs[0]).parent,
                     cur_masks[0].name + ".png").as_posix())
            Image.fromarray((merged_mask * 255).astype(np.uint8)).save(
                Path(self.output_dir, cur_masks[0].relative_to(self.mask_dirs[0]).parent, cur_masks[0].name).as_posix())


if __name__ == "__main__":
    tyro.cli(MergeMasksConfig).main()
