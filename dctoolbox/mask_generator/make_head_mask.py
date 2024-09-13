import typing as t
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro
from tqdm import tqdm


@dataclass
class HeadMaskGeneratorConfig:
    image_dir: Path
    """ Path to the Image folder """

    mask_dir: Path
    """ Path to the mask folder"""
    enlarge_factor: float = 1.0

    extension: t.Literal[".png", ".jpg", ".jpeg"] = ".png"

    def __post_init__(self):
        assert self.image_dir.exists(), f"{self.image_dir} does not exist"
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        assert self.enlarge_factor in {
            1.0,
            1.25,
            1.5,
        }, "enlarge factor should be 1.0, 1.25, 1.5"

    def main(self):
        Path(self.mask_dir).mkdir(parents=True, exist_ok=True)
        for f in tqdm(self.image_dir.rglob(f"*{self.extension}")):
            img = cv2.imread(str(f))
            out = np.zeros(img.shape[:2])
            h, w = out.shape
            if self.enlarge_factor == 1.0:
                h_factor = 0.8
                w_factor = 0.2
            elif self.enlarge_factor == 1.25:
                h_factor = 0.75
                w_factor = 0.25
            elif self.enlarge_factor == 1.5:
                h_factor = 0.73
                w_factor = 0.35
            else:
                raise ValueError
            if "DECXIN2023012348" in str(f):
                out[int(h * h_factor) : h, 0 : int(w * w_factor)] = 255
            if "DECXIN2023012347" in str(f):
                out[int(h * h_factor) : h, w - int(w * w_factor) : w] = 255
            output_path = self.mask_dir / f.relative_to(self.image_dir)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(Path(str(output_path) + ".png").as_posix(), out)


if __name__ == "__main__":
    tyro.cli(HeadMaskGeneratorConfig).main()
