import cv2
import numpy as np
import typing as t
import tyro
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm


@dataclass
class HeadMaskGeneratorConfig:
    image_dir: Path
    """ Path to the Image folder """

    mask_dir: Path
    """ Path to the mask folder"""

    extension: t.Literal[".png", ".jpg"] = ".png"

    def __post_init__(self):
        assert self.image_dir.exists(), f"{self.image_dir} does not exist"
        self.mask_dir.mkdir(parents=True, exist_ok=True)

    def main(self):
        Path(self.mask_dir).mkdir(parents=True, exist_ok=True)
        for f in tqdm(self.image_dir.rglob(f"*{self.extension}")):
            img = cv2.imread(str(f))
            out = np.zeros(img.shape[:2])
            h, w = out.shape
            if "DECXIN2023012348" in f:
                out[int(h * 0.7): h, 0: int(w * 0.3)] = 255
            if "DECXIN2023012347" in f:
                out[int(h * 0.7): h, w - int(w * 0.3): w] = 255
            out_f = str(f).replace("." + self.extension, ".png")
            cv2.imwrite(Path(self.mask_dir, Path(out_f).stem + ".png").as_posix(), out)


if __name__ == "__main__":
    tyro.cli(HeadMaskGeneratorConfig).main()
