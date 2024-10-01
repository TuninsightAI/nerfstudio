import typing as t
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import skimage
from loguru import logger
from tqdm import tqdm


@dataclass(kw_only=True)
class ExposeConfig:
    image_dir: Path
    image_extension: t.Literal["jpeg", "png", "jpg"] = "jpeg"
    save_dir: Path

    def main(self):
        image_paths: t.List[Path] = sorted(
            self.image_dir.rglob(f"*.{self.image_extension}")
        )
        save_paths = [
            self.save_dir / x.relative_to(self.image_dir) for x in image_paths
        ]
        logger.info(f"Found {len(image_paths)} images")
        logger.info(f"Running blurry check")

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self._apply_exposure_correction, cur_image_path, cur_save_path
                ): cur_image_path
                for cur_image_path, cur_save_path in zip(image_paths, save_paths)
            }
            for future in tqdm(futures):
                cur_image_path = futures[future]
                blurry_score = future.result()

    def _apply_exposure_correction(self, image_path: Path, save_path: Path) -> float:
        assert image_path.exists() and image_path.is_file(), image_path

        rgb_img = cv2.imread(image_path.as_posix())
        # convert from RGB color-space to YCrCb
        ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)
        # equalize the histogram of the Y channel
        # ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        ycrcb_img[:, :, 0] = (
            skimage.exposure.equalize_adapthist(ycrcb_img[:, :, 0] / 255.0) * 255.0
        )
        ycrcb_img = ycrcb_img.astype("uint8")

        # convert back to RGB color-space from YCrCb
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path.as_posix(), equalized_img)


if __name__ == "__main__":
    import tyro

    config = tyro.cli(ExposeConfig)

    result = config.main()
