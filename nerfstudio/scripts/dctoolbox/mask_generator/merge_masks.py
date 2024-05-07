import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from loguru import logger
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser("Prepare masks for dataset")
    parser.add_argument(
        "--mask-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Path to the Image folder",
    )
    parser.add_argument(
        "--merged-mask-dir",
        type=str,
        required=True,
        help="Path to the merged mask folder",
    )
    return parser.parse_args()


def main():
    args = get_args()
    mask_dirs = args.mask_dirs

    logger.info(f"Mask directories: {mask_dirs}")
    Path(args.merged_mask_dir).mkdir(parents=True, exist_ok=True)

    num_masks = [len(sorted(Path(x).rglob("*.png"))) for x in args.mask_dirs]
    assert num_masks, "No masks found"
    assert set(num_masks) == {
        num_masks[0]
    }, "Number of masks in each directory should be the same"

    def _get_image_array(image_path):
        with Image.open(image_path) as img:
            return np.array(img.convert("L")).astype(float)

    for cur_masks in tqdm(
        zip(*[sorted(Path(x).rglob("*.png")) for x in args.mask_dirs])
    ):
        assert len(set([x.stem for x in cur_masks])) == 1, "Mismatched mask names"

        masks = [_get_image_array(x) for x in cur_masks]
        merged_mask = 1 - np.array(sum(masks)).astype(np.uint8) / 255
        Image.fromarray((merged_mask * 255).astype(np.uint8)).save(
            Path(args.merged_mask_dir, cur_masks[0].name + ".png").as_posix()
        )


if __name__ == "__main__":
    main()
