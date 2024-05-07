import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser("Prepare masks for dataset")
parser.add_argument(
    "--image-dir", type=str, required=True, help="Path to the Image folder"
)
parser.add_argument(
    "--mask-dir", type=str, required=True, help="Path to the mask folder"
)
args = parser.parse_args()
image_path = args.image_dir
ext = "png"
search = "*"
# search = "*DECXIN2023012348*"
# search = "*DECXIN2023012347*"
Path(args.mask_dir).mkdir(parents=True, exist_ok=True)
for f in tqdm(glob.glob(image_path + "/" + search + "." + ext)):
    img = cv2.imread(f)
    out = np.zeros(img.shape[:2])
    h, w = out.shape
    if "DECXIN2023012348" in f:
        out[int(h * 0.7) : h, 0 : int(w * 0.3)] = 255
    if "DECXIN2023012347" in f:
        out[int(h * 0.7) : h, w - int(w * 0.3) : w] = 255
    out_f = f.replace("." + ext, ".png")
    cv2.imwrite(Path(args.mask_dir, Path(out_f).stem + ".png").as_posix(), out)
