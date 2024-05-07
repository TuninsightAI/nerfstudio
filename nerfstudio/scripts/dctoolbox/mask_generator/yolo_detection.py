import argparse
import os
from contextlib import redirect_stdout, contextmanager
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


@contextmanager
def suppress():
    with open(os.devnull, "w") as null:
        with redirect_stdout(null):
            yield


def predict_given_image_path(model, img_path: Path, margin=0.2) -> np.ndarray:
    with suppress():
        result, *_ = model(img_path.as_posix())  # predict on an image

    # Process results list
    boxes = result.boxes  # Boxes object for bounding box outputs
    pred_classes = boxes.cls
    probs = boxes.conf
    person_boxes = [
        boxes.xywh[i]
        for i in range(len(pred_classes))
        if pred_classes[i] == 0 and probs[i] > 0.75
    ]

    mask = np.zeros((result.orig_img.shape[0], result.orig_img.shape[1]))
    for cur_box in person_boxes:
        height_begin = int(cur_box[1] - cur_box[3] / 2 - margin * cur_box[3])
        height_begin = max(0, height_begin)
        height_end = int(cur_box[1] + cur_box[3] / 2 + margin * cur_box[3])
        height_end = min(mask.shape[0], height_end)
        width_begin = int(cur_box[0] - cur_box[2] / 2 - margin * cur_box[2])
        width_begin = max(0, width_begin)
        width_end = int(cur_box[0] + cur_box[2] / 2 + margin * cur_box[2])
        width_end = min(mask.shape[1], width_end)
        mask[height_begin:height_end, width_begin:width_end] = 255

    return mask.astype(np.uint8)


def get_args():
    parser = argparse.ArgumentParser("Prepare masks for dataset")
    parser.add_argument(
        "--image-dir", type=str, required=True, help="Path to the Image folder"
    )
    parser.add_argument(
        "--mask-dir", type=str, required=True, help="Path to the mask folder"
    )
    return parser.parse_args()


def main():
    args = get_args()
    image_dir = Path(args.image_dir)
    assert image_dir.exists(), f"Image directory {image_dir} does not exist"
    save_dir = Path(args.mask_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Load a model
    model = YOLO(
        "/home/jizong/Workspace/dConstruct/taichi-splatting/checkpoint/yolov8x.pt"
    )  # load a pretrained model (recommended for training)

    for f in tqdm(sorted(image_dir.rglob("*.png"))):
        mask = predict_given_image_path(model, f)
        Image.fromarray(mask).save(
            Path(args.mask_dir, f.stem + ".png").as_posix(),
        )


if __name__ == "__main__":
    main()
