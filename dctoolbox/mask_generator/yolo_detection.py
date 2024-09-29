import numpy as np
import os
import typing as t
import tyro
from PIL import Image
from contextlib import redirect_stdout, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


@contextmanager
def suppress():
    with open(os.devnull, "w") as null:
        with redirect_stdout(null):
            yield


def predict_given_image_path(model, img_path: Path, margin=0.2) -> np.ndarray:
    with suppress():
        result, *_ = model(img_path.as_posix(),  verbose=False)  # predict on an image

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


@dataclass
class YoloMaskGeneratorConfig:
    image_dir: Path
    """ Path to the Image folder """

    mask_dir: Path
    """ Path to the mask folder"""

    extension: t.Literal[".png", ".jpg", ".jpeg"] = ".jpeg"

    checkpoint_path: Path = Path(
        "/home/jizong/Workspace/dConstruct/taichi-splatting/checkpoint/yolov8x.pt"
    )

    def main(self):
        assert (
            self.image_dir.exists()
        ), f"Image directory {self.image_dir} does not exist"
        Path(self.mask_dir).mkdir(parents=True, exist_ok=True)

        # Load a model
        model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

        for f in tqdm(
            sorted(self.image_dir.rglob(f"*{self.extension}")), desc="Generating masks"
        ):
            mask = predict_given_image_path(model, f)
            relative_path = f.relative_to(self.image_dir)
            save_path = Path(self.mask_dir, str(relative_path) + ".png")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(save_path.as_posix())


if __name__ == "__main__":
    tyro.cli(YoloMaskGeneratorConfig).main()
