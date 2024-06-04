from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm
import typing as t


@dataclass
class UndistortConfig:
    input_dir: Path
    output_dir: Path
    cam_json: Path
    image_extension: str = "png"
    key_frame_list: list[str] = None

    def __post_init__(self):
        assert self.input_dir.exists(), self.input_dir
        assert self.cam_json.exists(), self.cam_json
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def main(self):
        with open(self.cam_json, 'r') as file:
            data = json.load(file)
            # Load the camera matrix and distortion coefficients from the file
        K = np.array(data['calibration']['intrinsics']['camera_matrix'], dtype=np.float64).reshape((3, 3))
        D = np.array(data['calibration']['intrinsics']['distortion_coeffs'], dtype=np.float64)

        for filename in tqdm(sorted(self.input_dir.glob(f"*.{self.image_extension}")), desc="Processing images"):
            # Read the image
            relative_path = str(filename.relative_to(self.input_dir))
            if self.key_frame_list is not None and relative_path not in self.key_frame_list:
                continue

            img = cv2.imread(str(filename))

            # Get the dimensions of the image
            height, width = img.shape[:2]

            # Calculate the optimal new camera matrix
            # new_camera_matrix, _ = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            #     K, D, (width, height), np.eye(3), balance=0)

            # Undistort the image using the fisheye model
            undistorted_img = cv2.fisheye.undistortImage(
                img, K, D, Knew=K, new_size=(width, height))
            output_path = self.output_dir / filename.relative_to(self.input_dir)
            # Save the undistorted image
            cv2.imwrite(output_path.as_posix(), undistorted_img)


def _iterate_camera(data_frames: t.List[t.Dict[str, t.Any]], camera_name: str) -> t.List[str]:
    camera_list = []
    for cur_frame in data_frames:
        camera_list.append(Path(cur_frame["imgName"][camera_name]).name)
    return camera_list


def undistort_folder(input_dir: Path, output_dir: Path, image_extension: str = "png",
                     converted_meta_json_path: Path | None = None):
    """
    .
    ├── DECXIN20230102350
    ├── DECXIN2023012346
    ├── DECXIN2023012347
    ├── DECXIN2023012348
    ├── LiDAR-122322001000
    └── slamMeta.json
    """
    slam_meta = input_dir / "slamMeta.json"
    with open(slam_meta, 'r') as file:
        camera_folders = json.load(file)["camSerial"]
        # Load the camera matrix and distortion coefficients from the file

    for camera_name in camera_folders:
        camera_folder_path = input_dir / camera_name
        logger.info(f"Processing {camera_folder_path}")
        camera_json_path = input_dir / camera_name / "camMeta.json"
        key_frame_list = None
        if converted_meta_json_path is not None:
            with open(converted_meta_json_path, 'r') as file:
                data = json.load(file)
            key_frame_list = _iterate_camera(data["data"], camera_name)
            logger.info(f"Key frame list: {key_frame_list[:5]}")
            # Load the camera matrix and distortion coefficients from the file

        UndistortConfig(camera_folder_path, output_dir / camera_name, cam_json=camera_json_path,
                        image_extension=image_extension, key_frame_list=key_frame_list).main()
