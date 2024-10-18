from __future__ import annotations

import json
import typing as t
import warnings
from dataclasses import dataclass
from multiprocessing.dummy import Pool
from pathlib import Path

import cv2
import numpy as np
from jaxtyping import Float
from loguru import logger
from tqdm import tqdm

from dctoolbox.double_sphere_camera.ds_camera import DSCamera


@dataclass(kw_only=True)
class UndistortConfig:
    input_dir: Path
    output_dir: Path
    output_mask_dir: Path | None = None
    cam_json: Path
    image_extension: str = "png"
    key_frame_list: list[str] = None
    enlarge_factor: float = 1.0

    def __post_init__(self):
        assert self.input_dir.exists(), self.input_dir
        assert self.cam_json.exists(), self.cam_json
        self.output_dir.mkdir(exist_ok=True, parents=True)
        if self.output_mask_dir is not None:
            self.output_mask_dir.mkdir(exist_ok=True, parents=True)

    def process_image(
        self, filename: Path, K: Float[np.ndarray, "3 3"], D: Float[np.ndarray, "4"]
    ) -> None:
        relative_path = str(filename.relative_to(self.input_dir))
        if self.key_frame_list is not None and relative_path not in self.key_frame_list:
            return

        img = cv2.imread(str(filename))

        # Get the dimensions of the image
        height, width = img.shape[:2]

        # new height and width given the enlarge factor
        # height = int(height * self.enlarge_factor)
        # width = int(width * self.enlarge_factor)

        # new camera matrix given the enlarge factor
        # this is to keep the image size to be the same.
        new_K = K.copy()
        new_K[0, 0] /= self.enlarge_factor
        new_K[1, 1] /= self.enlarge_factor
        # new_K[0, 2] /= self.enlarge_factor
        # new_K[1, 2] /= self.enlarge_factor

        # Undistort the image using the fisheye model
        undistorted_img = cv2.fisheye.undistortImage(
            img, K, D, Knew=new_K, new_size=(width, height)
        )

        output_path = self.output_dir / filename.relative_to(self.input_dir)
        # Save the undistorted image
        cv2.imwrite(output_path.as_posix(), undistorted_img)
        if self.output_mask_dir is not None:
            undistorted_mask = cv2.fisheye.undistortImage(
                np.ones_like(img), K, D, Knew=new_K, new_size=(width, height)
            )
            output_path = self.output_mask_dir / filename.relative_to(self.input_dir)
            # add an extension of .png
            output_path = output_path.parent / (output_path.name + ".png")
            # Save the undistorted image
            cv2.imwrite(
                output_path.as_posix(), ((1 - undistorted_mask) * 255).astype(np.uint8)
            )

    def main(self) -> Float[np.ndarray, "3 3"]:
        with open(self.cam_json, "r") as file:
            data = json.load(file)
            # Load the camera matrix and distortion coefficients from the file
        K = np.array(
            data["calibration"]["intrinsics"]["camera_matrix"], dtype=np.float64
        ).reshape((3, 3))
        D = np.array(
            data["calibration"]["intrinsics"]["distortion_coeffs"], dtype=np.float64
        )

        images = sorted(self.input_dir.glob(f"*.{self.image_extension}"))
        with Pool(32) as pool:
            workers = pool.imap_unordered(
                lambda filename: self.process_image(filename, K, D), images
            )
            for _ in tqdm(workers, total=len(images), desc="Processing images"):
                pass

        new_K = K.copy()
        new_K[0, 0] /= self.enlarge_factor
        new_K[1, 1] /= self.enlarge_factor
        return new_K


@dataclass(kw_only=True)
class UndistortDoubleSphereConfig(UndistortConfig):
    def main(self) -> Float[np.ndarray, "3 3"]:
        with open(self.cam_json, "r") as file:
            data = json.load(file)
        # in double sphere camera, the K has 6 parameters
        K = np.array(
            data["calibration"]["intrinsics"]["camera_matrix"], dtype=np.float64
        ).reshape(6)

        images = sorted(self.input_dir.glob(f"*.{self.image_extension}"))
        with Pool(32) as pool:
            workers = pool.imap_unordered(
                lambda filename: self.process_image(filename, K), images
            )
            for _ in tqdm(workers, total=len(images), desc="Processing images"):
                pass

        new_K = np.eye(3)
        new_K[0, 0] = K[2] / self.enlarge_factor
        new_K[1, 1] = K[3] / self.enlarge_factor
        new_K[0, 2] = K[4] / self.enlarge_factor
        new_K[1, 2] = K[5] / self.enlarge_factor
        return new_K

    def process_image(
        self, filename: Path, K: Float[np.ndarray, "6"], **kwargs
    ) -> None:
        relative_path = str(filename.relative_to(self.input_dir))
        if self.key_frame_list is not None and relative_path not in self.key_frame_list:
            return

        img = cv2.imread(str(filename))

        # Get the dimensions of the image
        height, width = img.shape[:2]

        camera = DSCamera(
            fx=float(K[2]),
            fy=float(K[3]),
            cx=float(K[4]),
            cy=float(K[5]),
            xi=float(K[0]),
            alpha=float(K[1]),
            height=height,
            width=width,
        )
        undistorted_img = camera.to_perspective(img, zoom_factor=self.enlarge_factor)

        output_path = self.output_dir / filename.relative_to(self.input_dir)
        # Save the undistorted image
        cv2.imwrite(output_path.as_posix(), undistorted_img)
        if self.output_mask_dir is not None:
            undistorted_mask = camera.to_perspective(
                np.ones_like(img), zoom_factor=self.enlarge_factor
            )
            output_path = self.output_mask_dir / filename.relative_to(self.input_dir)
            # add an extension of .png
            output_path = output_path.parent / (output_path.name + ".png")
            # Save the undistorted image
            cv2.imwrite(
                output_path.as_posix(), ((1 - undistorted_mask) * 255).astype(np.uint8)
            )


def _iterate_camera(
    data_frames: t.List[t.Dict[str, t.Any]], camera_name: str
) -> t.List[str]:
    camera_list = []
    for cur_frame in data_frames:
        camera_list.append(Path(cur_frame["imgName"][camera_name]).name)
    return camera_list


def undistort_folder(
    *,
    input_dir: Path,
    output_dir: Path,
    output_mask_dir: Path | None = None,
    image_extension: str = "png",
    converted_meta_json_path: Path,
    enlarge_factor: float = 1,
):
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
    with open(slam_meta, "r") as file:
        camera_folders = json.load(file)["camSerial"]
        # Load the camera matrix and distortion coefficients from the file
    new_camera_intrinsic = {}
    for camera_name in camera_folders:
        camera_folder_path = input_dir / camera_name
        logger.info(f"Processing {camera_folder_path}")
        camera_json_path = input_dir / camera_name / "camMeta.json"
        key_frame_list = None
        if converted_meta_json_path is not None:
            with open(converted_meta_json_path, "r") as file:
                data = json.load(file)
            key_frame_list = _iterate_camera(data["data"], camera_name)
            # logger.info(f"Key frame list: {key_frame_list[:5]}")
            # Load the camera matrix and distortion coefficients from the file
        with open(camera_json_path, "r") as _cam_config:
            camera_type = json.load(_cam_config)["calibration"]["intrinsics"][
                "camera_type"
            ]
        assert camera_type in ["OpenCVFisheye", "KalibrDoubleSphere"], camera_type

        if camera_type == "KalibrDoubleSphere":
            logger.info("Using double sphere camera")
            new_K = UndistortDoubleSphereConfig(
                input_dir=camera_folder_path,
                output_dir=output_dir / camera_name,
                output_mask_dir=output_mask_dir / camera_name
                if output_mask_dir
                else None,
                cam_json=camera_json_path,
                image_extension=image_extension,
                key_frame_list=key_frame_list,
                enlarge_factor=enlarge_factor,
            ).main()
        elif camera_type == "OpenCVFisheye":
            new_K = UndistortConfig(
                input_dir=camera_folder_path,
                output_dir=output_dir / camera_name,
                output_mask_dir=output_mask_dir / camera_name
                if output_mask_dir
                else None,
                cam_json=camera_json_path,
                image_extension=image_extension,
                key_frame_list=key_frame_list,
                enlarge_factor=enlarge_factor,
            ).main()
        else:
            raise ValueError(f"Unknown camera type: {camera_type}")
        new_camera_intrinsic[camera_name] = new_K.tolist()
    return new_camera_intrinsic


def update_meta_json(meta_path: Path, enlarge_factor: float):
    warnings.warn("This function is deprecated", DeprecationWarning)
    assert meta_path.exists(), meta_path
    with open(meta_path, "r") as file:
        _meta_data = json.load(file)

    calibration_info = _meta_data["calibrationInfo"]
    for camera_name, camera_info in calibration_info.items():
        intrinsic = camera_info["intrinsics"]["camera_matrix"]
        intrinsic[0] /= enlarge_factor
        intrinsic[4] /= enlarge_factor
        camera_info["intrinsics"]["camera_matrix"] = intrinsic

    with open(meta_path, "w") as file:
        json.dump(_meta_data, file, indent=4)
