# create some interface adaptors for the different types of interfaces
from __future__ import annotations

import json
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rich
import tyro
from tqdm import tqdm

from dctoolbox.pose_evals.interpolate_poses import LiDARDataInterpolator
from nerfstudio.data.utils.colmap_parsing_utils import qvec2rotmat, rotmat2qvec


@dataclass(slots=True, unsafe_hash=True)
class Extrinsic:
    qx: float
    qy: float
    qz: float
    qw: float
    px: float
    py: float
    pz: float
    name: str | None = None
    camera_name: str | None = None
    stem: str | None = None

    def as_array(self, format_: t.Literal["wxyz", "xyzw"]) -> np.ndarray:
        if format_ == "wxyz":
            return np.array(
                [self.qw, self.qx, self.qy, self.qz, self.px, self.py, self.pz]
            )
        elif format_ == "xyzw":
            return np.array(
                [self.px, self.py, self.pz, self.qw, self.qx, self.qy, self.qz]
            )
        raise ValueError(f"format_ must be 'wxyz' or 'xyzw' not {format_}")

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, image_path: str, camera_name: str):
        qvec = rotmat2qvec(matrix[:3, :3])
        return cls(
            qx=float(qvec[1]),
            qy=float(qvec[2]),
            qz=float(qvec[3]),
            qw=float(qvec[0]),
            px=float(matrix[0, 3]),
            py=float(matrix[1, 3]),
            pz=float(matrix[2, 3]),
            name=image_path,
            camera_name=camera_name,
            stem=Path(image_path).stem,
        )

    def to_matrix(self, format_: t.Literal["3x4", "4x4"] = "4x4"):
        if format_ == "3x4":
            matrix = np.zeros((3, 4))
            qvec = np.array([self.qw, self.qx, self.qy, self.qz])
            qvec /= np.linalg.norm(qvec)
            matrix[:3, :3] = qvec2rotmat(qvec)
            matrix[:3, 3] = np.array([self.px, self.py, self.pz])

            return matrix
        elif format_ == "4x4":
            matrix = np.eye(4)
            qvec = np.array([self.qw, self.qx, self.qy, self.qz])
            qvec /= np.linalg.norm(qvec)
            matrix[:3, :3] = qvec2rotmat(qvec)
            matrix[:3, 3] = np.array([self.px, self.py, self.pz])
            return matrix
        else:
            raise ValueError(f"format_ must be '3x4' or '4x4' not {format_}")


@dataclass(kw_only=True)
class CameraInfo:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    distortion_coeffs: list[float] | None = None
    name: str | None = None

    extrinsic: t.Optional[Extrinsic] = None

    dataframe: pd.DataFrame | None = None
    camera_type: t.Literal["KalibrDoubleSphere", "OpenCVFisheye"]


def read_camera_info(camera_json_path: Path) -> CameraInfo:
    with camera_json_path.open("r") as f:
        cur_camera_json = json.load(f)
    try:
        name = cur_camera_json["calibration"]["cam_serial"]
    except KeyError:
        name = cur_camera_json["calibration"]["camSerial"]

    camera_type = cur_camera_json["calibration"]["intrinsics"]["camera_type"]
    assert camera_type in ["KalibrDoubleSphere", "OpenCVFisheye"]
    if camera_type == "openCVFisheye":
        cur_camera_info = CameraInfo(
            fx=cur_camera_json["calibration"]["intrinsics"]["camera_matrix"][0],
            fy=cur_camera_json["calibration"]["intrinsics"]["camera_matrix"][4],
            cx=cur_camera_json["calibration"]["intrinsics"]["camera_matrix"][2],
            cy=cur_camera_json["calibration"]["intrinsics"]["camera_matrix"][5],
            width=cur_camera_json["calibration"]["intrinsics"]["width"],
            height=cur_camera_json["calibration"]["intrinsics"]["height"],
            name=name,
            distortion_coeffs=cur_camera_json["calibration"]["intrinsics"][
                "distortion_coeffs"
            ],
            camera_type=camera_type,
        )
    elif camera_type == "KalibrDoubleSphere":
        cur_camera_info = CameraInfo(
            fx=cur_camera_json["calibration"]["intrinsics"]["camera_matrix"][2],
            fy=cur_camera_json["calibration"]["intrinsics"]["camera_matrix"][3],
            cx=cur_camera_json["calibration"]["intrinsics"]["camera_matrix"][4],
            cy=cur_camera_json["calibration"]["intrinsics"]["camera_matrix"][5],
            width=cur_camera_json["calibration"]["intrinsics"]["width"],
            height=cur_camera_json["calibration"]["intrinsics"]["height"],
            name=name,
            distortion_coeffs=cur_camera_json["calibration"]["intrinsics"][
                "distortion_coeffs"
            ],
            camera_type=camera_type,
        )
    else:
        raise TypeError(f"camera_type {camera_type} not supported")
    try:
        cur_camera_info.extrinsic = Extrinsic(
            qx=cur_camera_json["calibration"]["extrinsics"]["qx"],
            qy=cur_camera_json["calibration"]["extrinsics"]["qy"],
            qz=cur_camera_json["calibration"]["extrinsics"]["qz"],
            qw=cur_camera_json["calibration"]["extrinsics"]["qw"],
            px=cur_camera_json["calibration"]["extrinsics"]["px"],
            py=cur_camera_json["calibration"]["extrinsics"]["py"],
            pz=cur_camera_json["calibration"]["extrinsics"]["pz"],
            camera_name=cur_camera_json["calibration"]["cam_serial"],
        )
    except KeyError:
        cur_camera_info.extrinsic = Extrinsic(
            qx=cur_camera_json["calibration"]["qx"],
            qy=cur_camera_json["calibration"]["qy"],
            qz=cur_camera_json["calibration"]["qz"],
            qw=cur_camera_json["calibration"]["qw"],
            px=cur_camera_json["calibration"]["x"],
            py=cur_camera_json["calibration"]["y"],
            pz=cur_camera_json["calibration"]["z"],
            camera_name=cur_camera_json["calibration"]["camSerial"],
        )
    cur_camera_info.dataframe = pd.DataFrame(cur_camera_json["data"]).drop(
        ["dataID"], axis=1
    )
    return cur_camera_info


def read_lidar_info(lidar_json_path: Path) -> pd.DataFrame:
    with lidar_json_path.open("r") as f:
        lidar_json = json.load(f)

    dataframe = pd.DataFrame(lidar_json["data"])
    dataframe = dataframe[
        (dataframe["isKeyframe"] == True) & (dataframe["hasImg"] == True)
    ]
    
    def create_rot(row):
        qvec = np.array([row["qw"], row["qx"], row["qy"], row["qz"]])
        qvec /= np.linalg.norm(qvec)
        return qvec2rotmat(qvec)

    def create_trans(row):
        return np.array([row["px"], row["py"], row["pz"]])

    def create_extrinsic(row):
        return np.concatenate(
            [create_rot(row), create_trans(row).reshape(-1, 1)], axis=1
        ).tolist()

    dataframe["extrinsics"] = dataframe["tf"].apply(create_extrinsic)
    del dataframe["tf"]
    dataframe["nearestImgs"] = dataframe["nearestImgs"].apply(
        lambda x: x[0].split("/")[-1]
    )
    return dataframe


def create_camera_pose(
    camerainfo: CameraInfo,
    lidar_frame: pd.DataFrame,
    lidar_interpolator: LiDARDataInterpolator | None = None,
) -> t.List[Extrinsic]:
    """
    return a list of c2w extrinsics for each image in the lidar frame
    :param camerainfo:
    :param lidar_frame:
    :return:
    """
    image_extrinsic = lidar_frame[["nearestImgs", "extrinsics"]]
    image_extrinsic = image_extrinsic.merge(
        camerainfo.dataframe, left_on="nearestImgs", right_on="filename", how="inner"
    )
    camera_c2o = camerainfo.extrinsic.to_matrix("4x4")

    c2ws = []
    pose_c2ws = []
    new_extrinsic = []
    for row_id, cur_image in image_extrinsic.iterrows():

        # find the new interpolated pose
        if lidar_interpolator is not None:
            lidar_timestamp = cur_image["timestamp"]
            lidar_pose = lidar_interpolator.get_interpolated_position(lidar_timestamp)
            lidar_quat = lidar_interpolator.get_interpolated_quaternion(lidar_timestamp)
            lidar_rot = qvec2rotmat(lidar_quat)
            c2w = np.concatenate([lidar_rot, lidar_pose.reshape(-1, 1)], axis=1)
        else:
            c2w = np.array(cur_image["extrinsics"])

        c2w = np.concatenate([c2w, np.array([[0, 0, 0, 1]])], axis=0)

        pose_c2w = np.dot(c2w, camera_c2o)
        c2ws.append(c2w)
        pose_c2ws.append(pose_c2w)

        new_extrinsic.append(
            Extrinsic.from_matrix(
                pose_c2w,
                image_path=Path(camerainfo.name, cur_image["nearestImgs"]).as_posix(),
                camera_name=camerainfo.name,
            )
        )

    return new_extrinsic


@dataclass
class InterfaceAdaptorConfig:
    slam_json_path: Path

    output_path: Path
    
    num_cameras: int = field(init=False)    
    interpolate_poses: bool = False

    def __post_init__(self):
        assert self.slam_json_path.exists(), "slam_json_path must be provided"
        assert self.output_path.suffix == ".json", "output_path must be a json file"
        self.output_path.parent.mkdir(exist_ok=True, parents=True)

    def main(self):
        with self.slam_json_path.open("r") as f:
            slam_json = json.load(f)

        camera_folders: t.List[Path] = [
            self.slam_json_path.parent / x for x in slam_json["camSerial"]
        ]

        # Set number of cameras
        self.num_cameras = len(camera_folders)
        
        camera_json_paths: t.List[Path] = [x / "camMeta.json" for x in camera_folders]

        try:
            lidar_json_path: Path = (
                self.slam_json_path.parent
                / slam_json["mainSensorSerial"]
                / "scanMeta.json"
            )
        except KeyError:
            lidar_json_path: Path = self.slam_json_path.parent / slam_json["scanMeta"]

        lidar_info = read_lidar_info(lidar_json_path)

        camera_infos: t.List[CameraInfo] = [
            read_camera_info(x) for x in camera_json_paths
        ]
        image_extrinics: t.List[Extrinsic] = []

        for cur_camera_info in camera_infos:
            if self.interpolate_poses:
                lidar_interpolator = LiDARDataInterpolator(lidar_json_path)
                ext_given_cam = create_camera_pose(
                    cur_camera_info, lidar_info, lidar_interpolator
                )
            else:
                ext_given_cam = create_camera_pose(cur_camera_info, lidar_info)
            image_extrinics.extend(ext_given_cam)

        self._to_json(camera_infos, image_extrinics)

    def _to_json(
        self, camera_infos: t.List[CameraInfo], image_extrinsics: t.List[Extrinsic]
    ):

        calibrationInfo = dict()
        for cur_camera_info in camera_infos:
            calibrationInfo[cur_camera_info.name] = dict(
                camSerial=cur_camera_info.name,
                intrinsics=dict(
                    camera_matrix=[
                        cur_camera_info.fx,
                        0,
                        cur_camera_info.cx,
                        0,
                        cur_camera_info.fy,
                        cur_camera_info.cy,
                        0,
                        0,
                        1,
                    ],
                    height=cur_camera_info.height,
                    width=cur_camera_info.width,
                    distortion_coeffs=cur_camera_info.distortion_coeffs,
                    camera_type=cur_camera_info.camera_type,
                ),
                qw=cur_camera_info.extrinsic.qw,
                qx=cur_camera_info.extrinsic.qx,
                qy=cur_camera_info.extrinsic.qy,
                qz=cur_camera_info.extrinsic.qz,
                x=cur_camera_info.extrinsic.px,
                y=cur_camera_info.extrinsic.py,
                z=cur_camera_info.extrinsic.pz,
            )

        def iterate_per_timestamps():
            remaining_image_extrinsics = image_extrinsics.copy()

            timestamps = sorted(
                set([Path(x.name).relative_to(x.camera_name) for x in image_extrinsics])
            )
            for cur_timestamp in timestamps:
                batch = []
                for cur_image_extrinsic in remaining_image_extrinsics:
                    if (
                        Path(cur_image_extrinsic.name).relative_to(
                            cur_image_extrinsic.camera_name
                        )
                        == cur_timestamp
                    ):
                        batch.append(cur_image_extrinsic)
                    if len(batch) == 4:
                        break
                yield sorted(batch, key=lambda x: x.name)
                remaining_image_extrinsics = [
                    x
                    for x in remaining_image_extrinsics
                    if x.stem not in [y.stem for y in batch]
                ]

        data = []
        for timestamp_batch in tqdm(iterate_per_timestamps()):
            cur_data = dict()
            cur_data["imgName"] = {x.camera_name: x.name for x in timestamp_batch}
            cur_data["worldTcam"] = {
                x.camera_name: dict(
                    zip(
                        ["qw", "qx", "qy", "qz", "px", "py", "pz"],
                        x.as_array(format_="wxyz").tolist(),
                    )
                )
                for x in timestamp_batch
            }
            
            # only add the data if all cameras caputred an image at that timestamp
            if len(cur_data["imgName"]) == self.num_cameras:
                data.append(cur_data)

        with open(self.output_path, "w") as f:
            json.dump(
                dict(
                    calibrationInfo=calibrationInfo,
                    data=data,
                ),
                f,
                indent=4,
            )


def entrance_point():
    config = tyro.cli(tyro.conf.FlagConversionOff[InterfaceAdaptorConfig])
    rich.print(config)
    config.main()


if __name__ == "__main__":
    entrance_point()

    # output_dir = Path(
    #     "/Users/vaibhavholani/development/computer_vision/dConstruct/data/pixel_lvl1_water2_resampled/"
    # )
    # InterfaceAdaptorConfig(
    #     slam_json_path=output_dir / "raw" / "slamMeta.json",
    #     output_path=output_dir / "undistorted" / "meta.json",
    # ).main()
