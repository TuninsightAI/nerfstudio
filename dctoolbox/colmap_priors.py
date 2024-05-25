import json
import numpy as np
import os
import rich
import torch
from dataclasses import dataclass
from pathlib import Path

from dctoolbox.utils import quat2rotation, rotation2quat

# opencv convention to robotics convention
S = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)


@dataclass(slots=True)
class Image:
    # [qw, qx, qy, qz, px, py, pz, cam_no, v]
    qw: float
    qx: float
    qy: float
    qz: float
    px: float
    py: float
    pz: float
    cam_no: int
    image_name: str

    def tolist(self):
        return [
            self.qw,
            self.qx,
            self.qy,
            self.qz,
            self.px,
            self.py,
            self.pz,
            self.cam_no,
            self.image_name,
        ]


@dataclass
class ColmapPriorConfig:
    meta_json: Path
    """path to the meta json file"""
    output_folder: Path
    """path to the reference folder"""
    image_dir: Path | None = None
    """image-dir, to check if the image name and the prior name corresponds"""
    image_extension: str = "png"

    def __post_init__(self):

        assert self.meta_json.exists(), self.meta_json
        self.output_folder.mkdir(exist_ok=True, parents=True)

        if self.image_dir is not None:
            assert self.image_dir.exists(), self.image_dir

    def main(self):
        with open(self.meta_json) as f:
            data = json.load(f)

        camCalib_ = data["calibrationInfo"]
        data = data["data"]
        cc = {}
        observations = []
        cam_ids = []
        for camera_id, camCalib in camCalib_.items():
            cx = camCalib["intrinsics"]["camera_matrix"][2]
            cy = camCalib["intrinsics"]["camera_matrix"][5]
            fx = camCalib["intrinsics"]["camera_matrix"][0]
            fy = camCalib["intrinsics"]["camera_matrix"][4]
            camera_intrinsics = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
            camera_width = camCalib["intrinsics"]["width"]
            camera_height = camCalib["intrinsics"]["height"]
            camera_distortion = camCalib["intrinsics"]["distortion_coeffs"]
            cc[camera_id] = (
                camera_intrinsics,
                camera_width,
                camera_height,
                camera_distortion,
            )
            cam_ids.append(camera_id)

        for i, d in enumerate(data):
            image_path_ = d["imgName"]
            for k, v in image_path_.items():
                px = d["worldTcam"][k]["px"]
                py = d["worldTcam"][k]["py"]
                pz = d["worldTcam"][k]["pz"]
                qw = d["worldTcam"][k]["qw"]
                qx = d["worldTcam"][k]["qx"]
                qy = d["worldTcam"][k]["qy"]
                qz = d["worldTcam"][k]["qz"]
                R = np.zeros((4, 4))
                qvec = np.array([qw, qx, qy, qz])
                norm = np.linalg.norm(qvec)
                qvec /= norm
                R[:3, :3] = quat2rotation(torch.from_numpy(qvec).float()[None, ...])[0]
                R[:3, 3] = np.array([px, py, pz])
                R[3, 3] = 1.0
                # assert np.linalg.det(R[:3, :3]) == 1
                # todo: check if normalized. If not, normalize it.
                # Q, T = SE3_to_quaternion_and_translation_torch(torch.from_numpy(R).unsqueeze(0).double())
                # assert torch.allclose(Q.float(), torch.from_numpy(qvec).float(), rtol=1e-3, atol=1e-3), (
                #     Q.float(), torch.from_numpy(qvec).float())
                # assert torch.allclose(T.float(), torch.from_numpy([px, py, pz]).float(), rtol=1e-3, atol=1e-3)
                # here the world coordinate is defined in robotics space, where z is up, x is left and y is right.
                # the camera coordinate is defined in opencv convention, where the camera is looking down the z axis,
                # y is down and x is right.

                # convert the world coordinate to camera coordinate.
                R = S.T.dot(R)  # this is the c2w in opencv convention.

                R_ = np.linalg.inv(R)  # this is the w2c in opencv convention.
                # fixme: do we need this?

                R_ = torch.tensor(R_)
                T = R_[:3, 3]
                Q = rotation2quat(R_[None, ...])[0]  # this is the w2c
                qw, qx, qy, qz = Q.numpy().flatten().tolist()
                px, py, pz = T.numpy().flatten().tolist()

                cam_no = cam_ids.index(k) + 1
                # v = (Path(v.split("_")[1])/v).as_posix()
                observations.append(Image(qw, qx, qy, qz, px, py, pz, cam_no, v))

        # check if the image and the extracted prior the same.
        if self.image_dir is not None:
            image_names = [
                str(x.relative_to(self.image_dir))
                for x in self.image_dir.rglob(f"*.{self.image_extension}")
            ]
            previous_observation_length = len(observations)

            observations = [x for x in observations if x.image_name in image_names]

            cur_observation_length = len(observations)

            if previous_observation_length == cur_observation_length:
                rich.print(f"Prior images match with images in the folder")
            elif previous_observation_length > cur_observation_length:
                rich.print(
                    f"Prune observations from {previous_observation_length} to {cur_observation_length}, "
                    f"due to inconsistency of prior and provided images."
                )
            else:
                raise RuntimeError(f"provided images are greater than the prior.")

        if Path(f"{self.output_folder}/points3D.txt").exists():
            os.remove(f"{self.output_folder}/points3D.txt")
        Path(f"{self.output_folder}/points3D.txt").touch()

        if Path(f"{self.output_folder}/images.txt").exists():
            os.remove(f"{self.output_folder}/images.txt")
        with open(f"{self.output_folder}/images.txt", "w") as f:
            for i, o in enumerate(observations):
                f.write(" ".join([str(v) for v in [i + 1] + o.tolist()]))
                f.write("\n\n")

        if Path(f"{self.output_folder}/cameras.txt").exists():
            os.remove(f"{self.output_folder}/cameras.txt")

        with open(f"{self.output_folder}/cameras.txt", "w") as f:
            for i, k in enumerate(cam_ids):
                # 1 PINHOLE 1280 720 552.0972804534814 556.6976050663338 656.1668309733055 359.7676882310413
                intrinsics = cc[k][0]
                params = [
                    i + 1,
                    "PINHOLE",
                    cc[k][1],
                    cc[k][2],
                    intrinsics[0][0],
                    intrinsics[1][1],
                    intrinsics[0][2],
                    intrinsics[1][2],
                ]
                f.write(" ".join([str(v) for v in params]) + "\n")
