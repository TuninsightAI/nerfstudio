import numpy as np
import sqlite3
import torch
from dataclasses import dataclass
from pathlib import Path

from dctoolbox.utils import quat2rotation


@dataclass
class ColmapPriorInjectionConfig:
    source_path: Path
    database_path: Path

    def __post_init__(self):
        assert self.source_path.exists(), self.source_path
        assert self.database_path.exists(), self.database_path

        sqlite3.register_adapter(np.array, lambda arr: arr.tobytes())

    def main(self):
        img_file = self.source_path / "images.txt"
        camera_file = self.source_path / "cameras.txt"

        assert img_file.exists()
        assert camera_file.exists()

        with open(img_file, "r") as f:
            images = [i.strip() for i in f.readlines()]
            images = [i.split(" ") for i in images if len(i) > 0]
        with open(camera_file, "r") as f:
            cameras = [i.strip().split(" ") for i in f.readlines()]

        # open database
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        # update each row
        for camera in cameras:
            idx, t, w, h, p0, p1, p2, p3 = camera
            params = [float(p) for p in [p0, p1, p2, p3]]
            arr = np.array(params, dtype="double")
            sql = """
            insert into cameras (camera_id, model, width, height, params, prior_focal_length)
            values (?, ?, ?, ?, ?,?)
            """
            cursor.execute(sql, (int(idx), 1, int(w), int(h), arr, 0))

        conn.commit()

        for image in images:
            # note that prior_qw, prior_qx, prior_qy, prior_qz are never used,
            # prior_tx, prior_ty, prior_tz camera center in world coordinate, that's saying the last column of c2w

            sql = """
                insert into images ( prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz, camera_id, name) 
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            (
                idx,
                qw,
                qx,
                qy,
                qz,
                px,
                py,
                pz,
                cam_no,
                name,
            ) = image  # these are elements for w2c matrix
            R = np.zeros((4, 4))
            R[:3, :3] = quat2rotation(
                torch.tensor([float(qw), float(qx), float(qy), float(qz)])[None, ...]
            )[0]
            R[:3, 3] = np.array([float(px), float(py), float(pz)])
            R[3, 3] = 1.0
            R = np.linalg.inv(R)  # this is the c2w matrix

            px, py, pz = R[
                :3, 3
            ].flatten()  # replace the px, py, pz by the camera center in world coordinate

            cursor.execute(
                sql,
                (
                    float(qw),
                    float(qx),
                    float(qy),
                    float(qz),
                    float(px),
                    float(py),
                    float(pz),
                    int(cam_no),
                    name,
                ),
            )
        conn.commit()
        conn.close()
