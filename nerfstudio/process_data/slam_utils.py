from typing import Dict, Any, List

import numpy as np

from nerfstudio.data.utils.colmap_parsing_utils import qvec2rotmat, rotmat2qvec, Image
from nerfstudio.process_data.process_data_utils import CameraModel

opencv2slam_transform = np.array(
    [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
)


def parse_slam_camera_params(camera_name: str, camera_params: dict):
    """
    "DECXIN20230102350",
     {
            "camSerial": "DECXIN20230102350",
            "intrinsics": {
                "camera_matrix": [
                    560.1595530518094,
                    0,
                    659.5430672762923,
                    0,
                    559.8752636760909,
                    354.5327242397341,
                    0,
                    0,
                    1
                ],
                "height": 720,
                "width": 1280,
                "distortion_coeffs": [
                    -0.034382209383996706,
                    -0.00684441861628481,
                    0.0014449054099189363,
                    -0.0005730249990893673
                ]
            },
            "qw": -0.26662394404411316,
            "qx": 0.26330623030662537,
            "qy": -0.6517673134803772,
            "qz": 0.6593791842460632,
            "x": 0.06962212920188904,
            "y": -0.02780185081064701,
            "z": -0.059692323207855225
        },


            camera_detail["intrinsics"]["camera_matrix"][0],
            camera_detail["intrinsics"]["camera_matrix"][4],
            camera_detail["intrinsics"]["camera_matrix"][2],
            camera_detail["intrinsics"]["camera_matrix"][5],
    """
    out: Dict[str, Any] = {
        "w": camera_params["intrinsics"]["width"],
        "h": camera_params["intrinsics"]["height"],
    }

    # Parameters match https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h

    # f, cx, cy, k

    # du = 0
    # dv = 0
    intrinsics = camera_params["intrinsics"]["camera_matrix"]
    out["fl_x"] = float(camera_params[0])
    out["fl_y"] = float(camera_params[4])
    out["cx"] = float(camera_params[2])
    out["cy"] = float(camera_params[5])
    out["k1"] = 0.0
    out["k2"] = 0.0
    out["p1"] = 0.0
    out["p2"] = 0.0
    camera_model = CameraModel.OPENCV
    out["camera_model"] = camera_model.value
    out["camera_name"] = camera_name
    return out


def iterate_word2cam_matrix(data_frames: List[Dict[str, Any]]):

    for cur_frame in data_frames:
        for cur_camera_name, cur_c2w in cur_frame["worldTcam"].items():
            if cur_camera_name in cur_frame["imgName"]:
                yield cur_frame["imgName"][cur_camera_name], cur_camera_name, cur_c2w


def parse_num2image(data_frame, camera_info):
    """
    {
            "imgName": {
                "DECXIN20230102350": "DECXIN20230102350/0000000000.jpeg",
                "DECXIN2023012346": "DECXIN2023012346/0000000000.jpeg",
                "DECXIN2023012347": "DECXIN2023012347/0000000000.jpeg",
                "DECXIN2023012348": "DECXIN2023012348/0000000000.jpeg"
            },
            "worldTcam": {
                "DECXIN20230102350": {
                    "qw": 0.0532809956527241,
                    "qx": -0.46568308702888217,
                    "qy": 0.6802059383852321,
                    "qz": -0.5635781040308177,
                    "px": 0.022161192726281396,
                    "py": -0.03049275291462227,
                    "pz": -0.08926902555207991
                },
                "DECXIN2023012346": {
                    "qw": 0.530153100163243,
                    "qx": -0.723784425260518,
                    "qy": 0.4374535509154571,
                    "qz": -0.060894884260184946,
                    "px": -0.012152331963738834,
                    "py": 0.06030197575795748,
                    "pz": -0.09436431824704344
                },
                "DECXIN2023012347": {
                    "qw": 0.43366172120045116,
                    "qx": -0.05821288618345082,
                    "qy": -0.5324410800508882,
                    "qz": 0.7246069746572266,
                    "px": -0.0749910828113337,
                    "py": -0.06715333212038976,
                    "pz": -0.04122713212493902
                },
                "DECXIN2023012348": {
                    "qw": 0.6817843810912588,
                    "qx": -0.5590663638927561,
                    "qy": -0.0660128926813958,
                    "qz": 0.4671800043490475,
                    "px": -0.09980893709039859,
                    "py": 0.04316303074896288,
                    "pz": -0.03101325036236975
                }
            }
        },
    """
    images = {}
    for image_id, cur_frame in enumerate(iterate_word2cam_matrix(data_frame)):
        image_name, camera_name, c2w = cur_frame
        c2w = {k: float(v) for k, v in c2w.items()}

        c2w_matrix = np.eye(4)
        qvec = np.array([c2w["qw"], c2w["qx"], c2w["qy"], c2w["qz"]])
        qvec /= np.linalg.norm(qvec)
        c2w_matrix[:3, :3] = qvec2rotmat(qvec)
        c2w_matrix[:3, 3] = np.array([c2w["px"], c2w["py"], c2w["pz"]])
        # slam to opencv
        c2w_matrix = opencv2slam_transform.T.dot(
            c2w_matrix
        )  # this is the c2w in opencv convention.
        w2c_matrix = np.linalg.inv(c2w_matrix)

        qvec = rotmat2qvec(w2c_matrix[:3, :3])
        qvec /= np.linalg.norm(qvec)
        tvec = w2c_matrix[:3, 3]

        camera_id = 1
        for cur_camera_id, camera_params in camera_info.items():
            if camera_params["camera_name"] == camera_name:
                camera_id = cur_camera_id
                break

        images[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=image_name,
            xys=None,
            point3D_ids=None,
        )

    return images
