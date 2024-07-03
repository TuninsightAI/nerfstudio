import json
import numpy as np
from scipy.interpolate import CubicSpline
from nerfstudio.cameras.camera_utils import quaternion_slerp

class LiDARDataInterpolator:
    def __init__(self, lidar_file_path):
        self.lidar_data = self._load_lidar_data(lidar_file_path)
        self.lidar_timestamps, self.px, self.py, self.pz, self.qw, self.qx, self.qy, self.qz = self._extract_lidar_data(self.lidar_data)
        self.spline_px = CubicSpline(self.lidar_timestamps, self.px)
        self.spline_py = CubicSpline(self.lidar_timestamps, self.py)
        self.spline_pz = CubicSpline(self.lidar_timestamps, self.pz)

    def _load_lidar_data(self, file_path):
        with file_path.open("r") as f:
            return json.load(f)

    def _extract_lidar_data(self, lidar_data):
        timestamps = []
        px, py, pz = [], [], []
        qw, qx, qy, qz = [], [], [], []

        for entry in lidar_data['data']:
            if 'tf' in entry and 'timestamp' in entry:
                timestamps.append(entry['timestamp'])
                px.append(entry['tf']['px'])
                py.append(entry['tf']['py'])
                pz.append(entry['tf']['pz'])
                qw.append(entry['tf']['qw'])
                qx.append(entry['tf']['qx'])
                qy.append(entry['tf']['qy'])
                qz.append(entry['tf']['qz'])

        return (np.array(timestamps), np.array(px), np.array(py), np.array(pz), 
                np.array(qw), np.array(qx), np.array(qy), np.array(qz))

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



    def get_interpolated_position(self, timestamp):
        x = self.spline_px(timestamp)
        y = self.spline_py(timestamp)
        z = self.spline_pz(timestamp)
        return np.array([x, y, z])

    def get_interpolated_quaternion(self, timestamp):
        if timestamp in self.lidar_timestamps:
            idx = np.where(self.lidar_timestamps == timestamp)[0][0]
            return [self.qw[idx], self.qx[idx], self.qy[idx], self.qz[idx]]

        if timestamp < self.lidar_timestamps[0]:
            return [self.qw[0], self.qx[0], self.qy[0], self.qz[0]]
        elif timestamp > self.lidar_timestamps[-1]:
            return [self.qw[-1], self.qx[-1], self.qy[-1], self.qz[-1]]

        t1_idx = np.searchsorted(self.lidar_timestamps, timestamp, side='right') - 1
        t2_idx = t1_idx + 1

        t1 = self.lidar_timestamps[t1_idx]
        t2 = self.lidar_timestamps[t2_idx]

        if t1 == t2:
            return [self.qw[t1_idx], self.qx[t1_idx], self.qy[t1_idx], self.qz[t1_idx]]

        ratio = (timestamp - t1) / (t2 - t1)

        q1 = [self.qw[t1_idx], self.qx[t1_idx], self.qy[t1_idx], self.qz[t1_idx]]
        q2 = [self.qw[t2_idx], self.qx[t2_idx], self.qy[t2_idx], self.qz[t2_idx]]

        qvec = np.array(quaternion_slerp(q1, q2, ratio))
       
        qvec /= np.linalg.norm(qvec)
        return qvec

if __name__ == "__main__":
    lidar_file_path = '/Users/vaibhavholani/development/computer_vision/dConstruct/data/pixel_lvl1_water2_resampled/raw/LiDAR-122322001000/scanMeta.json'
    lidar_interpolator = LiDARDataInterpolator(lidar_file_path)

    example_timestamp = 798121407  # Replace with your desired timestamp
    interpolated_position = lidar_interpolator.get_interpolated_position(example_timestamp)
    interpolated_quaternion = lidar_interpolator.get_interpolated_quaternion(example_timestamp)

    print(f"Interpolated position at timestamp {example_timestamp}: x={interpolated_position[0]}, y={interpolated_position[1]}, z={interpolated_position[2]}")
    print(f"Interpolated quaternion at timestamp {example_timestamp}: qw={interpolated_quaternion[0]}, qx={interpolated_quaternion[1]}, qy={interpolated_quaternion[2]}, qz={interpolated_quaternion[3]}")
