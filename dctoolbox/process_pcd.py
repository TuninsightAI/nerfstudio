import numpy as np
import open3d as o3d
import shutil
from dataclasses import dataclass
from pathlib import Path

S = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)


def remove_outlier(pcd: o3d.geometry.PointCloud, nb_points: int, std_ratio: float) -> o3d.geometry.PointCloud:
    """
    Remove outliers from a point cloud using RANSAC plane detection
    :param pcd: point cloud
    :param nb_points: number of points to consider for the plane detection
    :param std_ratio: standard deviation ratio for the plane detection
    :return: point cloud without outliers
    """
    pcd, inliers = pcd.remove_statistical_outlier(nb_points, std_ratio)
    return pcd


def voxel_down_sample(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    """
    Downsample a point cloud using voxel grid
    :param pcd: point cloud
    :param voxel_size: voxel size
    :return: downsampled point cloud
    """
    pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


def convert_slam_pcd_to_opencv(point_cloud: o3d.geometry.PointCloud):
    xyz = np.array(point_cloud.points)
    xyz = np.hstack([xyz, np.ones((xyz.shape[0], 1))])

    opencv_xyz = np.dot(S.T, xyz.T).T[:, :3]
    opencv_pcd = o3d.geometry.PointCloud()
    opencv_pcd.colors = point_cloud.colors
    opencv_pcd.points = o3d.utility.Vector3dVector(opencv_xyz)

    return opencv_pcd


@dataclass
class ProcessPCDConfig:
    input_path: Path
    output_path: Path
    voxel_size: float
    convert_to_opencv: bool = True

    def __post_init__(self):
        assert self.input_path.exists(), f"Input path {self.input_path} does not exist"

    def main(self):
        copied_pcd = False
        if self.input_path.suffix == ".dcloud":
            new_path = self.input_path.with_suffix(".pcd")
            shutil.copy(self.input_path, new_path)
            self.input_path = new_path
            copied_pcd = True

        pcd = o3d.io.read_point_cloud(self.input_path.as_posix())
        if self.convert_to_opencv:
            pcd = convert_slam_pcd_to_opencv(pcd)

        pcd = voxel_down_sample(pcd, self.voxel_size)
        pcd = remove_outlier(pcd, 20, 2.0)
        o3d.io.write_point_cloud(self.output_path.as_posix(), pcd)

        if copied_pcd:
            self.input_path.unlink()


if __name__ == "__main__":
    import tyro

    tyro.cli(tyro.conf.FlagConversionOff[ProcessPCDConfig]).main()
