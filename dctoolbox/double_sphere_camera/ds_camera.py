import json
from typing import Dict, Tuple

import cv2
import numpy as np


class DSCamera:
    """DSCamera class.
    V. Usenko, N. Demmel, and D. Cremers, "The Double Sphere Camera Model",
    Proc. of the Int. Conference on 3D Vision (3DV), 2018.
    """

    def __init__(
        self,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        xi: float,
        alpha: float,
        height: int,
        width: int,
    ):

        # Fisheye camera parameters
        self.h, self.w = height, width
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.xi = xi
        self.alpha = alpha

        self.fov = 180
        fov_rad = self.fov / 180 * np.pi
        self.fov_cos = np.cos(fov_rad / 2)
        self.intrinsic_keys = ["fx", "fy", "cx", "cy", "xi", "alpha"]

        # Valid mask for fisheye image
        self._valid_mask = None

    @property
    def img_size(self) -> Tuple[int, int]:
        return self.h, self.w

    @img_size.setter
    def img_size(self, img_size: Tuple[int, int]):
        self.h, self.w = map(int, img_size)

    @property
    def intrinsic(self) -> Dict[str, float]:
        intrinsic = {key: self.__dict__[key] for key in self.intrinsic_keys}
        return intrinsic

    @intrinsic.setter
    def intrinsic(self, intrinsic: Dict[str, float]):
        for key in self.intrinsic_keys:
            self.__dict__[key] = intrinsic[key]

    @property
    def valid_mask(self):
        if self._valid_mask is None:
            # Calculate and cache valid mask
            x = np.arange(self.w)
            y = np.arange(self.h)
            x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
            _, valid_mask = self.cam2world([x_grid, y_grid])
            self._valid_mask = valid_mask

        return self._valid_mask

    def __repr__(self):
        return (
            f"[{self.__class__.__name__}]\n img_size:{self.img_size},fov:{self.fov},\n"
            f" intrinsic:{json.dumps(self.intrinsic, indent=2)}"
        )

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def cam2world(self, point2D):
        """cam2world(point2D) projects a 2D point onto the unit sphere.
        point3D coord: x:right direction, y:down direction, z:front direction
        point2D coord: x:row direction, y:col direction (OpenCV image coordinate)
        Parameters
        ----------
        point2D : numpy array or list([u,v])
            array of point in image
        Returns
        -------
        unproj_pts : numpy array
            array of point on unit sphere
        valid_mask : numpy array
            array of valid mask
        """
        # Case: point2D = list([u, v]) or np.array()
        if isinstance(point2D, (list, np.ndarray)):
            u, v = point2D
        # Case: point2D = list([Scalar, Scalar])
        if not hasattr(u, "__len__"):
            u, v = np.array([u]), np.array([v])

        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        r2 = mx * mx + my * my

        # Check valid area
        s = 1 - (2 * self.alpha - 1) * r2
        valid_mask = s >= 0
        s[~valid_mask] = 0.0
        mz = (1 - self.alpha * self.alpha * r2) / (
            self.alpha * np.sqrt(s) + 1 - self.alpha
        )

        mz2 = mz * mz
        k1 = mz * self.xi + np.sqrt(mz2 + (1 - self.xi * self.xi) * r2)
        k2 = mz2 + r2
        k = k1 / k2

        # Unprojected unit vectors
        unproj_pts = k[..., np.newaxis] * np.stack([mx, my, mz], axis=-1)

        unproj_pts[..., 2] -= self.xi

        # Calculate fov
        unprojected_fov_cos = unproj_pts[..., 2]  # unproj_pts @ z_axis
        fov_mask = unprojected_fov_cos >= self.fov_cos
        valid_mask *= fov_mask
        return unproj_pts, valid_mask

    def world2cam(self, point3D):
        """world2cam(point3D) projects a 3D point on to the image.
        point3D coord: x:right direction, y:down direction, z:front direction
        point2D coord: x:row direction, y:col direction (OpenCV image coordinate).
        Parameters
        ----------
        point3D : numpy array or list([x, y, z])
            array of points in camera coordinate
        Returns
        -------
        proj_pts : numpy array
            array of points in image
        valid_mask : numpy array
            array of valid mask
        """
        x, y, z = point3D[..., 0], point3D[..., 1], point3D[..., 2]
        # Decide numpy or torch
        xp = np

        # Calculate fov
        point3D_fov_cos = point3D[..., 2]  # point3D @ z_axis
        fov_mask = point3D_fov_cos >= self.fov_cos

        # Calculate projection

        def square_root(*x):
            return xp.sqrt(sum([i * i for i in x]))

        d1 = square_root(x, y, z)
        zxi = self.xi * d1 + z
        d2 = square_root(x, y, zxi)

        div = self.alpha * d2 + (1 - self.alpha) * zxi
        u = self.fx * x / div + self.cx
        v = self.fy * y / div + self.cy

        # Projected points on image plane
        proj_pts = np.stack([u, v], axis=-1)

        # Check valid area
        if self.alpha <= 0.5:
            w1 = self.alpha / (1 - self.alpha)
        else:
            w1 = (1 - self.alpha) / self.alpha
        w2 = w1 + self.xi / xp.sqrt(2 * w1 * self.xi + self.xi * self.xi + 1)
        valid_mask = z > -w2 * d1
        valid_mask *= fov_mask

        return proj_pts, valid_mask

    def _warp_img(self, img, img_pts, valid_mask):
        # Remap
        img_pts = img_pts.astype(np.float32)
        out = cv2.remap(img, img_pts[..., 0], img_pts[..., 1], cv2.INTER_LINEAR)
        out[~valid_mask] = 0.0
        return out

    def to_perspective(self, img, zoom_factor=1.0):
        img_size = self.img_size
        # Generate 3D points
        h, w = img_size
        # z = f * min(img_size)
        # z = self.fx / 2
        z = 1
        # x = np.arange(w) - w / 2
        # y = np.arange(h) - h / 2
        new_intrinsic = np.array(
            [
                [self.fx / zoom_factor, 0, self.cx],
                [0.0, self.fy / zoom_factor, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        x = np.arange(w)
        y = np.arange(h)
        x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
        uvw = np.stack([x_grid, y_grid, np.full_like(x_grid, 1)], axis=-1)
        point3D = np.einsum("ji,hwi->hwj", np.linalg.inv(new_intrinsic), uvw)

        # Project on image plane
        img_pts, valid_mask = self.world2cam(point3D)
        out = self._warp_img(img, img_pts, valid_mask)
        return out
