# this file encodes helper function for epipolar geometry.

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

_EPS = np.finfo(float).eps * 4.0


def skew_symmetric(v: Tensor) -> Tensor:
    """Computes the skew symmetric matrix of a vector.

    Args:
        v: The vector.

    Returns:
        The skew symmetric matrix.
    """
    return torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def essential_matrix_from_fundamental_matrix(
    fundamental_matrix: Tensor, intrinsics1: Tensor, intrinsics2: Tensor
) -> Tensor:
    """Computes the essential matrix from the fundamental matrix.

    Args:
        fundamental_matrix: The fundamental matrix.
        intrinsics1: The intrinsics of the first camera.
        intrinsics2: The intrinsics of the second camera.

    Returns:
        The essential matrix.
    """
    return intrinsics2.T @ fundamental_matrix @ intrinsics1


def fundamental_matrix_from_essential_matrix(
    essential_matrix: Tensor, intrinsics1: Tensor, intrinsics2: Tensor
) -> Tensor:
    """Computes the fundamental matrix from the essential matrix.

    Args:
        essential_matrix: The essential matrix.
        intrinsics1: The intrinsics of the first camera.
        intrinsics2: The intrinsics of the second camera.

    Returns:
        The fundamental matrix.
    """
    return intrinsics2.inverse().T @ essential_matrix @ intrinsics1.inverse()


def compute_correspondence_epipolar_error(
    points1: Float[Tensor, "*batch 3"],
    points2: Float[Tensor, "*batch 3"],
    fundamental_matrix: Float[Tensor, "*batch 3 3"],
) -> Float[Tensor, "*batch"]:
    """Computes the epipolar error for correspondences.

    Args:
        points1: The points in the first image.
        points2: The points in the second image.
        fundamental_matrix: The fundamental matrix.

    Returns:
        The epipolar error for each correspondence.
    """
    points1_h = torch.cat([points1, torch.ones_like(points1[..., :1])], dim=-1)
    points2_h = torch.cat([points2, torch.ones_like(points2[..., :1])], dim=-1)

    # Compute the epipolar lines in the second image.
    epipolar_lines = fundamental_matrix @ points1_h.transpose(-1, -2)
    epipolar_lines /= torch.norm(epipolar_lines[..., :2], dim=-1, keepdim=True)

    # Compute the epipolar error.
    return torch.cross(points2_h, epipolar_lines, dim=-1).norm(dim=-1)


def estimate_fundamental_matrix(
    points1: Float[Tensor, "*batch 3"],
    points2: Float[Tensor, "*batch 3"],
    intrinsics1: Float[Tensor, "*batch 3 3"],
    intrinsics2: Float[Tensor, "*batch 3 3"],
) -> Float[Tensor, "*batch 3 3"]:
    """Estimate the fundamental matrix using the 8-point algorithm.

    Args:
        points1: The points in the first image.
        points2: The points in the second image.
        intrinsics1: The intrinsics of the first camera.
        intrinsics2: The intrinsics of the second camera.

    Returns:
        The fundamental matrix.
    """
    # Normalize the points.
    points1_normalized = intrinsics1.inverse() @ points1
    points2_normalized = intrinsics2.inverse() @ points2

    # Compute the fundamental matrix.
    fundamental_matrix = torch.linalg.lstsq(
        points1_normalized.transpose(-1, -2), points2_normalized.transpose(-1, -2)
    )

    return fundamental_matrix[0].transpose(-1, -2)


# what else?
# 1. compute_fundamental_matrix
# 2. compute_essential_matrix
# 3. compute_correspondence_epipolar_error
# 4. estimate_fundamental_matrix
# 5. estimate_essential_matrix
# 6. estimate_fundamental_matrix_from_essential_matrix
# 7. estimate_essential_matrix_from_fundamental_matrix
# 8. estimate_pose_from_essential_matrix
# 9. estimate_pose_from_fundamental_matrix


def estimate_pose_from_essential_matrix(
    essential_matrix: Float[Tensor, "*batch 3 3"],
    intrinsics1: Float[Tensor, "*batch 3 3"],
    intrinsics2: Float[Tensor, "*batch 3 3"],
) -> Float[Tensor, "*batch 3 4"]:
    """Estimate the pose from the essential matrix.

    Args:
        essential_matrix: The essential matrix.
        intrinsics1: The intrinsics of the first camera.
        intrinsics2: The intrinsics of the second camera.

    Returns:
        The pose.
    """
    pose = torch.linalg.solve(essential_matrix, intrinsics1.T)
    return pose


def estimate_pose_from_fundamental_matrix(
    fundamental_matrix: Float[Tensor, "*batch 3 3"],
    intrinsics1: Float[Tensor, "*batch 3 3"],
    intrinsics2: Float[Tensor, "*batch 3 3"],
) -> Float[Tensor, "*batch 3 4"]:
    """Estimate the pose from the fundamental matrix.

    Args:
        fundamental_matrix: The fundamental matrix.
        intrinsics1: The intrinsics of the first camera.
        intrinsics2: The intrinsics of the second camera.

    Returns:
        The pose.
    """
    pose = torch.linalg.solve(fundamental_matrix, intrinsics1.T)
    return pose


def eight_point_essential_matrix(
    img1_points, img2_points, camera_1_matrix, camera_2_matrix, device="cpu"
):
    """
    The eight-point algorithm is used to compute the essential matrix from
    corresponding points in two images. N should be greater equal 8.

    img1_points: points on image 1, in shape of N * 2
    img2_points: points on image 2, in shape of N * 2
    camera_1_matrix, camera_2_matrix: camera matrix in form 3 * 3
    """
    if img1_points.shape[1] != 2 or img2_points.shape[1] != 2:
        raise ValueError("Dimention of each point in the image should be 2.")
    if img1_points.shape[0] != img2_points.shape[0]:
        raise ValueError("Number of points in image one and two should be equal.")
    if len(img1_points) < 5:
        raise ValueError(
            "Number of corresponding points should be greater or equal to 8."
        )
    if camera_1_matrix.shape[0] != 3 and camera_1_matrix.shape[1] != 3:
        raise ValueError("Inputed camera matrix is not correct.")

    with torch.no_grad():

        num_corresponding = img1_points.shape[0]

        # convert to tensor
        img1_points = torch.tensor(
            data=img1_points, dtype=torch.float32, device=device
        )  # N * 2
        img2_points = torch.tensor(
            data=img2_points, dtype=torch.float32, device=device
        )  # N * 2
        camera_1_matrix_tensor = torch.tensor(
            data=camera_1_matrix, dtype=torch.float32, device=device
        )  # 3 * 3
        camera_2_matrix_tensor = torch.tensor(
            data=camera_2_matrix, dtype=torch.float32, device=device
        )  # 3 * 3

        # convert points to homogeneous
        img1_points_hmg = torch.cat(
            (img1_points, torch.ones((num_corresponding, 1), device=device)), dim=1
        )  # N * 3
        img2_points_hmg = torch.cat(
            (img2_points, torch.ones((num_corresponding, 1), device=device)), dim=1
        )  # N * 3

        # find local ray direction that passes points in image 1
        # local ray direction = inverse of camera matrix * point on image in homogeneous
        img1_lrd = torch.matmul(
            camera_1_matrix_tensor.inverse(), img1_points_hmg.t()
        ).t()  # N * 3
        # Calculate the norms of each row
        row_norms = torch.norm(img1_lrd, dim=1, keepdim=True)
        # Normalize each row by dividing by its norm
        img1_lrd = img1_lrd / row_norms

        # find local ray direction that passes points in image 2
        img2_lrd = torch.matmul(
            camera_2_matrix_tensor.inverse(), img2_points_hmg.t()
        ).t()  # N * 3
        # Calculate the norms of each row
        row_norms = torch.norm(img2_lrd, dim=1, keepdim=True)
        # Normalize each row by dividing by its norm
        img2_lrd = img2_lrd / row_norms

        # convert each correspoding local ray direction pair from
        # [x1, y1, 1] and [x2, y2, 1] to
        # [x1x2, y1x2, x2, x1y2, y1y2, y2, x1, y1, 1] in an efficient way by
        # calculating the Kronecker product for the batch
        kron_product = torch.bmm(
            img2_lrd.view(num_corresponding, 3, 1),
            img1_lrd.view(num_corresponding, 1, 3),
        ).view(
            num_corresponding, -1
        )  # N * 9
        Y = kron_product.t()  # 9 * N

        # flatten essential matirx(e) can be obtained by finding left singular vector
        # corresponding to lowest singular value of SVD decomposition of Y
        U, S, Vh = torch.linalg.svd(Y, full_matrices=True)

        # essential matrix
        e = U[:, -1]
        E = torch.reshape(e, shape=(3, 3))

        # approximate E by a rank 2 matrix
        U, S, Vh = torch.linalg.svd(E, full_matrices=True)
        S[2] = 0.0
        E_rank2 = U @ torch.diag(S) @ Vh

        # epipole in image 1
        ep_1 = Vh[-1, :]
        if ep_1[2] != 0:
            ep_1_normalized = ep_1 / ep_1[2]
        # epipole in image 2
        ep_2 = U[:, -1]
        if ep_2[2] != 0:
            ep_2_normalized = ep_2 / ep_2[2]

        return {
            "essential_matrix": E_rank2.cpu().numpy(),
            "epipole_img_1": ep_1_normalized.cpu().numpy(),
            "epipole_img_2": ep_2_normalized.cpu().numpy(),
        }
