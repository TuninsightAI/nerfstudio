import torch
from functools import wraps
from jaxtyping import Float
from threading import Thread
from torch import Tensor
from torch.nn import functional as F

__all__ = ["quat2rotation", "rotation2quat", "run_in_thread"]


def run_in_thread(_func=None, *, daemon: bool = False, name: str | None = None):
    def decorator_run(func):
        @wraps(func)
        def run(*args, **kwargs):
            proc = Thread(
                target=func, args=args, kwargs=kwargs, daemon=daemon, name=name
            )
            proc.start()

            return proc

        return run

    if _func is None:
        return decorator_run
    else:
        return decorator_run(_func)


def quat2rotation(r: Float[Tensor, "batch 4"]) -> Float[Tensor, "batch 3 3"]:
    """
    Convert quaternion to rotation matrix, w, x, y, z
    :param r:
    :return:
    """
    assert r.ndim == 2, r.shape
    q = F.normalize(r, p=2, dim=1)
    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotation2quat(R: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 4"]:
    q = torch.zeros(
        R.shape[0], 4, device=R.device, dtype=R.dtype
    )  # (batch_size, 4) x, y, z, w
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q0_mask = trace > 0
    q1_mask = (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2]) & ~q0_mask
    q2_mask = (R[..., 1, 1] > R[..., 2, 2]) & ~q0_mask & ~q1_mask
    q3_mask = ~q0_mask & ~q1_mask & ~q2_mask
    if q0_mask.any():
        R_for_q0 = R[q0_mask]
        S_for_q0 = 0.5 / torch.sqrt(1 + trace[q0_mask])
        q[q0_mask, 3] = 0.25 / S_for_q0
        q[q0_mask, 0] = (R_for_q0[..., 2, 1] - R_for_q0[..., 1, 2]) * S_for_q0
        q[q0_mask, 1] = (R_for_q0[..., 0, 2] - R_for_q0[..., 2, 0]) * S_for_q0
        q[q0_mask, 2] = (R_for_q0[..., 1, 0] - R_for_q0[..., 0, 1]) * S_for_q0

    if q1_mask.any():
        R_for_q1 = R[q1_mask]
        S_for_q1 = 2.0 * torch.sqrt(
            1 + R_for_q1[..., 0, 0] - R_for_q1[..., 1, 1] - R_for_q1[..., 2, 2]
        )
        q[q1_mask, 0] = 0.25 * S_for_q1
        q[q1_mask, 1] = (R_for_q1[..., 0, 1] + R_for_q1[..., 1, 0]) / S_for_q1
        q[q1_mask, 2] = (R_for_q1[..., 0, 2] + R_for_q1[..., 2, 0]) / S_for_q1
        q[q1_mask, 3] = (R_for_q1[..., 2, 1] - R_for_q1[..., 1, 2]) / S_for_q1

    if q2_mask.any():
        R_for_q2 = R[q2_mask]
        S_for_q2 = 2.0 * torch.sqrt(
            1 + R_for_q2[..., 1, 1] - R_for_q2[..., 0, 0] - R_for_q2[..., 2, 2]
        )
        q[q2_mask, 0] = (R_for_q2[..., 0, 1] + R_for_q2[..., 1, 0]) / S_for_q2
        q[q2_mask, 1] = 0.25 * S_for_q2
        q[q2_mask, 2] = (R_for_q2[..., 1, 2] + R_for_q2[..., 2, 1]) / S_for_q2
        q[q2_mask, 3] = (R_for_q2[..., 0, 2] - R_for_q2[..., 2, 0]) / S_for_q2

    if q3_mask.any():
        R_for_q3 = R[q3_mask]
        S_for_q3 = 2.0 * torch.sqrt(
            1 + R_for_q3[..., 2, 2] - R_for_q3[..., 0, 0] - R_for_q3[..., 1, 1]
        )
        q[q3_mask, 0] = (R_for_q3[..., 0, 2] + R_for_q3[..., 2, 0]) / S_for_q3
        q[q3_mask, 1] = (R_for_q3[..., 1, 2] + R_for_q3[..., 2, 1]) / S_for_q3
        q[q3_mask, 2] = 0.25 * S_for_q3
        q[q3_mask, 3] = (R_for_q3[..., 1, 0] - R_for_q3[..., 0, 1]) / S_for_q3

    q = q[:, [3, 0, 1, 2]]
    q = F.normalize(q, p=2, dim=1)

    return q
