from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class ObsPacket:
    """Raw inputs needed to build the MoRE observation.

    Conventions follow `deploy_mujoco/deploy_mujoco_with_resi.py`:
    - `quat_wxyz` is base orientation quaternion in (w, x, y, z).
    - `omega_xyz` is base angular velocity (x, y, z).
    - `qj`/`dqj` are joint pos/vel with length `num_actions`.
    - `cmd` is (vx, vy, wz).
    - `gait_cmd` is the gait command vector (e.g. one-hot or logits).

    `depth_image`:
    - If provided, should be a float32 array (H, W).
    - By default we treat it as raw metric depth (meters) and will apply
      the same clipping/normalization pipeline as MuJoCo deploy.
    """

    qj: np.ndarray
    dqj: np.ndarray
    quat_wxyz: np.ndarray
    omega_xyz: np.ndarray
    cmd: np.ndarray
    gait_cmd: np.ndarray
    depth_image: Optional[np.ndarray] = None
    timestamp_s: Optional[float] = None


class BaseRealObsActionInterface(Protocol):
    """Interface between the deploy loop and the external world.

    You should implement this with your robot SDK / middleware.

    Minimal contract:
    - `recv_obs()` returns the latest `ObsPacket`.
    - `send_target_joint_pos()` sends desired joint positions.

    Optional:
    - `close()` to release resources.
    """

    def recv_obs(self) -> ObsPacket:
        raise NotImplementedError

    def send_target_joint_pos(
        self,
        target_q: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return
