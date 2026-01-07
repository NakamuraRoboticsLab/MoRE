from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .base import BaseRealObsActionInterface, ObsPacket


@dataclass
class DummyConfig:
    num_actions: int
    num_gaits: int
    cmd_init: np.ndarray
    gait_cmd: np.ndarray


class DummyRealInterface(BaseRealObsActionInterface):
    """A tiny interface for smoke-testing the deploy loop.

    - Returns zeros for state, fixed `cmd` and `gait_cmd`.
    - Depth image is a constant plane.
    """

    def __init__(self, cfg: DummyConfig):
        self._cfg = cfg
        self._t0 = time.monotonic()
        self._last_target = np.zeros(cfg.num_actions, dtype=np.float32)

    def recv_obs(self) -> ObsPacket:
        t = time.monotonic() - self._t0
        qj = (0.05 * np.sin(2 * np.pi * 0.5 * t) * np.ones(self._cfg.num_actions)).astype(np.float32)
        dqj = np.zeros(self._cfg.num_actions, dtype=np.float32)
        quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        omega_xyz = np.zeros(3, dtype=np.float32)
        cmd = self._cfg.cmd_init.astype(np.float32)
        gait_cmd = self._cfg.gait_cmd.astype(np.float32)
        depth = (1.0 * np.ones((64, 64), dtype=np.float32))
        return ObsPacket(
            qj=qj,
            dqj=dqj,
            quat_wxyz=quat_wxyz,
            omega_xyz=omega_xyz,
            cmd=cmd,
            gait_cmd=gait_cmd,
            depth_image=depth,
            timestamp_s=time.time(),
        )

    def send_target_joint_pos(self, target_q: np.ndarray, kp: np.ndarray | None = None, kd: np.ndarray | None = None) -> None:
        self._last_target = np.asarray(target_q, dtype=np.float32)

    def close(self) -> None:
        return
