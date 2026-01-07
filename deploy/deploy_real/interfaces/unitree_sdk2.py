from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .base import BaseRealObsActionInterface, ObsPacket
from .teleimager_depth import TeleImagerDepthClient, TeleImagerDepthConfig


@dataclass(frozen=True)
class UnitreeSdk2Config:
    # DDS
    net_interface: str
    msg_type: str  # "hg" or "go"
    imu_type: str  # "pelvis" or "torso"
    lowcmd_topic: str
    lowstate_topic: str

    # Mapping
    joint2motor_idx: list[int]  # length == num_actions

    # Optional: for imu_type == "torso" (transform IMU to pelvis frame)
    waist_yaw_motor_idx: Optional[int] = None

    # Startup / safety
    control_dt: float = 0.02
    move_to_default_s: float = 2.0
    require_remote: bool = True

    # Remote cmd mapping
    use_remote_cmd: bool = True
    remote_max_cmd: np.ndarray = np.array([0.8, 0.5, 1.57], dtype=np.float32)

    # Optional additional motors to hold at fixed targets (e.g. arms/waist)
    fixed_motors: Optional[list[dict[str, Any]]] = None


class StopDeploy(Exception):
    """Raised to stop the deploy loop (e.g. remote 'select')."""


class UnitreeSdk2Interface(BaseRealObsActionInterface):
    """Unitree SDK2 (DDS) interface based on unitree_ref.

    It mirrors the reference startup flow:
    - zero torque until remote START
    - move to default pose
    - hold default until remote A
    - run policy until remote SELECT

    Note: this file intentionally lazy-imports `unitree_sdk2py` so that the repo
    can still be used without the SDK installed.
    """

    def __init__(self, deploy_cfg: Any):
        # `deploy_cfg` is DeployConfig from deploy_real_with_resi.py.
        raw = getattr(deploy_cfg, "extra", None)
        if not isinstance(raw, dict):
            raise ValueError("DeployConfig.extra missing; please reload config via deploy_real_with_resi.py")

        self._cfg = UnitreeSdk2Config(
            net_interface=str(raw["net_interface"]),
            msg_type=str(raw["msg_type"]),
            imu_type=str(raw.get("imu_type", "pelvis")),
            lowcmd_topic=str(raw.get("lowcmd_topic", "rt/lowcmd")),
            lowstate_topic=str(raw.get("lowstate_topic", "rt/lowstate")),
            joint2motor_idx=[int(x) for x in raw["joint2motor_idx"]],
            waist_yaw_motor_idx=int(raw["waist_yaw_motor_idx"]) if "waist_yaw_motor_idx" in raw else None,
            control_dt=float(raw.get("control_dt", 0.02)),
            move_to_default_s=float(raw.get("move_to_default_s", 2.0)),
            require_remote=bool(raw.get("require_remote", True)),
            use_remote_cmd=bool(raw.get("use_remote_cmd", True)),
            remote_max_cmd=np.array(raw.get("remote_max_cmd", [0.8, 0.5, 1.57]), dtype=np.float32),
            fixed_motors=raw.get("fixed_motors"),
        )

        self._num_actions = int(getattr(deploy_cfg, "num_actions"))
        if len(self._cfg.joint2motor_idx) != self._num_actions:
            raise ValueError(f"joint2motor_idx length must be {self._num_actions}")

        self._default_angles = np.asarray(getattr(deploy_cfg, "default_angles"), dtype=np.float32)
        self._gait_cmd = np.asarray(getattr(deploy_cfg, "gait_cmd"), dtype=np.float32)
        self._kps = getattr(deploy_cfg, "kps", None)
        self._kds = getattr(deploy_cfg, "kds", None)

        if self._kps is not None:
            self._kps = np.asarray(self._kps, dtype=np.float32)
        if self._kds is not None:
            self._kds = np.asarray(self._kds, dtype=np.float32)

        if self._kps is None or self._kds is None:
            raise ValueError("For unitree_sdk2 interface, please provide kps and kds in YAML (length == num_actions).")

        # Lazy imports
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
            from unitree_sdk2py.utils.crc import CRC

            self._ChannelFactoryInitialize = ChannelFactoryInitialize
            self._ChannelPublisher = ChannelPublisher
            self._ChannelSubscriber = ChannelSubscriber
            self._LowCmdGo = LowCmdGo
            self._LowCmdHG = LowCmdHG
            self._LowStateGo = LowStateGo
            self._LowStateHG = LowStateHG
            self._lowcmd_go_ctor = unitree_go_msg_dds__LowCmd_
            self._lowcmd_hg_ctor = unitree_hg_msg_dds__LowCmd_
            self._lowstate_go_ctor = unitree_go_msg_dds__LowState_
            self._lowstate_hg_ctor = unitree_hg_msg_dds__LowState_
            self._CRC = CRC
        except Exception as e:
            raise ImportError(
                "unitree_sdk2py is required for UnitreeSdk2Interface. "
                "Please install Unitree SDK2 python package and its dependencies."
            ) from e

        # Helpers from unitree_ref
        from deploy.deploy_real.unitree_ref.common.command_helper import (
            create_damping_cmd,
            create_zero_cmd,
            init_cmd_go,
            init_cmd_hg,
            MotorMode,
        )
        from deploy.deploy_real.unitree_ref.common.remote_controller import RemoteController, KeyMap
        from deploy.deploy_real.unitree_ref.common.rotation_helper import transform_imu_data

        self._create_zero_cmd = create_zero_cmd
        self._create_damping_cmd = create_damping_cmd
        self._init_cmd_go = init_cmd_go
        self._init_cmd_hg = init_cmd_hg
        self._MotorMode = MotorMode
        self._RemoteController = RemoteController
        self._KeyMap = KeyMap
        self._transform_imu_data = transform_imu_data

        # DDS init
        self._ChannelFactoryInitialize(0, self._cfg.net_interface)

        self._remote = self._RemoteController()

        self._low_state = None
        self._low_cmd = None
        self._publisher = None

        self._mode_machine = 0
        self._mode_pr = self._MotorMode.PR

        self._init_dds()
        self._wait_for_low_state()
        self._init_low_cmd()

        self._state = "ZERO"  # ZERO -> MOVE -> HOLD -> RUN
        self._move_t0 = None
        self._move_init_pos = None
        self._last_cmd = np.zeros(3, dtype=np.float32)

        # Optional TeleImager depth client
        self._depth_client: TeleImagerDepthClient | None = None
        if "teleimager_host" in raw:
            down_hw = raw.get("teleimager_downsample_hw", None)
            down_hw_t = None
            if isinstance(down_hw, (list, tuple)) and len(down_hw) == 2:
                down_hw_t = (int(down_hw[0]), int(down_hw[1]))
            self._depth_client = TeleImagerDepthClient(
                TeleImagerDepthConfig(
                    host=str(raw["teleimager_host"]),
                    request_port=int(raw.get("teleimager_request_port", 60000)),
                    cam_topic=str(raw.get("teleimager_cam_topic", "head_camera")),
                    depth_unit_scale_m=float(raw.get("teleimager_depth_unit_scale_m", 0.001)),
                    downsample_hw=down_hw_t,
                )
            )

    def _init_dds(self) -> None:
        if self._cfg.msg_type == "hg":
            self._low_cmd = self._lowcmd_hg_ctor()
            self._low_state = self._lowstate_hg_ctor()

            self._publisher = self._ChannelPublisher(self._cfg.lowcmd_topic, self._LowCmdHG)
            self._publisher.Init()

            sub = self._ChannelSubscriber(self._cfg.lowstate_topic, self._LowStateHG)
            sub.Init(self._lowstate_hg_handler, 10)
            self._subscriber = sub

        elif self._cfg.msg_type == "go":
            self._low_cmd = self._lowcmd_go_ctor()
            self._low_state = self._lowstate_go_ctor()

            self._publisher = self._ChannelPublisher(self._cfg.lowcmd_topic, self._LowCmdGo)
            self._publisher.Init()

            sub = self._ChannelSubscriber(self._cfg.lowstate_topic, self._LowStateGo)
            sub.Init(self._lowstate_go_handler, 10)
            self._subscriber = sub

        else:
            raise ValueError("msg_type must be 'hg' or 'go'")

    def _lowstate_hg_handler(self, msg: Any) -> None:
        self._low_state = msg
        self._mode_machine = int(getattr(msg, "mode_machine", 0))
        self._remote.set(msg.wireless_remote)

    def _lowstate_go_handler(self, msg: Any) -> None:
        self._low_state = msg
        self._remote.set(msg.wireless_remote)

    def _send_cmd(self) -> None:
        self._low_cmd.crc = self._CRC().Crc(self._low_cmd)
        self._publisher.Write(self._low_cmd)

    def _wait_for_low_state(self) -> None:
        # wait until subscriber receives data
        while getattr(self._low_state, "tick", 0) == 0:
            time.sleep(self._cfg.control_dt)

    def _init_low_cmd(self) -> None:
        if self._cfg.msg_type == "hg":
            self._init_cmd_hg(self._low_cmd, self._mode_machine, self._mode_pr)
        else:
            weak = []
            self._init_cmd_go(self._low_cmd, weak_motor=weak)

    def _read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        qj = np.zeros(self._num_actions, dtype=np.float32)
        dqj = np.zeros(self._num_actions, dtype=np.float32)
        for i, motor_idx in enumerate(self._cfg.joint2motor_idx):
            st = self._low_state.motor_state[motor_idx]
            qj[i] = float(st.q)
            dqj[i] = float(st.dq)
        return qj, dqj

    def _read_imu(self) -> tuple[np.ndarray, np.ndarray]:
        quat = np.array(self._low_state.imu_state.quaternion, dtype=np.float32)
        omega = np.array(self._low_state.imu_state.gyroscope, dtype=np.float32)

        if self._cfg.imu_type == "torso":
            if self._cfg.waist_yaw_motor_idx is None:
                raise ValueError("imu_type='torso' requires waist_yaw_motor_idx in YAML")
            waist_yaw = float(self._low_state.motor_state[self._cfg.waist_yaw_motor_idx].q)
            waist_yaw_omega = float(self._low_state.motor_state[self._cfg.waist_yaw_motor_idx].dq)
            quat, omega = self._transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=quat,
                imu_omega=np.array([omega], dtype=np.float32),
            )
            omega = np.array(omega, dtype=np.float32).reshape(3)

        return quat, omega

    def _fill_fixed_motors(self) -> None:
        if not self._cfg.fixed_motors:
            return
        for ent in self._cfg.fixed_motors:
            motor_idx = int(ent["motor_idx"])
            q = float(ent.get("q", 0.0))
            qd = float(ent.get("qd", 0.0))
            kp = float(ent.get("kp", 0.0))
            kd = float(ent.get("kd", 0.0))
            tau = float(ent.get("tau", 0.0))
            self._low_cmd.motor_cmd[motor_idx].q = q
            self._low_cmd.motor_cmd[motor_idx].qd = qd
            self._low_cmd.motor_cmd[motor_idx].kp = kp
            self._low_cmd.motor_cmd[motor_idx].kd = kd
            self._low_cmd.motor_cmd[motor_idx].tau = tau

    def recv_obs(self) -> ObsPacket:
        if self._cfg.require_remote and self._state == "RUN":
            if self._remote.button[self._KeyMap.select] == 1:
                self._create_damping_cmd(self._low_cmd)
                self._send_cmd()
                raise StopDeploy("Remote select pressed")

        # Safety state machine: send appropriate lowcmd even before RUN.
        if self._cfg.require_remote:
            if self._state == "ZERO":
                if self._remote.button[self._KeyMap.start] != 1:
                    self._create_zero_cmd(self._low_cmd)
                    self._send_cmd()
                else:
                    self._state = "MOVE"
                    self._move_t0 = time.monotonic()
                    qj, _ = self._read_joint_state()
                    self._move_init_pos = qj

            if self._state == "MOVE":
                assert self._move_t0 is not None
                assert self._move_init_pos is not None
                alpha = (time.monotonic() - self._move_t0) / max(self._cfg.move_to_default_s, 1e-6)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                target = (1 - alpha) * self._move_init_pos + alpha * self._default_angles
                # write default target to controlled motors
                for i, motor_idx in enumerate(self._cfg.joint2motor_idx):
                    self._low_cmd.motor_cmd[motor_idx].q = float(target[i])
                    self._low_cmd.motor_cmd[motor_idx].qd = 0.0
                    self._low_cmd.motor_cmd[motor_idx].kp = float(self._kps[i])
                    self._low_cmd.motor_cmd[motor_idx].kd = float(self._kds[i])
                    self._low_cmd.motor_cmd[motor_idx].tau = 0.0
                self._fill_fixed_motors()
                self._send_cmd()
                if alpha >= 1.0:
                    self._state = "HOLD"

            if self._state == "HOLD":
                if self._remote.button[self._KeyMap.A] != 1:
                    for i, motor_idx in enumerate(self._cfg.joint2motor_idx):
                        self._low_cmd.motor_cmd[motor_idx].q = float(self._default_angles[i])
                        self._low_cmd.motor_cmd[motor_idx].qd = 0.0
                        self._low_cmd.motor_cmd[motor_idx].kp = float(self._kps[i])
                        self._low_cmd.motor_cmd[motor_idx].kd = float(self._kds[i])
                        self._low_cmd.motor_cmd[motor_idx].tau = 0.0
                    self._fill_fixed_motors()
                    self._send_cmd()
                else:
                    self._state = "RUN"

        qj, dqj = self._read_joint_state()
        quat, omega = self._read_imu()

        depth_image = None
        if self._depth_client is not None:
            try:
                depth_image = self._depth_client.get_depth_meters()
            except Exception:
                depth_image = None

        if self._cfg.use_remote_cmd:
            cmd = np.array(
                [
                    float(self._remote.ly),
                    float(self._remote.lx) * -1.0,
                    float(self._remote.rx) * -1.0,
                ],
                dtype=np.float32,
            )
            cmd = cmd * self._cfg.remote_max_cmd
            self._last_cmd = cmd
        else:
            cmd = self._last_cmd

        gait_cmd = self._gait_cmd

        return ObsPacket(
            qj=qj,
            dqj=dqj,
            quat_wxyz=quat,
            omega_xyz=omega,
            cmd=cmd,
            gait_cmd=gait_cmd,
            depth_image=depth_image,
            timestamp_s=time.time(),
        )

    def send_target_joint_pos(self, target_q: np.ndarray, kp: np.ndarray | None = None, kd: np.ndarray | None = None) -> None:
        # Only send policy command in RUN when remote gating is enabled.
        if self._cfg.require_remote and self._state != "RUN":
            return

        kp_use = np.asarray(kp if kp is not None else self._kps, dtype=np.float32)
        kd_use = np.asarray(kd if kd is not None else self._kds, dtype=np.float32)

        for i, motor_idx in enumerate(self._cfg.joint2motor_idx):
            self._low_cmd.motor_cmd[motor_idx].q = float(target_q[i])
            self._low_cmd.motor_cmd[motor_idx].qd = 0.0
            self._low_cmd.motor_cmd[motor_idx].kp = float(kp_use[i])
            self._low_cmd.motor_cmd[motor_idx].kd = float(kd_use[i])
            self._low_cmd.motor_cmd[motor_idx].tau = 0.0

        self._fill_fixed_motors()
        self._send_cmd()

    def close(self) -> None:
        try:
            self._create_damping_cmd(self._low_cmd)
            self._send_cmd()
        except Exception:
            pass

        if self._depth_client is not None:
            try:
                self._depth_client.close()
            except Exception:
                pass
