from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .base import BaseRealObsActionInterface, ObsPacket
from .teleimager_depth import TeleImagerDepthClient, TeleImagerDepthConfig


def _list_available_interfaces() -> list[str]:
    """Best-effort list of local network interface names."""
    try:
        import socket

        return sorted({name for _, name in socket.if_nameindex()})
    except Exception:
        return []


def _ipv4_addrs_by_interface() -> dict[str, list[str]]:
    """Best-effort IPv4 address lookup using `ip` (Linux).

    Returns a dict of interface -> list of IPv4 addresses (as strings).
    """
    try:
        out = subprocess.check_output(["ip", "-o", "-4", "addr", "show"], text=True)
    except Exception:
        return {}

    by_iface: dict[str, list[str]] = {}
    # Example line:
    # 2: enp3s0    inet 192.168.123.2/24 brd 192.168.123.255 scope global dynamic noprefixroute enp3s0\
    pattern = re.compile(r"^\d+:\s+(?P<iface>\S+)\s+inet\s+(?P<ip>\d+\.\d+\.\d+\.\d+)/")
    for line in out.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        iface = m.group("iface").split("@")[0]
        ip = m.group("ip")
        by_iface.setdefault(iface, []).append(ip)
    return by_iface


def _resolve_net_interface(requested: str) -> str:
    """Resolve a usable NIC name.

    - If requested is empty or 'auto': pick a reasonable default.
    - If requested doesn't exist: try to pick a unique reasonable default; otherwise error.
    """
    requested = (requested or "").strip()
    requested = requested.split("@")[0]  # tolerate names like eth0@if3

    available = _list_available_interfaces()
    if not available:
        # Can't validate; fall back to requested (or empty) and let SDK error.
        return requested

    if requested in ("", "auto"):
        return _pick_default_interface(available)

    if requested in available:
        return requested

    # Simple prefix/substring match (helps when user copies from `ip link` output)
    matches = [n for n in available if n == requested or n.startswith(requested) or requested in n]
    if len(matches) == 1:
        return matches[0]

    # If config is wrong, only auto-fallback when choice is unambiguous.
    try:
        auto = _pick_default_interface(available)
        print(
            f"[unitree_sdk2] Warning: net_interface {requested!r} not found; using {auto!r}. "
            "Set net_interface in YAML to silence this."
        )
        return auto
    except Exception:
        available_s = ", ".join(available)
        raise ValueError(
            f"net_interface {requested!r} not found. Available interfaces: {available_s}. "
            "Set net_interface to a valid NIC name (or 'auto')."
        )


def _pick_default_interface(available: list[str]) -> str:
    """Pick a best-effort default NIC.

    Preference order:
    1) An interface with IPv4 in 192.168.123.* (common Unitree robot subnet)
    2) The only non-loopback interface (if unique)
    3) Common wired names (eth0, en*, eno*, ens*)
    4) Otherwise error (ambiguous)
    """
    ipv4 = _ipv4_addrs_by_interface()
    for iface in available:
        for ip in ipv4.get(iface, []):
            if ip.startswith("192.168.123."):
                return iface

    non_loopback = [n for n in available if n != "lo"]
    if len(non_loopback) == 1:
        return non_loopback[0]

    common = [n for n in non_loopback if n == "eth0" or n.startswith(("en", "eno", "ens"))]
    if len(common) == 1:
        return common[0]

    raise RuntimeError(
        "Cannot auto-select a network interface (multiple candidates). "
        "Please set net_interface explicitly in YAML."
    )


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

    # Debug
    debug_remote: bool = False
    debug_remote_interval_s: float = 0.5

    # Optional additional motors to hold at fixed targets (e.g. arms/waist)
    fixed_motors: Optional[list[dict[str, Any]]] = None

    # If true, apply PD hold to all motors NOT in joint2motor_idx (and not in fixed_motors).
    # This is useful for robots with more DoFs (e.g. 29DoF G1 with waist+wrist joints)
    # when the policy only controls a subset.
    lock_unused_motors: bool = False
    lock_unused_kp: float = 50.0
    lock_unused_kd: float = 1.0
    lock_unused_q: float = 0.0
    lock_unused_exclude: Optional[list[int]] = None

    # Separate PD holds (useful when you want different stiffness per group)
    lock_waist_motors: bool = False
    waist_motor_indices: Optional[list[int]] = None
    waist_lock_kp: float = 80.0
    waist_lock_kd: float = 2.0
    waist_lock_q: float = 0.0

    lock_wrist_motors: bool = False
    wrist_motor_indices: Optional[list[int]] = None
    wrist_lock_kp: float = 20.0
    wrist_lock_kd: float = 0.5
    wrist_lock_q: float = 0.0


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

        requested_net_interface = str(raw.get("net_interface", "")).strip()
        resolved_net_interface = _resolve_net_interface(requested_net_interface)

        self._cfg = UnitreeSdk2Config(
            net_interface=resolved_net_interface,
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
            debug_remote=bool(raw.get("debug_remote", False)),
            debug_remote_interval_s=float(raw.get("debug_remote_interval_s", 0.5)),
            fixed_motors=raw.get("fixed_motors"),
            lock_unused_motors=bool(raw.get("lock_unused_motors", False)),
            lock_unused_kp=float(raw.get("lock_unused_kp", 50.0)),
            lock_unused_kd=float(raw.get("lock_unused_kd", 1.0)),
            lock_unused_q=float(raw.get("lock_unused_q", 0.0)),
            lock_unused_exclude=raw.get("lock_unused_exclude"),

            lock_waist_motors=bool(raw.get("lock_waist_motors", False)),
            waist_motor_indices=raw.get("waist_motor_indices"),
            waist_lock_kp=float(raw.get("waist_lock_kp", 80.0)),
            waist_lock_kd=float(raw.get("waist_lock_kd", 2.0)),
            waist_lock_q=float(raw.get("waist_lock_q", 0.0)),

            lock_wrist_motors=bool(raw.get("lock_wrist_motors", False)),
            wrist_motor_indices=raw.get("wrist_motor_indices"),
            wrist_lock_kp=float(raw.get("wrist_lock_kp", 20.0)),
            wrist_lock_kd=float(raw.get("wrist_lock_kd", 0.5)),
            wrist_lock_q=float(raw.get("wrist_lock_q", 0.0)),
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

        # Remote key mapping overrides (some remotes map START/A/SELECT differently)
        self._start_button_idx = int(raw.get("start_button_idx", self._KeyMap.start))
        self._run_button_idx = int(raw.get("run_button_idx", self._KeyMap.A))
        self._stop_button_idx = int(raw.get("stop_button_idx", self._KeyMap.select))

        # DDS init
        try:
            self._ChannelFactoryInitialize(0, self._cfg.net_interface)
        except Exception as e:
            available = _list_available_interfaces()
            available_s = ", ".join(available) if available else "(none)"
            hint = (
                "Failed to initialize Unitree DDS ChannelFactory. "
                f"Requested net_interface={requested_net_interface!r}, resolved to {self._cfg.net_interface!r}. "
                f"Available interfaces: {available_s}. "
                "Fix: set 'net_interface' in your YAML to the NIC connected to the robot (check via 'ip link' or 'ifconfig'), "
                "or set it to 'auto' to let the script pick a reasonable default."
            )
            raise RuntimeError(hint) from e

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

        self._dbg_last_remote_print_t = 0.0
        self._dbg_last_remote_sig: tuple[int, int, int, int, int] | None = None

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
        fixed_indices: set[int] = set()
        if self._cfg.fixed_motors:
            for ent in self._cfg.fixed_motors:
                motor_idx = int(ent["motor_idx"])
                fixed_indices.add(motor_idx)
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

        if self._low_state is None:
            return

        def _lock_target(
            motor_idx: int,
            target_q: float,
            kp: float,
            kd: float,
        ) -> None:
            self._low_cmd.motor_cmd[motor_idx].q = float(target_q)
            self._low_cmd.motor_cmd[motor_idx].qd = 0.0
            self._low_cmd.motor_cmd[motor_idx].kp = float(kp)
            self._low_cmd.motor_cmd[motor_idx].kd = float(kd)
            self._low_cmd.motor_cmd[motor_idx].tau = 0.0

        # Apply group-specific locks first (waist/wrists).
        group_locked: set[int] = set()
        if self._cfg.lock_waist_motors and self._cfg.waist_motor_indices:
            for idx in self._cfg.waist_motor_indices:
                motor_idx = int(idx)
                group_locked.add(motor_idx)
                _lock_target(
                    motor_idx,
                    self._cfg.waist_lock_q,
                    self._cfg.waist_lock_kp,
                    self._cfg.waist_lock_kd,
                )

        if self._cfg.lock_wrist_motors and self._cfg.wrist_motor_indices:
            for idx in self._cfg.wrist_motor_indices:
                motor_idx = int(idx)
                group_locked.add(motor_idx)
                _lock_target(
                    motor_idx,
                    self._cfg.wrist_lock_q,
                    self._cfg.wrist_lock_kp,
                    self._cfg.wrist_lock_kd,
                )

        if not self._cfg.lock_unused_motors:
            return

        try:
            num_motors = len(self._low_state.motor_state)
        except Exception:
            return

        used = set(int(x) for x in self._cfg.joint2motor_idx)
        used |= fixed_indices
        used |= group_locked
        exclude = set(int(x) for x in (self._cfg.lock_unused_exclude or []))

        kp = float(self._cfg.lock_unused_kp)
        kd = float(self._cfg.lock_unused_kd)
        target_q = float(self._cfg.lock_unused_q)

        for motor_idx in range(num_motors):
            if motor_idx in used or motor_idx in exclude:
                continue
            _lock_target(motor_idx, target_q, kp, kd)

    def _maybe_print_remote_debug(self) -> None:
        if not self._cfg.debug_remote:
            return

        now = time.monotonic()
        if (now - self._dbg_last_remote_print_t) < max(self._cfg.debug_remote_interval_s, 0.05):
            return

        # buttons packed as bitmask for stable change detection
        buttons = getattr(self._remote, "button", [0] * 16)
        mask = 0
        for i, v in enumerate(buttons[:16]):
            if int(v) == 1:
                mask |= (1 << i)

        # quantize sticks to reduce spam from tiny jitter
        lx = int(round(float(getattr(self._remote, "lx", 0.0)) * 100))
        ly = int(round(float(getattr(self._remote, "ly", 0.0)) * 100))
        rx = int(round(float(getattr(self._remote, "rx", 0.0)) * 100))
        ry = int(round(float(getattr(self._remote, "ry", 0.0)) * 100))
        sig = (mask, lx, ly, rx, ry)

        if sig != self._dbg_last_remote_sig:
            self._dbg_last_remote_sig = sig

        start = int(buttons[self._start_button_idx]) if len(buttons) > self._start_button_idx else 0
        select = int(buttons[self._stop_button_idx]) if len(buttons) > self._stop_button_idx else 0
        a_btn = int(buttons[self._run_button_idx]) if len(buttons) > self._run_button_idx else 0

        tick = int(getattr(self._low_state, "tick", 0)) if self._low_state is not None else 0
        print(
            f"[unitree_sdk2][remote] tick={tick} state={self._state} "
            f"start={start} A={a_btn} select={select} "
            f"lx={lx/100:.2f} ly={ly/100:.2f} rx={rx/100:.2f} ry={ry/100:.2f} "
            f"mask=0x{mask:04x}"
        )
        self._dbg_last_remote_print_t = now

    def recv_obs(self) -> ObsPacket:
        self._maybe_print_remote_debug()
        if self._cfg.require_remote and self._state == "RUN":
            if self._remote.button[self._stop_button_idx] == 1:
                self._create_damping_cmd(self._low_cmd)
                self._send_cmd()
                raise StopDeploy("Remote select pressed")

        # Safety state machine: send appropriate lowcmd even before RUN.
        if self._cfg.require_remote:
            if self._state == "ZERO":
                if self._remote.button[self._start_button_idx] != 1:
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
                if self._remote.button[self._run_button_idx] != 1:
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
