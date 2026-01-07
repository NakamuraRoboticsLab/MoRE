from __future__ import annotations

import importlib
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

try:
    from legged_gym import LEGGED_GYM_ROOT_DIR
except ModuleNotFoundError:
    # Allow running this file directly without installing the package.
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from legged_gym import LEGGED_GYM_ROOT_DIR


def get_gravity_orientation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    qw = float(quaternion_wxyz[0])
    qx = float(quaternion_wxyz[1])
    qy = float(quaternion_wxyz[2])
    qz = float(quaternion_wxyz[3])

    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def _maybe_import_cv2() -> Any:
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


@dataclass(frozen=True)
class DeployConfig:
    policy_path: str

    control_hz: float
    run_duration_s: float

    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    action_scale: float
    cmd_scale: np.ndarray

    num_actions: int
    num_obs: int
    obs_history_len: int

    default_angles: np.ndarray

    cmd_init: np.ndarray
    gait_cmd: np.ndarray

    # Optional PD gains for sim-to-real (hardware side can apply them)
    kps: np.ndarray | None
    kds: np.ndarray | None

    # Raw config passthrough for hardware-specific interfaces
    extra: dict[str, Any]

    depth_image_is_normalized: bool
    depth_far_clip: float
    depth_near_clip: float
    depth_buffer_len: int
    cam_update_interval: int

    crop_image: bool
    crop_size: np.ndarray

    gaussian_filter: bool
    gaussian_filter_kernel: int
    gaussian_filter_sigma: float

    gaussian_noise: bool
    gaussian_noise_std: float

    depth_dis_noise: float

    visualize_depth: bool


def load_config(path: str) -> DeployConfig:
    with open(path, "r") as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)

    def _p(key: str, default: Any = None) -> Any:
        if key in raw:
            return raw[key]
        if default is not None:
            return default
        raise KeyError(f"Missing config key: {key}")

    policy_path = str(_p("policy_path")).replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    return DeployConfig(
        policy_path=policy_path,
        control_hz=float(_p("control_hz")),
        run_duration_s=float(_p("run_duration_s", 0)),
        ang_vel_scale=float(_p("ang_vel_scale")),
        dof_pos_scale=float(_p("dof_pos_scale")),
        dof_vel_scale=float(_p("dof_vel_scale")),
        action_scale=float(_p("action_scale")),
        cmd_scale=np.array(_p("cmd_scale"), dtype=np.float32),
        num_actions=int(_p("num_actions")),
        num_obs=int(_p("num_obs")),
        obs_history_len=int(_p("obs_history_len")),
        default_angles=np.array(_p("default_angles"), dtype=np.float32),
        cmd_init=np.array(_p("cmd_init"), dtype=np.float32),
        gait_cmd=np.array(_p("gait_cmd"), dtype=np.float32),
        kps=np.array(raw["kps"], dtype=np.float32) if "kps" in raw else None,
        kds=np.array(raw["kds"], dtype=np.float32) if "kds" in raw else None,
        extra=dict(raw),
        depth_image_is_normalized=bool(_p("depth_image_is_normalized", False)),
        depth_far_clip=float(_p("depth_far_clip")),
        depth_near_clip=float(_p("depth_near_clip")),
        depth_buffer_len=int(_p("depth_buffer_len")),
        cam_update_interval=int(_p("cam_update_interval")),
        crop_image=bool(_p("crop_image", False)),
        crop_size=np.array(_p("crop_size", [0, 0, 0, 0]), dtype=np.int32),
        gaussian_filter=bool(_p("gaussian_filter", False)),
        gaussian_filter_kernel=int(_p("gaussian_filter_kernel", 5)),
        gaussian_filter_sigma=float(_p("gaussian_filter_sigma", 1.0)),
        gaussian_noise=bool(_p("gaussian_noise", False)),
        gaussian_noise_std=float(_p("gaussian_noise_std", 0.0)),
        depth_dis_noise=float(_p("depth_dis_noise", 0.0)),
        visualize_depth=bool(_p("visualize_depth", False)),
    )


def process_depth_image_np(depth_image: np.ndarray, cfg: DeployConfig) -> torch.Tensor:
    """Match MuJoCo deploy preprocessing.

    Output is torch tensor (H, W) float32 normalized to [-0.5, 0.5].
    """
    depth = depth_image.astype(np.float32, copy=True)
    depth += cfg.depth_dis_noise * 2 * (np.random.rand(1).astype(np.float32) - 0.5)
    if cfg.gaussian_noise and cfg.gaussian_noise_std > 0:
        depth += cfg.gaussian_noise_std * np.random.randn(*depth.shape).astype(np.float32)
    depth = np.clip(depth, cfg.depth_near_clip, cfg.depth_far_clip)

    if cfg.depth_image_is_normalized:
        depth_norm = depth
    else:
        denom = (cfg.depth_far_clip - cfg.depth_near_clip)
        denom = denom if denom != 0 else 1.0
        depth_norm = (depth - cfg.depth_near_clip) / denom - 0.5

    depth_t = torch.from_numpy(depth_norm)

    if cfg.crop_image:
        clip_left, clip_top, clip_right, clip_bottom = [int(x) for x in cfg.crop_size]
        h, w = depth_t.shape
        left = clip_left
        right = w - clip_right
        top = clip_top
        bottom = h - clip_bottom
        depth_t = depth_t[top:bottom, left:right]
        depth_t = F.interpolate(
            depth_t.unsqueeze(0).unsqueeze(0),
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    else:
        if tuple(depth_t.shape) != (64, 64):
            depth_t = F.interpolate(
                depth_t.unsqueeze(0).unsqueeze(0),
                size=(64, 64),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

    if cfg.gaussian_filter:
        cv2 = _maybe_import_cv2()
        if cv2 is not None:
            kernel = int(cfg.gaussian_filter_kernel)
            sigma = float(cfg.gaussian_filter_sigma)
            img = cv2.GaussianBlur(depth_t.numpy(), (kernel, kernel), sigma)
            depth_t = torch.from_numpy(img)

    return depth_t.to(dtype=torch.float32)


def load_interface(import_path: str, cfg: DeployConfig):
    """Load interface from `module:ClassName` or built-in alias.

    Built-in:
    - dummy -> deploy.deploy_real.interfaces.dummy:DummyRealInterface
    """
    if import_path == "dummy":
        import_path = "deploy.deploy_real.interfaces.dummy:DummyRealInterface"
    if import_path == "unitree_sdk2":
        import_path = "deploy.deploy_real.interfaces.unitree_sdk2:UnitreeSdk2Interface"

    if ":" not in import_path:
        raise ValueError("interface must be in form module:ClassName or 'dummy'")

    module_name, class_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    # Allow the class to accept `cfg` or `DummyConfig`.
    try:
        return cls(cfg)
    except TypeError:
        from deploy.deploy_real.interfaces.dummy import DummyConfig

        dummy_cfg = DummyConfig(
            num_actions=cfg.num_actions,
            num_gaits=int(cfg.gait_cmd.shape[0]),
            cmd_init=cfg.cmd_init,
            gait_cmd=cfg.gait_cmd,
        )
        return cls(dummy_cfg)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        type=str,
        help="YAML file name under deploy/deploy_real/configs/",
    )
    parser.add_argument(
        "--interface",
        type=str,
        default="dummy",
        help="Interface import path module:ClassName or 'dummy'",
    )
    args = parser.parse_args()

    cfg_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config_file}"
    cfg = load_config(cfg_path)

    interface = load_interface(args.interface, cfg)

    torch.set_grad_enabled(False)
    policy = torch.jit.load(cfg.policy_path, map_location="cpu")
    policy.eval()

    num_gaits = int(cfg.gait_cmd.shape[0])

    action = np.zeros(cfg.num_actions, dtype=np.float32)
    obs = np.zeros(cfg.num_obs, dtype=np.float32)

    trajectory_history = torch.zeros(size=(1, cfg.obs_history_len, cfg.num_obs - num_gaits), dtype=torch.float32)
    depth_image_buffer = torch.zeros(1, cfg.depth_buffer_len, 64, 64, dtype=torch.float32)
    depth_buf_initialized = False

    control_dt = 1.0 / cfg.control_hz
    start_wall = time.monotonic()
    next_tick = start_wall
    tick = 0

    from deploy.deploy_real.interfaces import StopDeploy

    try:
        while True:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick += control_dt

            if cfg.run_duration_s > 0 and (time.monotonic() - start_wall) > cfg.run_duration_s:
                break

            try:
                pkt = interface.recv_obs()
            except StopDeploy:
                break

            qj = np.asarray(pkt.qj, dtype=np.float32)
            dqj = np.asarray(pkt.dqj, dtype=np.float32)
            quat = np.asarray(pkt.quat_wxyz, dtype=np.float32)
            omega = np.asarray(pkt.omega_xyz, dtype=np.float32)
            cmd = np.asarray(pkt.cmd, dtype=np.float32)
            gait_cmd = np.asarray(pkt.gait_cmd, dtype=np.float32)

            if qj.shape[0] != cfg.num_actions or dqj.shape[0] != cfg.num_actions:
                raise ValueError(f"qj/dqj size mismatch: expected {cfg.num_actions}")
            if quat.shape[0] != 4 or omega.shape[0] != 3:
                raise ValueError("quat_wxyz must be (4,), omega_xyz must be (3,)")
            if cmd.shape[0] != 3:
                raise ValueError("cmd must be (3,)")
            if gait_cmd.shape[0] != num_gaits:
                raise ValueError(f"gait_cmd must be ({num_gaits},)")

            # Build obs (identical layout to MuJoCo deploy)
            qj_scaled = (qj - cfg.default_angles) * cfg.dof_pos_scale
            dqj_scaled = dqj * cfg.dof_vel_scale
            gravity_orientation = get_gravity_orientation(quat)
            omega_scaled = omega * cfg.ang_vel_scale

            obs[:num_gaits] = gait_cmd
            obs[num_gaits : num_gaits + 3] = cmd * cfg.cmd_scale
            obs[num_gaits + 3 : num_gaits + 6] = omega_scaled
            obs[num_gaits + 6 : num_gaits + 9] = gravity_orientation

            base = num_gaits + 9
            obs[base : base + cfg.num_actions] = qj_scaled
            obs[base + cfg.num_actions : base + 2 * cfg.num_actions] = dqj_scaled
            obs[base + 2 * cfg.num_actions : base + 3 * cfg.num_actions] = action

            # Depth buffer update (optional)
            if (tick % cfg.cam_update_interval) == 0 and pkt.depth_image is not None:
                depth_t = process_depth_image_np(np.asarray(pkt.depth_image), cfg)
                if not depth_buf_initialized:
                    depth_image_buffer = torch.stack([depth_t] * cfg.depth_buffer_len, dim=0).unsqueeze(0)
                    depth_buf_initialized = True
                else:
                    depth_image_buffer = torch.cat(
                        [depth_image_buffer[:, 1:, ...], depth_t.unsqueeze(0).unsqueeze(1)],
                        dim=1,
                    )

                if cfg.visualize_depth:
                    cv2 = _maybe_import_cv2()
                    if cv2 is not None:
                        cv2.namedWindow("depth image", cv2.WINDOW_NORMAL)
                        cv2.imshow("depth image", (depth_image_buffer[0, -1].numpy() + 0.5))
                        cv2.waitKey(1)

            # Trajectory history (exclude gait_cmd part)
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            trajectory_history = torch.cat([trajectory_history[:, 1:], obs_tensor.unsqueeze(1)[..., num_gaits:]], dim=1)

            # Policy inference
            # Keep the exact depth slice as MuJoCo deploy (frames 7:9 when buffer_len=10)
            depth_slice = depth_image_buffer[:, 7:9, ...] if depth_buf_initialized else depth_image_buffer[:, 0:2, ...]
            action = policy(obs_tensor, trajectory_history, depth_slice).detach().cpu().numpy().squeeze().astype(np.float32)

            target_dof_pos = action * cfg.action_scale + cfg.default_angles
            interface.send_target_joint_pos(target_dof_pos, kp=cfg.kps, kd=cfg.kds)

            tick += 1

    finally:
        try:
            interface.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
