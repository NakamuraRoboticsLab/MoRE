# deploy_real

This folder contains a *real-world* deploy loop that mirrors the observation layout and policy call signature used by [deploy/deploy_mujoco/deploy_mujoco_with_resi.py](../deploy_mujoco/deploy_mujoco_with_resi.py).

Key difference: **the observation inputs come from an external interface** (robot SDK / middleware), instead of MuJoCo state.

## What it expects

The deploy loop builds the same observation vector as MuJoCo deploy:

- `gait_cmd` (length = `num_gaits`)
- `cmd` (vx, vy, wz) scaled by `cmd_scale`
- `omega` (base ang vel) scaled by `ang_vel_scale`
- `gravity_orientation` computed from `quat_wxyz`
- `qj` and `dqj` (joint pos/vel) scaled by `dof_pos_scale` / `dof_vel_scale`
- previous `action`

Optional:
- `depth_image` (64x64 or any HxW; script will resize/crop) processed the same way as MuJoCo deploy.

## Run (smoke test)

From repo root:

```bash
python deploy/deploy_real/deploy_real_with_resi.py g1_16dof_resi_moe_real.yaml --interface dummy
```

## Run on a real robot

### Option A: Unitree SDK2 (DDS) interface (unitree_ref-style)

Use the provided Unitree DDS interface:

```bash
python deploy/deploy_real/deploy_real_with_resi.py g1_16dof_resi_moe_real_unitree_sdk2.yaml --interface unitree_sdk2
```

Notes:
- Edit `net_interface`, `msg_type`, and especially `joint2motor_idx` in the YAML to match your robot.
- This path requires `unitree_sdk2py` installed and a working DDS connection.

## Plug in your own hardware interface

Implement the interface defined in [deploy/deploy_real/interfaces/base.py](interfaces/base.py):

- `recv_obs() -> ObsPacket`
- `send_target_joint_pos(target_q: np.ndarray) -> None`

Then run:

```bash
python deploy/deploy_real/deploy_real_with_resi.py g1_16dof_resi_moe_real.yaml \
  --interface your_module.your_iface:YourInterface
```

Your class can accept either the full deploy config object or your own arguments; the loader tries to pass `cfg` first.

## Depth / vision interface

Depth is carried through the `ObsPacket.depth_image` field returned by your interface's `recv_obs()`.

- Definition: [deploy/deploy_real/interfaces/base.py](interfaces/base.py)
- Processing (clip/normalize/crop/resize): `process_depth_image_np()` in [deploy/deploy_real/deploy_real_with_resi.py](deploy_real_with_resi.py)

Expected format:
- `depth_image`: `np.ndarray` float32 with shape `(H, W)`.
- If it is raw metric depth (meters), keep `depth_image_is_normalized: false` and set `depth_near_clip/depth_far_clip`.
- If you already provide normalized depth in `[-0.5, 0.5]`, set `depth_image_is_normalized: true`.

### Using TeleImager depth client

This repo includes `teleimager/` which can publish depth as ZMQ uint16 (z16) and a python client.

If you use `--interface unitree_sdk2`, you can optionally attach TeleImager depth by setting these YAML keys:
- `teleimager_host`
- `teleimager_request_port` (default 60000)
- `teleimager_cam_topic` (e.g. `head_camera`)
- `teleimager_depth_unit_scale_m` (uint16 -> meters, commonly 0.001)
- `teleimager_downsample_hw` (optional, set to `[64, 64]` to directly downsample to policy input size)

Code:
- TeleImager depth wrapper: [deploy/deploy_real/interfaces/teleimager_depth.py](interfaces/teleimager_depth.py)
- Unitree interface attaching depth: [deploy/deploy_real/interfaces/unitree_sdk2.py](interfaces/unitree_sdk2.py)
