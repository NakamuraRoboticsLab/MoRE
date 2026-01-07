from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TeleImagerDepthConfig:
    host: str
    request_port: int = 60000
    cam_topic: str = "head_camera"

    # Teleimager depth stream is published as uint16 (z16). Convert to meters via this scale.
    # For Intel RealSense, this is often 0.001 (mm->m) or device-specific; if you know the
    # server-side RealSense depth_scale, set it here.
    depth_unit_scale_m: float = 0.001

    # Optional downsample before feeding to policy.
    # If set, output will be (H, W). Leave None to keep original camera resolution.
    downsample_hw: Optional[Tuple[int, int]] = None


def downsample_depth(depth: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Downsample a depth image to out_hw.

    Uses area-resampling when possible.
    Input: float32 array (H, W)
    Output: float32 array (out_h, out_w)
    """
    depth_f = np.asarray(depth, dtype=np.float32)
    out_h, out_w = int(out_hw[0]), int(out_hw[1])

    try:
        import cv2  # type: ignore

        return cv2.resize(depth_f, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    except Exception:
        # Fallback: torch area interpolation
        try:
            import torch
            import torch.nn.functional as F

            t = torch.from_numpy(depth_f).unsqueeze(0).unsqueeze(0)
            t2 = F.interpolate(t, size=(out_h, out_w), mode="area")
            return t2.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        except Exception:
            # Last resort: naive stride sampling
            in_h, in_w = depth_f.shape
            ys = (np.linspace(0, in_h - 1, out_h)).astype(np.int32)
            xs = (np.linspace(0, in_w - 1, out_w)).astype(np.int32)
            return depth_f[np.ix_(ys, xs)].astype(np.float32)


class TeleImagerDepthClient:
    """Thin wrapper around `teleimager.image_client.ImageClient` for depth frames."""

    def __init__(self, cfg: TeleImagerDepthConfig):
        self._cfg = cfg
        try:
            from teleimager.image_client import ImageClient  # type: ignore

            self._client = ImageClient(host=cfg.host, request_port=int(cfg.request_port))
        except Exception as e:
            raise ImportError(
                "teleimager is required for TeleImagerDepthClient. "
                "Make sure `teleimager` is installed (or `teleimager/src` is on PYTHONPATH) and ZMQ deps are available."
            ) from e

    def get_depth_meters(self) -> Optional[np.ndarray]:
        depth_u16, _fps = self._client.get_depth_frame(self._cfg.cam_topic)
        if depth_u16 is None:
            return None

        depth_m = depth_u16.astype(np.float32) * float(self._cfg.depth_unit_scale_m)

        if self._cfg.downsample_hw is not None:
            depth_m = downsample_depth(depth_m, self._cfg.downsample_hw)

        return depth_m

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
