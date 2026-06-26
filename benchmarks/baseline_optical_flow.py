# benchmarks/baseline_optical_flow.py
"""Optical flow baseline using OpenCV Farneback dense flow."""

import logging
import numpy as np

logger = logging.getLogger("baseline_flow")


class OpticalFlowInterpolator:
    """Farneback dense optical flow baseline."""

    name = "Optical Flow (Farneback)"

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
    ):
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            raise ImportError("opencv-python required: pip install opencv-python")

        self.params = dict(
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=0,
        )

    def _to_uint8(self, arr: np.ndarray) -> np.ndarray:
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)

    def _warp(self, img: np.ndarray, flow: np.ndarray) -> np.ndarray:
        H, W = img.shape
        map_x, map_y = np.meshgrid(np.arange(W, dtype=np.float32),
                                    np.arange(H, dtype=np.float32))
        map_x += flow[..., 0]
        map_y += flow[..., 1]
        warped = self.cv2.remap(
            img.astype(np.float32), map_x, map_y,
            interpolation=self.cv2.INTER_LINEAR,
            borderMode=self.cv2.BORDER_REPLICATE,
        )
        return warped

    def predict(self, t0: np.ndarray, t2: np.ndarray, timestep: float = 0.5) -> np.ndarray:
        t0_u8 = self._to_uint8(t0)
        t2_u8 = self._to_uint8(t2)

        flow_fwd = self.cv2.calcOpticalFlowFarneback(t0_u8, t2_u8, None, **self.params)
        flow_bwd = self.cv2.calcOpticalFlowFarneback(t2_u8, t0_u8, None, **self.params)

        flow_t_fwd =  timestep * flow_fwd
        flow_t_bwd = -(1 - timestep) * flow_bwd

        warped_fwd = self._warp(t0, flow_t_fwd)
        warped_bwd = self._warp(t2, flow_t_bwd)

        blended = (1 - timestep) * warped_fwd + timestep * warped_bwd
        return np.clip(blended, 0, 1).astype(np.float32)