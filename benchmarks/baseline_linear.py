# benchmarks/baseline_linear.py
"""Linear interpolation baseline — pixel-wise alpha blend between t0 and t2."""

import numpy as np


class LinearInterpolator:
    """Baseline: simple pixel-wise linear blend at t=0.5."""

    name = "Linear Interpolation"

    def predict(self, t0: np.ndarray, t2: np.ndarray, timestep: float = 0.5) -> np.ndarray:
        return ((1 - timestep) * t0 + timestep * t2).astype(np.float32)