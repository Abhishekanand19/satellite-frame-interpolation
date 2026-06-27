import base64, io, logging, os
from typing import Optional
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger("utils")

BG = "#0b0f19"

def arr_to_b64(
    arr: np.ndarray,
    cmap: str = "inferno_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: tuple = (6, 6),
    dpi: int = 120,
) -> str:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=BG)
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                pad_inches=0, facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    from matplotlib.colors import hsv_to_rgb
    u = flow[0] if flow.ndim == 3 else flow
    v = flow[1] if flow.ndim == 3 else np.zeros_like(flow)
    mag   = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)
    hsv   = np.stack([
        (angle + np.pi) / (2 * np.pi),
        np.ones_like(mag),
        mag / (mag.max() + 1e-8),
    ], axis=-1)
    return hsv_to_rgb(hsv)


def denorm_bt(arr: np.ndarray, bt_min=180.0, bt_max=320.0) -> np.ndarray:
    return (arr * (bt_max - bt_min) + bt_min).astype(np.float32)


def safe_float(v) -> float:
    try:
        f = float(v)
        return 0.0 if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except Exception:
        return 0.0
