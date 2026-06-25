# src/preprocessing/goes_preprocessor.py
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# ── Constants ──────────────────────────────────────────────────────────────
BT_MIN = 180.0   # K  (cold cloud tops)
BT_MAX = 320.0   # K  (warm surface)
TARGET_SIZE = (512, 512)   # H x W for model input


# ── Core functions ─────────────────────────────────────────────────────────

def load_bt_from_nc(nc_path: str) -> np.ndarray:
    """Load brightness temperature array from GOES .nc file. Returns float32 HxW."""
    ds = xr.open_dataset(nc_path, engine='netcdf4')
    for candidate in ['CMI', 'Rad', 'BT', 'brightness_temperature']:
        if candidate in ds.data_vars:
            bt = ds[candidate].values.astype(np.float32)
            ds.close()
            if bt.ndim == 3:
                bt = bt[0]  # drop time dim if present
            return bt
    # fallback
    var = list(ds.data_vars)[0]
    bt = ds[var].values.astype(np.float32)
    ds.close()
    if bt.ndim == 3:
        bt = bt[0]
    return bt


def normalize_bt(
    bt: np.ndarray,
    bt_min: float = BT_MIN,
    bt_max: float = BT_MAX,
    clip: bool = True,
) -> np.ndarray:
    """
    Normalize BT to [0, 1] using physical temperature range.
    Handles NaN by replacing with bt_min.
    Returns float32.
    """
    bt = bt.astype(np.float32)
    bt = np.nan_to_num(bt, nan=bt_min)
    if clip:
        bt = np.clip(bt, bt_min, bt_max)
    normalized = (bt - bt_min) / (bt_max - bt_min)
    return normalized.astype(np.float32)


def denormalize_bt(
    normalized: np.ndarray,
    bt_min: float = BT_MIN,
    bt_max: float = BT_MAX,
) -> np.ndarray:
    """Reverse normalize back to Kelvin."""
    return (normalized * (bt_max - bt_min) + bt_min).astype(np.float32)


def resize_array(
    arr: np.ndarray,
    target_size: Tuple[int, int] = TARGET_SIZE,
    mode: str = 'bilinear',
) -> np.ndarray:
    """
    Resize 2D float32 array to target_size (H, W).
    Uses torch interpolation for quality.
    """
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
    t = F.interpolate(t, size=target_size, mode=mode, align_corners=False)
    return t.squeeze().numpy().astype(np.float32)


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    """
    Convert HxW float32 numpy array to torch tensor shape (1, 1, H, W).
    Suitable for single-channel model input.
    """
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # B=1, C=1, H, W


def preprocess_frame(
    nc_path: str,
    target_size: Tuple[int, int] = TARGET_SIZE,
    bt_min: float = BT_MIN,
    bt_max: float = BT_MAX,
    return_tensor: bool = True,
) -> Tuple:
    """
    Full pipeline: load → normalize → resize → tensor.
    Returns: (tensor or array, raw_bt, normalized_bt)
    """
    raw_bt = load_bt_from_nc(nc_path)
    normalized = normalize_bt(raw_bt, bt_min, bt_max)
    resized = resize_array(normalized, target_size)
    
    if return_tensor:
        return to_tensor(resized), raw_bt, normalized
    return resized, raw_bt, normalized


def preprocess_batch(
    nc_paths: list,
    target_size: Tuple[int, int] = TARGET_SIZE,
    bt_min: float = BT_MIN,
    bt_max: float = BT_MAX,
) -> torch.Tensor:
    """
    Process a list of .nc paths → batched tensor (B, 1, H, W).
    """
    frames = []
    for p in nc_paths:
        tensor, _, _ = preprocess_frame(p, target_size, bt_min, bt_max)
        frames.append(tensor)  # each: (1, 1, H, W)
    return torch.cat(frames, dim=0)  # (B, 1, H, W)


# ── CLI test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, matplotlib.pyplot as plt

    search_dirs = ["data/goes19/raw/day001", "data/goes19/raw/day002"]
    nc_files = []
    for d in search_dirs:
        nc_files += sorted(Path(d).glob("*.nc"))[:2]

    if not nc_files:
        print("No .nc files found.")
        sys.exit(1)

    nc_path = str(nc_files[0])
    print(f"Testing with: {nc_path}")

    tensor, raw_bt, norm = preprocess_frame(nc_path)
    print(f"Raw BT   → shape: {raw_bt.shape}, range: [{raw_bt.min():.1f}, {raw_bt.max():.1f}] K")
    print(f"Norm     → shape: {norm.shape}, range: [{norm.min():.4f}, {norm.max():.4f}]")
    print(f"Tensor   → shape: {tensor.shape}, dtype: {tensor.dtype}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(raw_bt, cmap='inferno_r')
    axes[0].set_title(f'Raw BT\n[{raw_bt.min():.0f}K – {raw_bt.max():.0f}K]')
    axes[1].imshow(tensor.squeeze().numpy(), cmap='inferno_r', vmin=0, vmax=1)
    axes[1].set_title(f'Preprocessed {TARGET_SIZE}\n[0–1 normalized]')
    plt.tight_layout()
    out = Path("outputs/preprocess_test.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")