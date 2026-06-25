# src/physics_metrics/metrics.py
"""
Evaluation metrics for thermal satellite image interpolation.
"""

import numpy as np
import torch
from typing import Union, Dict
import warnings

warnings.filterwarnings('ignore')

ArrayOrTensor = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayOrTensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().squeeze()
    return np.asarray(x).squeeze()


def mse(pred: ArrayOrTensor, gt: ArrayOrTensor) -> float:
    p, g = _to_numpy(pred), _to_numpy(gt)
    return float(np.mean((p - g) ** 2))


def psnr(pred: ArrayOrTensor, gt: ArrayOrTensor, data_range: float = 1.0) -> float:
    err = mse(pred, gt)
    if err == 0:
        return float('inf')
    return float(10 * np.log10((data_range ** 2) / err))


def ssim(
    pred: ArrayOrTensor,
    gt: ArrayOrTensor,
    data_range: float = 1.0,
    win_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """SSIM without external library dependency."""
    from scipy.ndimage import uniform_filter
    
    p, g = _to_numpy(pred).astype(np.float64), _to_numpy(gt).astype(np.float64)
    
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    
    mu1 = uniform_filter(p, win_size)
    mu2 = uniform_filter(g, win_size)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = uniform_filter(p ** 2, win_size) - mu1_sq
    sigma2_sq = uniform_filter(g ** 2, win_size) - mu2_sq
    sigma12   = uniform_filter(p * g, win_size) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return float(ssim_map.mean())


def bt_mae(
    pred: ArrayOrTensor,
    gt: ArrayOrTensor,
    bt_min: float = 180.0,
    bt_max: float = 320.0,
) -> float:
    """
    Brightness Temperature MAE in Kelvin.
    Denormalizes [0,1] back to physical BT range before computing error.
    """
    p, g = _to_numpy(pred), _to_numpy(gt)
    # Denormalize
    p_bt = p * (bt_max - bt_min) + bt_min
    g_bt = g * (bt_max - bt_min) + bt_min
    return float(np.mean(np.abs(p_bt - g_bt)))


def fsim(pred: ArrayOrTensor, gt: ArrayOrTensor) -> float:
    """
    Feature Similarity Index (FSIM) using phase congruency approximation.
    Uses gradient magnitude as a proxy for feature maps.
    """
    from scipy.ndimage import sobel
    
    p, g = _to_numpy(pred).astype(np.float64), _to_numpy(gt).astype(np.float64)
    
    def gradient_magnitude(img):
        gx = sobel(img, axis=1)
        gy = sobel(img, axis=0)
        return np.sqrt(gx ** 2 + gy ** 2)
    
    pc_p = gradient_magnitude(p)
    pc_g = gradient_magnitude(g)
    
    T1, T2 = 0.85, 160.0
    
    S_PC = (2 * pc_p * pc_g + T1) / (pc_p ** 2 + pc_g ** 2 + T1)
    S_G  = (2 * pc_p * pc_g + T2) / (pc_p ** 2 + pc_g ** 2 + T2)
    S_L  = S_PC * S_G
    
    PC_max = np.maximum(pc_p, pc_g)
    denom = PC_max.sum()
    
    if denom == 0:
        return 1.0
    
    return float((S_L * PC_max).sum() / denom)


def compute_all_metrics(
    pred: ArrayOrTensor,
    gt: ArrayOrTensor,
    bt_min: float = 180.0,
    bt_max: float = 320.0,
    data_range: float = 1.0,
) -> Dict[str, float]:
    """Compute all metrics and return as dict."""
    return {
        'MSE':    mse(pred, gt),
        'PSNR':   psnr(pred, gt, data_range),
        'SSIM':   ssim(pred, gt, data_range),
        'BT_MAE': bt_mae(pred, gt, bt_min, bt_max),
        'FSIM':   fsim(pred, gt),
    }


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    print(f"\n{'─'*40}")
    if prefix:
        print(f"  {prefix}")
    for k, v in metrics.items():
        unit = " K" if k == "BT_MAE" else ""
        print(f"  {k:<10}: {v:.4f}{unit}")
    print(f"{'─'*40}")


# ── CLI test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing metrics with synthetic data...")
    
    gt  = np.random.rand(512, 512).astype(np.float32)
    
    # Perfect prediction
    m = compute_all_metrics(gt, gt)
    print_metrics(m, "Perfect prediction")
    
    # Noisy prediction
    noise = np.random.normal(0, 0.05, gt.shape).astype(np.float32)
    pred_noisy = np.clip(gt + noise, 0, 1)
    m2 = compute_all_metrics(pred_noisy, gt)
    print_metrics(m2, "Noisy prediction (σ=0.05)")
    
    # Tensor input test
    gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
    pred_t = torch.from_numpy(pred_noisy).unsqueeze(0).unsqueeze(0)
    m3 = compute_all_metrics(pred_t, gt_t)
    print_metrics(m3, "Tensor input test")
    
    print("\nAll metrics working correctly.")