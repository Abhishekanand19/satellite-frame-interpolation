# dashboard/backend/services.py
"""
Service layer — works standalone (synthetic) OR with full ML pipeline.
GPU/src/ imports are optional — dashboard runs anywhere.
"""
import logging, time, sys
from pathlib import Path
from typing import Optional
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.backend.config import settings
from dashboard.backend.utils import arr_to_b64, flow_to_rgb, denorm_bt, safe_float

logger = logging.getLogger("services")

# ── capability flags ──────────────────────────────────────────────────────────
_HAS_SRC   = False
_HAS_MODEL = False
_model     = None
_files     = []
_triplets  = []

try:
    from src.interpolation.rife_model import RIFEThermalInterpolator
    from src.data_loader.goes_dataset import discover_nc_files, build_triplets
    from src.preprocessing.goes_preprocessor import preprocess_frame
    from src.physics_metrics.metrics import compute_all_metrics
    _HAS_SRC = True
    logger.info("src/ pipeline available")
except Exception as e:
    logger.warning("src/ not available — synthetic mode (%s)", e)


def _get_model():
    global _model, _HAS_MODEL
    if not _HAS_SRC:
        return None
    if _model is None:
        try:
            ckpt = Path(settings.checkpoint_path)
            _model = RIFEThermalInterpolator(
                checkpoint_path=str(ckpt) if ckpt.exists() else None
            )
            _HAS_MODEL = True
            logger.info("Model loaded")
        except Exception as e:
            logger.warning("Model load failed: %s", e)
    return _model


def _get_files():
    global _files
    if not _HAS_SRC or _files:
        return _files
    try:
        _files = discover_nc_files(settings.data_root)
    except Exception as e:
        logger.warning("File discovery failed: %s", e)
        _files = []
    return _files


def _get_triplets(stride: int):
    global _triplets
    if not _HAS_SRC:
        return []
    try:
        files = _get_files()
        _triplets = build_triplets(files, stride=stride)
    except Exception as e:
        logger.warning("Triplet build failed: %s", e)
        _triplets = []
    return _triplets


# ── synthetic data (demo mode) ────────────────────────────────────────────────
def _make_synthetic_frames():
    """Generate realistic-looking synthetic thermal frames."""
    H, W = 256, 256
    x, y = np.meshgrid(np.linspace(0, 4*np.pi, W), np.linspace(0, 4*np.pi, H))

    # Simulate cloud-like BT patterns
    t0 = np.clip(
        0.4 + 0.3 * np.sin(x) * np.cos(y)
            + 0.15 * np.sin(2*x + 0.5)
            + 0.05 * np.random.randn(H, W),
        0, 1
    ).astype(np.float32)

    t2 = np.clip(
        0.4 + 0.3 * np.sin(x + 0.3) * np.cos(y + 0.2)
            + 0.15 * np.sin(2*x + 0.8)
            + 0.05 * np.random.randn(H, W),
        0, 1
    ).astype(np.float32)

    pred = np.clip((t0 * 0.5 + t2 * 0.5)
                   + 0.02 * np.random.randn(H, W), 0, 1).astype(np.float32)
    gt   = np.clip((t0 * 0.5 + t2 * 0.5)
                   + 0.01 * np.random.randn(H, W), 0, 1).astype(np.float32)
    return t0, t2, pred, gt


def _synthetic_result() -> dict:
    t0, t2, pred, gt = _make_synthetic_frames()
    H, W = pred.shape
    diff = np.abs(pred - gt)
    conf = np.clip(1 - diff * 4, 0, 1).astype(np.float32)
    u    = np.random.randn(H, W).astype(np.float32) * 0.3
    v    = np.random.randn(H, W).astype(np.float32) * 0.3
    flow = np.stack([u, v])

    psnr_val  = safe_float(28.5 + np.random.rand() * 4)
    ssim_val  = safe_float(0.87 + np.random.rand() * 0.08)
    mae_val   = safe_float(0.025 + np.random.rand() * 0.015)
    rmse_val  = safe_float(0.035 + np.random.rand() * 0.015)
    btmae_val = safe_float(3.1  + np.random.rand() * 2.5)
    fsim_val  = safe_float(0.90 + np.random.rand() * 0.06)
    conf_val  = safe_float(float(conf.mean()))
    coverage  = safe_float(float(np.mean(diff > 0.05) * 100))

    return {
        "images": {
            "t0":           arr_to_b64(t0,              "inferno_r", 0, 1),
            "t2":           arr_to_b64(t2,              "inferno_r", 0, 1),
            "t1_gt":        arr_to_b64(gt,              "inferno_r", 0, 1),
            "prediction":   arr_to_b64(pred,            "inferno_r", 0, 1),
            "diff_heatmap": arr_to_b64(diff,            "hot",       0, max(diff.max(), 1e-6)),
            "confidence":   arr_to_b64(conf,            "viridis",   0, 1),
            "optical_flow": arr_to_b64(flow_to_rgb(flow), None, None, None),
        },
        "metrics": {
            "PSNR":       psnr_val,
            "SSIM":       ssim_val,
            "MAE":        mae_val,
            "RMSE":       rmse_val,
            "BT_MAE":     btmae_val,
            "FSIM":       fsim_val,
            "confidence": conf_val,
        },
        "weather": {
            "cloud_coverage_pct":   round(coverage, 1),
            "motion_speed_ms":      round(safe_float(np.random.uniform(8, 35)), 2),
            "motion_direction_deg": round(safe_float(np.random.uniform(0, 360)), 1),
            "alert":      "No severe weather detected",
            "alert_level":"nominal",
        },
        "bt_stats": {
            "pred_mean_K": safe_float(float(denorm_bt(pred).mean())),
            "t0_mean_K":   safe_float(float(denorm_bt(t0).mean())),
            "diff_max_K":  safe_float(float(denorm_bt(np.abs(pred - gt)).max())),
        },
        "benchmark": {
            "ThermalIFNet": {"PSNR": psnr_val,        "SSIM": ssim_val,        "time_ms": 48},
            "OpticalFlow":  {"PSNR": psnr_val - 3.4,  "SSIM": ssim_val - 0.051,"time_ms": 12},
            "Linear":       {"PSNR": psnr_val - 7.1,  "SSIM": ssim_val - 0.119,"time_ms":  1},
        },
        "source": "synthetic",
        "inference_time_s": round(0.04 + np.random.rand() * 0.02, 3),
    }


# ── real inference ────────────────────────────────────────────────────────────
def _real_result(triplet_index, stride,
                 t0_bytes=None, t2_bytes=None, gt_bytes=None) -> dict:
    t_start = time.perf_counter()
    model   = _get_model()

    if t0_bytes and t2_bytes:
        t0_t, t2_t, t1_np = _load_uploaded(t0_bytes, t2_bytes, gt_bytes)
    else:
        triplets = _get_triplets(stride)
        if not triplets or triplet_index >= len(triplets):
            raise ValueError("No triplets available")
        t0_p, t1_p, t2_p = triplets[triplet_index]
        t0_t, _, _ = preprocess_frame(str(t0_p))
        t2_t, _, _ = preprocess_frame(str(t2_p))
        t1_t, _, _ = preprocess_frame(str(t1_p))
        t1_np = t1_t.squeeze().numpy()

    pred_t, flow_t, conf_t = model.infer(t0_t, t2_t)
    pred = pred_t.squeeze().numpy()
    flow = flow_t.squeeze().numpy()
    conf = conf_t.squeeze().numpy()
    t0   = t0_t.squeeze().numpy()
    t2   = t2_t.squeeze().numpy()
    diff = np.abs(pred - t1_np)

    raw_m   = compute_all_metrics(pred, t1_np)
    elapsed = time.perf_counter() - t_start

    return {
        "images": {
            "t0":           arr_to_b64(t0,              "inferno_r", 0, 1),
            "t2":           arr_to_b64(t2,              "inferno_r", 0, 1),
            "t1_gt":        arr_to_b64(t1_np,           "inferno_r", 0, 1),
            "prediction":   arr_to_b64(pred,            "inferno_r", 0, 1),
            "diff_heatmap": arr_to_b64(diff,            "hot",       0, max(diff.max(), 1e-6)),
            "confidence":   arr_to_b64(conf,            "viridis",   0, 1),
            "optical_flow": arr_to_b64(flow_to_rgb(flow[:2]), None, None, None),
        },
        "metrics": {
            "PSNR":       safe_float(raw_m.get("PSNR", 0)),
            "SSIM":       safe_float(raw_m.get("SSIM", 0)),
            "MAE":        safe_float(float(np.mean(np.abs(pred - t1_np)))),
            "RMSE":       safe_float(float(np.sqrt(np.mean((pred - t1_np)**2)))),
            "BT_MAE":     safe_float(raw_m.get("BT_MAE", 0)),
            "FSIM":       safe_float(raw_m.get("FSIM", 0)),
            "confidence": safe_float(float(conf.mean())),
        },
        "weather": _compute_weather(diff, flow, conf),
        "bt_stats": {
            "pred_mean_K": safe_float(float(denorm_bt(pred).mean())),
            "t0_mean_K":   safe_float(float(denorm_bt(t0).mean())),
            "diff_max_K":  safe_float(float(np.abs(denorm_bt(pred)
                                                    - denorm_bt(t1_np)).max())),
        },
        "benchmark": {
            "ThermalIFNet": {"PSNR": safe_float(raw_m.get("PSNR", 0)),
                             "SSIM": safe_float(raw_m.get("SSIM", 0)), "time_ms": round(elapsed*1000)},
            "OpticalFlow":  {"PSNR": 0.0, "SSIM": 0.0, "time_ms": 0},
            "Linear":       {"PSNR": 0.0, "SSIM": 0.0, "time_ms": 0},
        },
        "source":           "model",
        "inference_time_s": round(elapsed, 3),
    }


def _load_uploaded(t0_b, t2_b, gt_b):
    import tempfile, os
    def _nc(b):
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            f.write(b); return f.name
    p0 = _nc(t0_b); p2 = _nc(t2_b)
    try:
        t0_t, _, _ = preprocess_frame(p0)
        t2_t, _, _ = preprocess_frame(p2)
        t1_np = ((t0_t + t2_t) / 2).squeeze().numpy()
        if gt_b:
            pg = _nc(gt_b)
            try:
                tg, _, _ = preprocess_frame(pg)
                t1_np = tg.squeeze().numpy()
            finally:
                os.unlink(pg)
    finally:
        os.unlink(p0); os.unlink(p2)
    return t0_t, t2_t, t1_np


def _compute_weather(diff, flow, conf) -> dict:
    coverage = float(np.mean(diff > 0.1) * 100)
    u = flow[0] if flow.ndim == 3 else flow
    v = flow[1] if flow.ndim == 3 else np.zeros_like(flow)
    speed = float(np.mean(np.sqrt(u**2 + v**2)))
    angle = float(np.degrees(np.arctan2(v.mean(), u.mean())) % 360)
    alert = ("Severe convection detected" if coverage > 60 else
             "Moderate cloud activity"    if coverage > 30 else
             "No severe weather detected")
    level = "danger" if coverage > 60 else "warning" if coverage > 30 else "nominal"
    return {
        "cloud_coverage_pct":   round(coverage, 1),
        "motion_speed_ms":      round(speed, 2),
        "motion_direction_deg": round(angle, 1),
        "alert": alert, "alert_level": level,
    }


# ── public API ────────────────────────────────────────────────────────────────
def run_interpolation(
    triplet_index: int = 0,
    stride: int = 10,
    t0_bytes: Optional[bytes] = None,
    t2_bytes: Optional[bytes] = None,
    gt_bytes:  Optional[bytes] = None,
) -> dict:
    if _HAS_SRC and (_get_model() is not None or _get_files()):
        try:
            return _real_result(triplet_index, stride, t0_bytes, t2_bytes, gt_bytes)
        except Exception as e:
            logger.warning("Real inference failed (%s) — falling back to synthetic", e)
    return _synthetic_result()


def get_system_status() -> dict:
    files = _get_files()
    status = {
        "gpu_available":      False,
        "gpu_name":           "CPU (Demo Mode)",
        "gpu_memory_gb":      0,
        "cpu_pct":            0.0,
        "ram_gb":             0.0,
        "ram_total_gb":       0.0,
        "nc_files":           len(files),
        "model_loaded":       _model is not None,
        "checkpoint_exists":  Path(settings.checkpoint_path).exists(),
        "dataset":            "GOES-19 ABI M6C13",
        "model":              "ThermalIFNet (RIFE)",
        "mode":               "model" if (_HAS_SRC and _model) else "synthetic",
    }
    try:
        import torch
        status["gpu_available"] = torch.cuda.is_available()
        status["gpu_name"]      = (torch.cuda.get_device_name(0)
                                   if torch.cuda.is_available() else "CPU (Demo Mode)")
        status["gpu_memory_gb"] = (round(torch.cuda.get_device_properties(0)
                                         .total_memory / 1e9, 1)
                                   if torch.cuda.is_available() else 0)
    except Exception:
        pass
    try:
        import psutil
        status["cpu_pct"]    = psutil.cpu_percent(interval=0.1)
        status["ram_gb"]     = round(psutil.virtual_memory().used / 1e9, 1)
        status["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    return status


def get_triplet_list(stride: int = 10, limit: int = 200) -> dict:
    files    = _get_files()
    triplets = _get_triplets(stride)[:limit]
    return {
        "total":    len(triplets),
        "files":    len(files),
        "stride":   stride,
        "triplets": [
            {"index": i, "t0": Path(t[0]).name,
             "t1": Path(t[1]).name, "t2": Path(t[2]).name}
            for i, t in enumerate(triplets)
        ],
    }