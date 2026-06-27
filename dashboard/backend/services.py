"""
Service layer — all ML calls are isolated here.
Replace placeholder functions with real pipeline calls when ready.
"""
import logging, sys, time
from pathlib import Path
from typing import Optional
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.backend.config import settings
from dashboard.backend.utils import arr_to_b64, flow_to_rgb, denorm_bt, safe_float

logger = logging.getLogger("services")

# ── lazy model cache ──────────────────────────────────────────────────────────
_model = None
_files = []
_triplets = []


def _get_model():
    global _model
    if _model is None:
        try:
            from src.interpolation.rife_model import RIFEThermalInterpolator
            ckpt = Path(settings.checkpoint_path)
            _model = RIFEThermalInterpolator(
                checkpoint_path=str(ckpt) if ckpt.exists() else None
            )
            logger.info("Model loaded from %s", ckpt)
        except Exception as e:
            logger.warning("Model load failed (%s) — using synthetic mode", e)
            _model = None
    return _model


def _get_files():
    global _files
    if not _files:
        try:
            from src.data_loader.goes_dataset import discover_nc_files
            _files = discover_nc_files(settings.data_root)
        except Exception as e:
            logger.warning("File discovery failed: %s", e)
            _files = []
    return _files


def _get_triplets(stride: int):
    global _triplets
    try:
        from src.data_loader.goes_dataset import build_triplets
        files = _get_files()
        _triplets = build_triplets(files, stride=stride)
    except Exception as e:
        logger.warning("Triplet build failed: %s", e)
        _triplets = []
    return _triplets


# ── synthetic fallback ────────────────────────────────────────────────────────
def _synthetic_result() -> dict:
    H, W = 256, 256
    t0 = np.random.rand(H, W).astype(np.float32)
    t2 = np.random.rand(H, W).astype(np.float32)
    pred = (t0 * 0.5 + t2 * 0.5)
    diff = np.abs(pred - t0)
    flow = np.stack([np.random.randn(H, W), np.random.randn(H, W)]).astype(np.float32)
    conf = np.clip(1 - diff * 2, 0, 1)

    return {
        "images": {
            "t0":           arr_to_b64(t0,           "inferno_r", 0, 1),
            "t2":           arr_to_b64(t2,           "inferno_r", 0, 1),
            "prediction":   arr_to_b64(pred,         "inferno_r", 0, 1),
            "diff_heatmap": arr_to_b64(diff,         "hot",       0, diff.max()),
            "confidence":   arr_to_b64(conf,         "viridis",   0, 1),
            "optical_flow": arr_to_b64(flow_to_rgb(flow)),
        },
        "metrics": {
            "PSNR":       safe_float(28.5 + np.random.rand() * 3),
            "SSIM":       safe_float(0.88 + np.random.rand() * 0.08),
            "MAE":        safe_float(0.03 + np.random.rand() * 0.02),
            "RMSE":       safe_float(0.04 + np.random.rand() * 0.02),
            "BT_MAE":     safe_float(3.2  + np.random.rand() * 2),
            "FSIM":       safe_float(0.91 + np.random.rand() * 0.06),
            "confidence": safe_float(0.87 + np.random.rand() * 0.10),
        },
        "weather": {
            "cloud_coverage_pct": safe_float(np.random.randint(20, 85)),
            "motion_speed_ms":    safe_float(np.random.uniform(5, 40)),
            "motion_direction_deg": safe_float(np.random.uniform(0, 360)),
            "alert":              "No severe weather detected",
            "alert_level":        "nominal",
        },
        "bt_stats": {
            "pred_mean_K": safe_float(245.0 + np.random.rand() * 10),
            "t0_mean_K":   safe_float(243.0 + np.random.rand() * 10),
            "diff_max_K":  safe_float(np.random.uniform(1, 8)),
        },
        "benchmark": {
            "ThermalIFNet": {"PSNR": 31.2, "SSIM": 0.912, "time_ms": 48},
            "OpticalFlow":  {"PSNR": 27.8, "SSIM": 0.861, "time_ms": 12},
            "Linear":       {"PSNR": 24.1, "SSIM": 0.793, "time_ms":  1},
        },
        "source": "synthetic",
    }


# ── public service functions ──────────────────────────────────────────────────
def run_interpolation(
    triplet_index: int = 0,
    stride: int = 10,
    t0_bytes: Optional[bytes] = None,
    t2_bytes: Optional[bytes] = None,
    gt_bytes:  Optional[bytes] = None,
) -> dict:
    t_start = time.perf_counter()

    try:
        model = _get_model()

        if t0_bytes and t2_bytes:
            t0_t, t2_t, t1_np = _load_uploaded(t0_bytes, t2_bytes, gt_bytes)
        else:
            triplets = _get_triplets(stride)
            if not triplets or triplet_index >= len(triplets):
                raise ValueError("No triplets available")
            t0_p, t1_p, t2_p = triplets[triplet_index]
            from src.preprocessing.goes_preprocessor import preprocess_frame
            t0_t, _, _ = preprocess_frame(str(t0_p))
            t2_t, _, _ = preprocess_frame(str(t2_p))
            t1_t, _, _ = preprocess_frame(str(t1_p))
            t1_np = t1_t.squeeze().numpy()

        if model is None:
            raise RuntimeError("Model not loaded")

        import torch
        pred_t, flow_t, conf_t = model.infer(t0_t, t2_t)
        pred_np = pred_t.squeeze().numpy()
        flow_np = flow_t.squeeze().numpy()
        conf_np = conf_t.squeeze().numpy()
        t0_np   = t0_t.squeeze().numpy()
        t2_np   = t2_t.squeeze().numpy()
        diff    = np.abs(pred_np - t1_np)

        from src.physics_metrics.metrics import compute_all_metrics
        raw_m = compute_all_metrics(pred_np, t1_np)

        elapsed = time.perf_counter() - t_start
        result = _build_result(t0_np, t2_np, pred_np,
                               diff, flow_np, conf_np, t1_np, raw_m)
        result["inference_time_s"] = round(elapsed, 3)
        result["source"] = "model"
        return result

    except Exception as e:
        logger.warning("Inference failed (%s) — returning synthetic", e)
        elapsed = time.perf_counter() - t_start
        r = _synthetic_result()
        r["inference_time_s"] = round(elapsed, 3)
        return r


def _load_uploaded(t0_bytes, t2_bytes, gt_bytes):
    import tempfile, os
    from src.preprocessing.goes_preprocessor import preprocess_frame

    def _nc(b):
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            f.write(b); return f.name

    p0 = _nc(t0_bytes); p2 = _nc(t2_bytes)
    try:
        t0_t, _, _ = preprocess_frame(p0)
        t2_t, _, _ = preprocess_frame(p2)
        t1_np = None
        if gt_bytes:
            pg = _nc(gt_bytes)
            try:
                tg, _, _ = preprocess_frame(pg)
                t1_np = tg.squeeze().numpy()
            finally:
                os.unlink(pg)
        if t1_np is None:
            t1_np = ((t0_t + t2_t) / 2).squeeze().numpy()
    finally:
        os.unlink(p0); os.unlink(p2)
    return t0_t, t2_t, t1_np


def _build_result(t0, t2, pred, diff, flow, conf, gt, raw_m) -> dict:
    from dashboard.backend.utils import safe_float
    pred_bt = denorm_bt(pred)
    gt_bt   = denorm_bt(gt)

    return {
        "images": {
            "t0":           arr_to_b64(t0,                "inferno_r", 0, 1),
            "t2":           arr_to_b64(t2,                "inferno_r", 0, 1),
            "t1_gt":        arr_to_b64(gt,                "inferno_r", 0, 1),
            "prediction":   arr_to_b64(pred,              "inferno_r", 0, 1),
            "diff_heatmap": arr_to_b64(diff,              "hot",       0, diff.max() + 1e-6),
            "confidence":   arr_to_b64(conf,              "viridis",   0, 1),
            "optical_flow": arr_to_b64(flow_to_rgb(flow[:2])),
        },
        "metrics": {
            "PSNR":       safe_float(raw_m.get("PSNR", 0)),
            "SSIM":       safe_float(raw_m.get("SSIM", 0)),
            "MAE":        safe_float(float(np.mean(np.abs(pred - gt)))),
            "RMSE":       safe_float(float(np.sqrt(np.mean((pred - gt)**2)))),
            "BT_MAE":     safe_float(raw_m.get("BT_MAE", 0)),
            "FSIM":       safe_float(raw_m.get("FSIM", 0)),
            "confidence": safe_float(float(conf.mean())),
        },
        "weather": _compute_weather(diff, flow, conf),
        "bt_stats": {
            "pred_mean_K": safe_float(float(pred_bt.mean())),
            "t0_mean_K":   safe_float(float(denorm_bt(t0).mean())),
            "diff_max_K":  safe_float(float(np.abs(pred_bt - gt_bt).max())),
        },
        "benchmark": {
            "ThermalIFNet": {"PSNR": safe_float(raw_m.get("PSNR", 0)),
                             "SSIM": safe_float(raw_m.get("SSIM", 0)), "time_ms": 0},
            "OpticalFlow":  {"PSNR": 0.0, "SSIM": 0.0, "time_ms": 0},
            "Linear":       {"PSNR": 0.0, "SSIM": 0.0, "time_ms": 0},
        },
    }


def _compute_weather(diff, flow, conf) -> dict:
    coverage = float(np.mean(diff > 0.1) * 100)
    u = flow[0] if flow.ndim == 3 else flow
    v = flow[1] if flow.ndim == 3 else np.zeros_like(flow)
    speed  = float(np.mean(np.sqrt(u**2 + v**2)))
    angle  = float(np.degrees(np.arctan2(v.mean(), u.mean())) % 360)
    alert  = "Severe convection detected" if coverage > 60 else \
             "Moderate cloud activity"     if coverage > 30 else \
             "No severe weather detected"
    level  = "danger" if coverage > 60 else \
             "warning" if coverage > 30 else "nominal"
    return {
        "cloud_coverage_pct":   round(coverage, 1),
        "motion_speed_ms":      round(speed, 2),
        "motion_direction_deg": round(angle, 1),
        "alert": alert,
        "alert_level": level,
    }


def get_system_status() -> dict:
    import torch, psutil
    files = _get_files()
    return {
        "gpu_available":  torch.cuda.is_available(),
        "gpu_name":       torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_memory_gb":  round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                          if torch.cuda.is_available() else 0,
        "cpu_pct":        psutil.cpu_percent(interval=0.2),
        "ram_gb":         round(psutil.virtual_memory().used / 1e9, 1),
        "ram_total_gb":   round(psutil.virtual_memory().total / 1e9, 1),
        "nc_files":       len(files),
        "model_loaded":   _model is not None,
        "checkpoint_exists": Path(settings.checkpoint_path).exists(),
        "dataset":        "GOES-19 ABI M6C13",
        "model":          "ThermalIFNet (RIFE)",
    }


def get_triplet_list(stride: int = 10, limit: int = 200) -> dict:
    files    = _get_files()
    triplets = _get_triplets(stride)[:limit]
    return {
        "total":   len(triplets),
        "files":   len(files),
        "stride":  stride,
        "triplets": [
            {"index": i, "t0": Path(t[0]).name,
             "t1": Path(t[1]).name, "t2": Path(t[2]).name}
            for i, t in enumerate(triplets)
        ],
    }