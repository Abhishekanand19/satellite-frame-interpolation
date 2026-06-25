# dashboard/backend/main.py
"""
FastAPI backend for ISRO PS12 Dashboard.
Serves interpolation results, metrics, and optical flow data.
"""

import sys
import json
import time
import base64
import io
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocessing.goes_preprocessor import (
    preprocess_frame, denormalize_bt, BT_MIN, BT_MAX
)
from src.data_loader.goes_dataset import discover_nc_files, build_triplets
from src.interpolation.rife_model import RIFEThermalInterpolator, DEVICE
from src.physics_metrics.metrics import compute_all_metrics

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ISRO PS12 — Satellite Frame Interpolation",
    description="AI-powered temporal super-resolution for GOES/INSAT imagery",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────
_model: Optional[RIFEThermalInterpolator] = None
_files: List[Path] = []
_triplets = []


def get_model() -> RIFEThermalInterpolator:
    global _model
    if _model is None:
        ckpt = ROOT / "models/checkpoints/best_model.pth"
        _model = RIFEThermalInterpolator(
            checkpoint_path=str(ckpt) if ckpt.exists() else None
        )
    return _model


def get_files() -> List[Path]:
    global _files
    if not _files:
        _files = discover_nc_files(str(ROOT / "data/goes19/raw"))
    return _files


def array_to_b64_png(arr: np.ndarray, cmap: str = 'inferno_r',
                      vmin: float = None, vmax: float = None) -> str:
    """Convert numpy array to base64 PNG string."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """Convert optical flow (2, H, W) to RGB visualization."""
    u = flow[0]
    v = flow[1]
    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)
    
    # HSV → RGB
    hue = (angle + np.pi) / (2 * np.pi)
    sat = np.ones_like(hue)
    val = magnitude / (magnitude.max() + 1e-8)
    
    hsv = np.stack([hue, sat, val], axis=-1)
    from matplotlib.colors import hsv_to_rgb
    rgb = hsv_to_rgb(hsv)
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────────────────────

class InterpolationRequest(BaseModel):
    t0_index: int
    t2_index: int
    timestep: float = 0.5


class TripletRequest(BaseModel):
    triplet_index: int
    stride: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "ISRO PS12 API running", "device": str(DEVICE)}


@app.get("/api/files")
def list_files():
    files = get_files()
    return {
        "total": len(files),
        "files": [
            {
                "index": i,
                "name": f.name,
                "day": f.parent.name,
                "path": str(f),
            }
            for i, f in enumerate(files[:200])  # limit response size
        ]
    }


@app.get("/api/triplets")
def list_triplets(stride: int = 10, limit: int = 50):
    files = get_files()
    triplets = build_triplets(files, stride=stride, max_triplets=limit)
    return {
        "total": len(triplets),
        "stride_minutes": stride,
        "triplets": [
            {
                "index": i,
                "t0": Path(t[0]).name,
                "t1_gt": Path(t[1]).name,
                "t2": Path(t[2]).name,
            }
            for i, t in enumerate(triplets)
        ]
    }


@app.post("/api/interpolate")
async def interpolate(req: TripletRequest):
    """Run RIFE interpolation on a triplet and return all data for dashboard."""
    start = time.time()
    
    files = get_files()
    triplets = build_triplets(files, stride=req.stride)
    
    if req.triplet_index >= len(triplets):
        raise HTTPException(status_code=400, detail=f"Triplet index {req.triplet_index} out of range (max {len(triplets)-1})")
    
    t0_path, t1_path, t2_path = triplets[req.triplet_index]
    
    # Preprocess
    t0_tensor, t0_raw, _ = preprocess_frame(str(t0_path))
    t1_tensor, t1_raw, _ = preprocess_frame(str(t1_path))
    t2_tensor, t2_raw, _ = preprocess_frame(str(t2_path))
    
    # Inference
    model = get_model()
    pred_tensor, flow_tensor, conf_tensor = model.infer(t0_tensor, t2_tensor)
    
    pred_np  = pred_tensor.squeeze().numpy()
    flow_np  = flow_tensor.squeeze().numpy()   # (4,H,W)
    conf_np  = conf_tensor.squeeze().numpy()
    t0_np    = t0_tensor.squeeze().numpy()
    t1_np    = t1_tensor.squeeze().numpy()
    t2_np    = t2_tensor.squeeze().numpy()
    
    # Metrics
    metrics = compute_all_metrics(pred_np, t1_np)
    
    # Difference heatmap
    diff = np.abs(pred_np - t1_np)
    
    # Optical flow (use first 2 channels)
    flow_rgb = flow_to_rgb(flow_np[:2])
    
    # Denorm for BT display
    pred_bt = denormalize_bt(pred_np)
    t1_bt   = denormalize_bt(t1_np)
    diff_bt = np.abs(pred_bt - t1_bt)
    
    elapsed = time.time() - start
    
    return {
        "triplet": {
            "index": req.triplet_index,
            "t0_file": t0_path.name,
            "t1_file": t1_path.name,
            "t2_file": t2_path.name,
        },
        "images": {
            "t0":         array_to_b64_png(t0_np,  'inferno_r', 0, 1),
            "t1_gt":      array_to_b64_png(t1_np,  'inferno_r', 0, 1),
            "t2":         array_to_b64_png(t2_np,  'inferno_r', 0, 1),
            "prediction": array_to_b64_png(pred_np,'inferno_r', 0, 1),
            "diff_heatmap": array_to_b64_png(diff, 'hot',       0, diff.max()),
            "confidence": array_to_b64_png(conf_np,'viridis',   0, 1),
            "optical_flow": array_to_b64_png(flow_rgb, None, None, None),
        },
        "bt_stats": {
            "t0_mean_K":   float(denormalize_bt(t0_np).mean()),
            "pred_mean_K": float(pred_bt.mean()),
            "gt_mean_K":   float(t1_bt.mean()),
            "diff_max_K":  float(diff_bt.max()),
        },
        "metrics": metrics,
        "inference_time_s": round(elapsed, 3),
    }


@app.get("/api/frame/{file_index}")
def get_frame(file_index: int, cmap: str = 'inferno_r'):
    """Return a single frame as base64 PNG."""
    files = get_files()
    if file_index >= len(files):
        raise HTTPException(status_code=404, detail="File index out of range")
    
    tensor, raw_bt, norm = preprocess_frame(str(files[file_index]))
    img = tensor.squeeze().numpy()
    
    return {
        "index": file_index,
        "file": files[file_index].name,
        "image_b64": array_to_b64_png(img, cmap),
        "stats": {
            "bt_min": float(np.nanmin(raw_bt)),
            "bt_max": float(np.nanmax(raw_bt)),
            "bt_mean": float(np.nanmean(raw_bt)),
        }
    }


@app.get("/api/health")
def health():
    files = get_files()
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "nc_files_found": len(files),
        "model_loaded": _model is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }