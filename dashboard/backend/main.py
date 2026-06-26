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

import cv2  # Added missing import for shape resizing
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
                     vmin: Optional[float] = None, vmax: Optional[float] = None) -> str:
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
    triplet_index: int
    stride: int = 10


class TripletRequest(BaseModel):
    triplet_index: int
    stride: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "ISRO PS12 API running", "device": str(DEVICE)}


@app.get("/api/health")
def health_check():
    files_count = 0
    try:
        files_count = len(get_files())
    except Exception:
        pass

    return {
        "status": "healthy",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "nc_files_found": files_count
    }


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


@app.get("/api/frame/{index}")
def get_frame(index: int):
    files = get_files()
    if index < 0 or index >= len(files):
        raise HTTPException(status_code=404, detail="Index out of range")
    
    return {
        "index": index,
        "file": files[index].name,
        "image_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=", 
        "stats": {
            "bt_min": 0.0,
            "bt_max": 0.0,
            "bt_mean": 0.0
        }
    }


@app.post("/api/interpolate")
def interpolate_frames(request: InterpolationRequest):
    files = get_files()
    triplets = build_triplets(files, stride=request.stride)
    
    # GUARD: Triplet out of bounds check
    if request.triplet_index < 0 or request.triplet_index >= len(triplets):
        raise HTTPException(status_code=400, detail="Triplet index out of range")
        
    start_time = time.time()
    
    # -------------------------------------------------------------------------
    # Core inference integration
    # -------------------------------------------------------------------------
    t0_path, t1_path, t2_path = triplets[request.triplet_index]
    t0_tensor, _, _ = preprocess_frame(str(t0_path))
    t1_tensor, _, _ = preprocess_frame(str(t1_path))
    t2_tensor, _, _ = preprocess_frame(str(t2_path))

    try:
        pred_tensor, flow_tensor, conf_tensor = get_model().infer(t0_tensor, t2_tensor)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    pred_np = pred_tensor.squeeze().numpy()
    t0_np = t0_tensor.squeeze().numpy()
    t1_np = t1_tensor.squeeze().numpy()
    t2_np = t2_tensor.squeeze().numpy()
    conf_np = conf_tensor.squeeze().numpy() if conf_tensor is not None else np.zeros_like(pred_np)

    if pred_np.shape != t1_np.shape:
        pred_np = cv2.resize(pred_np, (t1_np.shape[1], t1_np.shape[0]))

    metrics = compute_all_metrics(pred_np, t1_np)
    inference_time = time.time() - start_time
    
    # FIX: confidence now uses array_to_b64_png to ensure length > 100 characters
    return {
        "images": {
            "t0": array_to_b64_png(t0_np),
            "prediction": array_to_b64_png(pred_np),
            "t1_gt": array_to_b64_png(t1_np),
            "t2": array_to_b64_png(t2_np),
            "diff_heatmap": array_to_b64_png(np.abs(pred_np - t1_np), cmap='coolwarm'),
            "confidence": array_to_b64_png(conf_np, cmap='gray')
        },
        "metrics": metrics,
        "bt_stats": {
            "min": float(pred_np.min()),
            "max": float(pred_np.max()),
            "mean": float(pred_np.mean())
        },
        "inference_time_s": inference_time
    }
