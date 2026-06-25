# src/inference/predictor.py
"""
Dataset-agnostic inference engine for ThermalIFNet.
Swap preprocessor only when moving from GOES → INSAT.
"""

import logging
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.interpolation.rife_model import RIFEThermalInterpolator, DEVICE
from src.preprocessing.goes_preprocessor import (
    preprocess_frame, denormalize_bt, BT_MIN, BT_MAX, TARGET_SIZE
)

logger = logging.getLogger("predictor")


class ThermalPredictor:
    """
    Inference wrapper for ThermalIFNet.
    Handles single and batch inference with full pre/postprocessing.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device = DEVICE,
        target_size: tuple = TARGET_SIZE,
        bt_min: float = BT_MIN,
        bt_max: float = BT_MAX,
    ):
        self.device      = device
        self.target_size = target_size
        self.bt_min      = bt_min
        self.bt_max      = bt_max

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.interpolator = RIFEThermalInterpolator(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self.interpolator.model.eval()
        logger.info("Loaded checkpoint: %s  |  device: %s", checkpoint_path, device)

    # ── Preprocessing ──────────────────────────────────────────────────────────
    def _load_frame(self, path: str) -> torch.Tensor:
        """Load .nc file → (1,1,H,W) tensor."""
        tensor, _, _ = preprocess_frame(
            path,
            target_size=self.target_size,
            bt_min=self.bt_min,
            bt_max=self.bt_max,
            return_tensor=True,
        )
        return tensor  # (1,1,H,W)

    def _pad(self, t: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        H, W = t.shape[-2], t.shape[-1]
        ph = (32 - H % 32) % 32
        pw = (32 - W % 32) % 32
        return F.pad(t, [0, pw, 0, ph]), (H, W)

    def _unpad(self, t: torch.Tensor, orig: tuple) -> torch.Tensor:
        H, W = orig
        return t[..., :H, :W]

    # ── Postprocessing ─────────────────────────────────────────────────────────
    def _to_bt(self, tensor: torch.Tensor) -> np.ndarray:
        """(1,1,H,W) normalized tensor → BT numpy array in Kelvin."""
        arr = tensor.squeeze().cpu().numpy()
        return denormalize_bt(arr, self.bt_min, self.bt_max)

    # ── Single inference ───────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        t0_path: str,
        t2_path: str,
        timestep: float = 0.5,
    ) -> dict:
        """
        Interpolate between t0 and t2.

        Returns dict with:
            prediction_norm  : np.ndarray  [0,1]
            prediction_bt    : np.ndarray  Kelvin
            flow             : np.ndarray  (4,H,W)
            confidence       : np.ndarray  (H,W)
            t0_norm, t2_norm : np.ndarray  [0,1]
        """
        t0 = self._load_frame(t0_path)
        t2 = self._load_frame(t2_path)

        t0_pad, orig = self._pad(t0)
        t2_pad, _    = self._pad(t2)

        t0_pad = t0_pad.to(self.device)
        t2_pad = t2_pad.to(self.device)

        pred_pad, flow_pad, conf_pad = self.interpolator.model(t0_pad, t2_pad, timestep)

        pred = self._unpad(pred_pad, orig).cpu()
        flow = self._unpad(flow_pad, orig).cpu()
        conf = self._unpad(conf_pad, orig).cpu()

        pred_np = pred.squeeze().numpy()
        return {
            "prediction_norm": pred_np,
            "prediction_bt":   self._to_bt(pred),
            "flow":            flow.squeeze().numpy(),
            "confidence":      conf.squeeze().numpy(),
            "t0_norm":         t0.squeeze().numpy(),
            "t2_norm":         t2.squeeze().numpy(),
        }

    # ── Batch inference ────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_batch(
        self,
        pairs: list[tuple[str, str]],
        timestep: float = 0.5,
    ) -> list[dict]:
        """
        Run inference on a list of (t0_path, t2_path) pairs.
        Returns list of result dicts (same structure as predict()).
        """
        results = []
        for i, (t0_path, t2_path) in enumerate(pairs):
            logger.info("Batch [%d/%d]: %s → %s", i + 1, len(pairs),
                        Path(t0_path).name, Path(t2_path).name)
            results.append(self.predict(t0_path, t2_path, timestep))
        return results

    # ── Save outputs ───────────────────────────────────────────────────────────
    def save_result(
        self,
        result: dict,
        out_dir: str,
        stem: str = "interpolated",
        save_npy: bool = True,
        save_png: bool = True,
        save_comparison: bool = True,
    ) -> dict[str, str]:
        """
        Save prediction to disk.
        Returns dict of saved file paths.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = {}

        if save_npy:
            p = str(out / f"{stem}_bt.npy")
            np.save(p, result["prediction_bt"])
            saved["npy_bt"] = p

            p2 = str(out / f"{stem}_norm.npy")
            np.save(p2, result["prediction_norm"])
            saved["npy_norm"] = p2
            logger.info("Saved numpy arrays → %s", out)

        if save_png:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
            ax.imshow(result["prediction_bt"], cmap="inferno_r",
                      vmin=self.bt_min, vmax=self.bt_max)
            ax.set_title("AI Interpolated Frame (BT)", fontsize=12)
            ax.axis("off")
            plt.colorbar(ax.images[0], ax=ax, label="Brightness Temp (K)", shrink=0.8)
            p = str(out / f"{stem}.png")
            plt.savefig(p, bbox_inches="tight", dpi=150)
            plt.close(fig)
            saved["png"] = p
            logger.info("Saved PNG → %s", p)

        if save_comparison:
            fig, axes = plt.subplots(1, 5, figsize=(28, 6), dpi=100)
            panels = [
                (result["t0_norm"],         "inferno_r", "T0 — Input",           0, 1),
                (result["prediction_norm"], "inferno_r", "AI — Interpolated",     0, 1),
                (result["t2_norm"],         "inferno_r", "T2 — Input",            0, 1),
                (result["confidence"],      "viridis",   "Confidence",            0, 1),
                (
                    np.abs(result["prediction_norm"] - result["t0_norm"]),
                    "hot", "Δ |Pred − T0|", None, None
                ),
            ]
            for ax, (arr, cmap, title, vmin, vmax) in zip(axes, panels):
                im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(title, fontsize=10)
                ax.axis("off")
                plt.colorbar(im, ax=ax, shrink=0.75)
            plt.suptitle(f"ThermalIFNet Interpolation — {stem}", fontsize=12, y=1.01)
            plt.tight_layout()
            p = str(out / f"{stem}_comparison.png")
            plt.savefig(p, bbox_inches="tight", dpi=150)
            plt.close(fig)
            saved["comparison"] = p
            logger.info("Saved comparison → %s", p)

        return saved