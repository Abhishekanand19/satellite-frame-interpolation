# src/visualization/visualize.py
"""
Visualization module for ThermalIFNet.
Generates heatmaps, comparisons, GIFs, overlays — publication quality.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import imageio
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.predictor import ThermalPredictor
from src.preprocessing.goes_preprocessor import (
    preprocess_frame, denormalize_bt, BT_MIN, BT_MAX, TARGET_SIZE
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("visualize")

# ── Style ──────────────────────────────────────────────────────────────────────
BG      = "#080c14"
PANEL   = "#0d1117"
ACCENT  = "#00e5ff"
WHITE   = "#e0e0e0"
DPI     = 150
CMAP_BT = "inferno_r"
CMAP_DIFF = "RdBu_r"
CMAP_CONF = "viridis"

def _style_ax(ax, title: str = "", fontsize: int = 10) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=WHITE, fontsize=fontsize, pad=6, fontweight="bold")
    ax.axis("off")

def _cbar(fig, im, ax, label: str = "") -> None:
    cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
    cb.ax.tick_params(colors="gray", labelsize=7)
    cb.set_label(label, color="gray", fontsize=7)
    cb.outline.set_edgecolor("#333")


# ── Core Visualizer ───────────────────────────────────────────────────────────
class ThermalVisualizer:

    def __init__(self, out_dir: str = "outputs/visualization"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Difference heatmap ─────────────────────────────────────────────────
    def difference_heatmap(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        stem: str = "diff",
        title: str = "Difference Heatmap",
    ) -> str:
        diff   = pred - gt
        absdiff = np.abs(diff)
        vmax   = max(np.percentile(absdiff, 99), 1e-6)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
        fig.suptitle(title, color=WHITE, fontsize=13, y=1.01)

        panels = [
            (pred,    CMAP_BT,   "Prediction",          0, 1),
            (gt,      CMAP_BT,   "Ground Truth",         0, 1),
            (diff,    CMAP_DIFF, "Δ (Pred − GT)",       -vmax, vmax),
        ]
        for ax, (arr, cmap, label, vmin, vmx) in zip(axes, panels):
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmx, interpolation="nearest")
            _style_ax(ax, label)
            _cbar(fig, im, ax, "BT norm" if cmap == CMAP_BT else "Δ norm")

        # Stats box
        stats = (f"MAE={absdiff.mean():.5f}  "
                 f"MAX={absdiff.max():.5f}  "
                 f"RMSE={np.sqrt(np.mean(diff**2)):.5f}")
        fig.text(0.5, -0.02, stats, ha="center", color="gray", fontsize=9,
                 fontfamily="monospace")

        plt.tight_layout()
        p = str(self.out_dir / f"{stem}_heatmap.png")
        fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info("Saved → %s", p)
        return p

    # ── 2. BT heatmap in Kelvin ───────────────────────────────────────────────
    def bt_heatmap(
        self,
        pred_bt: np.ndarray,
        gt_bt: np.ndarray,
        stem: str = "bt_diff",
    ) -> str:
        diff    = pred_bt - gt_bt
        absdiff = np.abs(diff)
        vmax    = max(np.percentile(absdiff, 99), 0.1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
        fig.suptitle("Brightness Temperature Comparison (K)", color=WHITE, fontsize=13, y=1.01)

        for ax, (arr, cmap, label, vmin, vmx) in zip(axes, [
            (pred_bt, CMAP_BT,   "Prediction (K)",   BT_MIN, BT_MAX),
            (gt_bt,   CMAP_BT,   "Ground Truth (K)", BT_MIN, BT_MAX),
            (diff,    CMAP_DIFF, "Δ BT (K)",        -vmax,   vmax),
        ]):
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmx, interpolation="nearest")
            _style_ax(ax, label)
            _cbar(fig, im, ax, "K")

        fig.text(0.5, -0.02,
                 f"MAE={absdiff.mean():.3f}K  MAX={absdiff.max():.3f}K  "
                 f"RMSE={np.sqrt(np.mean(diff**2)):.3f}K",
                 ha="center", color="gray", fontsize=9, fontfamily="monospace")

        plt.tight_layout()
        p = str(self.out_dir / f"{stem}_bt_heatmap.png")
        fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info("Saved → %s", p)
        return p

    # ── 3. Side-by-side comparison (5-panel) ─────────────────────────────────
    def side_by_side(
        self,
        t0: np.ndarray,
        pred: np.ndarray,
        t2: np.ndarray,
        gt: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None,
        stem: str = "compare",
        t0_label: str = "T0 (Input)",
        t2_label: str = "T2 (Input)",
    ) -> str:
        has_gt   = gt is not None
        has_conf = confidence is not None
        panels = [t0, pred, t2]
        labels = [t0_label, "AI Interpolated", t2_label]
        cmaps  = [CMAP_BT, CMAP_BT, CMAP_BT]
        vmins  = [0, 0, 0]
        vmaxs  = [1, 1, 1]

        if has_gt:
            diff = np.abs(pred - gt)
            panels += [gt, diff]
            labels += ["Ground Truth", "|Pred − GT|"]
            cmaps  += [CMAP_BT, "hot"]
            vmins  += [0, 0]
            vmaxs  += [1, diff.max() + 1e-6]

        if has_conf:
            panels.append(confidence)
            labels.append("Confidence")
            cmaps.append(CMAP_CONF)
            vmins.append(0)
            vmaxs.append(1)

        n    = len(panels)
        fig  = plt.figure(figsize=(6 * n, 6), facecolor=BG)
        gs   = gridspec.GridSpec(1, n, figure=fig, wspace=0.04)
        axes = [fig.add_subplot(gs[0, i]) for i in range(n)]

        for ax, arr, label, cmap, vmin, vmax in zip(axes, panels, labels, cmaps, vmins, vmaxs):
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            _style_ax(ax, label, fontsize=11)
            _cbar(fig, im, ax)

        fig.suptitle("ThermalIFNet — Frame Interpolation", color=WHITE, fontsize=14, y=1.02)
        plt.tight_layout()
        p = str(self.out_dir / f"{stem}_compare.png")
        fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info("Saved → %s", p)
        return p

    # ── 4. Optical flow RGB ───────────────────────────────────────────────────
    def flow_visualization(
        self,
        flow: np.ndarray,
        stem: str = "flow",
    ) -> str:
        """flow: (4,H,W) or (2,H,W)"""
        from matplotlib.colors import hsv_to_rgb
        u = flow[0] if flow.ndim == 3 else flow
        v = flow[1] if flow.ndim == 3 else np.zeros_like(flow)

        mag   = np.sqrt(u**2 + v**2)
        angle = np.arctan2(v, u)
        hue   = (angle + np.pi) / (2 * np.pi)
        sat   = np.ones_like(hue)
        val   = mag / (mag.max() + 1e-8)
        flow_rgb = hsv_to_rgb(np.stack([hue, sat, val], axis=-1))

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
        fig.suptitle("Optical Flow Visualization", color=WHITE, fontsize=13, y=1.01)

        axes[0].imshow(flow_rgb, interpolation="nearest")
        _style_ax(axes[0], "Flow (RGB: hue=dir, bright=mag)")

        im1 = axes[1].imshow(mag, cmap="plasma", interpolation="nearest")
        _style_ax(axes[1], "Flow Magnitude")
        _cbar(fig, im1, axes[1], "px/frame")

        im2 = axes[2].imshow(angle, cmap="hsv", vmin=-np.pi, vmax=np.pi,
                              interpolation="nearest")
        _style_ax(axes[2], "Flow Angle")
        _cbar(fig, im2, axes[2], "rad")

        plt.tight_layout()
        p = str(self.out_dir / f"{stem}_flow.png")
        fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info("Saved → %s", p)
        return p

    # ── 5. Animated GIF ───────────────────────────────────────────────────────
    def animated_gif(
        self,
        frames: list[np.ndarray],
        labels: list[str],
        stem: str = "animation",
        fps: int = 2,
        cmap: str = CMAP_BT,
        vmin: float = 0.0,
        vmax: float = 1.0,
        loop: int = 0,
    ) -> str:
        images = []
        norm   = Normalize(vmin=vmin, vmax=vmax)
        sm     = ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))

        for frame, label in zip(frames, labels):
            fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
            im = ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            _style_ax(ax, label, fontsize=12)
            _cbar(fig, im, ax, "BT norm")
            fig.suptitle("ThermalIFNet — Temporal Interpolation",
                         color=WHITE, fontsize=11, y=1.0)
            plt.tight_layout()

            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            images.append(buf.reshape(h, w, 3))
            plt.close(fig)

        p = str(self.out_dir / f"{stem}.gif")
        imageio.mimsave(p, images, fps=fps, loop=loop)
        logger.info("Saved GIF → %s  (%d frames @ %dfps)", p, len(images), fps)
        return p

    # ── 6. Prediction overlay ─────────────────────────────────────────────────
    def prediction_overlay(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        stem: str = "overlay",
        alpha: float = 0.5,
    ) -> str:
        """Blend prediction and GT with alpha to show alignment quality."""
        from matplotlib.colors import hsv_to_rgb

        blend = pred * alpha + gt * (1 - alpha)
        diff  = pred - gt

        fig = plt.figure(figsize=(20, 8), facecolor=BG)
        gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.06)

        specs = [
            (fig.add_subplot(gs[0, 0]), pred,   CMAP_BT,   "Prediction",      0,   1),
            (fig.add_subplot(gs[0, 1]), gt,     CMAP_BT,   "Ground Truth",    0,   1),
            (fig.add_subplot(gs[0, 2]), blend,  CMAP_BT,   f"Blend α={alpha}",0,   1),
            (fig.add_subplot(gs[0, 3]), diff,   CMAP_DIFF, "Δ (Pred − GT)", -0.2, 0.2),
            (fig.add_subplot(gs[1, 0]), np.abs(diff),     "hot",    "|Δ|",          0, np.abs(diff).max()),
            (fig.add_subplot(gs[1, 1]), pred**2 - gt**2,  CMAP_DIFF,"Δ Energy",    None, None),
            (fig.add_subplot(gs[1, 2]), np.clip(pred - gt + 0.5, 0, 1), CMAP_BT, "Shifted Δ", 0, 1),
            (fig.add_subplot(gs[1, 3]), (pred + gt) / 2,  CMAP_BT, "Mean Frame",   0, 1),
        ]

        for ax, arr, cmap, label, vmin, vmax in specs:
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            _style_ax(ax, label, fontsize=9)
            _cbar(fig, im, ax)

        fig.suptitle("Prediction Overlay Analysis", color=WHITE, fontsize=14, y=1.01)
        p = str(self.out_dir / f"{stem}_overlay.png")
        fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info("Saved → %s", p)
        return p

    # ── 7. Batch from inference results ──────────────────────────────────────
    def batch_visualize(
        self,
        results: list[dict],
        gt_arrays: Optional[list[np.ndarray]] = None,
        fps: int = 3,
    ) -> dict:
        saved = {"heatmaps": [], "comparisons": [], "overlays": [], "gifs": []}

        for i, res in enumerate(tqdm(results, desc="Visualizing", dynamic_ncols=True)):
            stem   = f"sample_{i:04d}"
            pred   = res["prediction_norm"]
            t0     = res["t0_norm"]
            t2     = res["t2_norm"]
            conf   = res.get("confidence")
            flow   = res.get("flow")
            gt     = gt_arrays[i] if gt_arrays else None

            saved["heatmaps"].append(
                self.difference_heatmap(pred, gt if gt is not None else t0, stem=stem)
            )
            saved["comparisons"].append(
                self.side_by_side(t0, pred, t2, gt=gt, confidence=conf, stem=stem)
            )
            if gt is not None:
                saved["overlays"].append(
                    self.prediction_overlay(pred, gt, stem=stem)
                )
            if flow is not None:
                self.flow_visualization(flow, stem=stem)

        # Master GIF across all predictions
        if results:
            frames = [r["prediction_norm"] for r in results]
            labels = [f"Sample {i:04d}" for i in range(len(results))]
            saved["gifs"].append(
                self.animated_gif(frames, labels, stem="batch_predictions", fps=fps)
            )

            # T0 → Pred → T2 triplet GIF for first sample
            r0 = results[0]
            saved["gifs"].append(
                self.animated_gif(
                    [r0["t0_norm"], r0["prediction_norm"], r0["t2_norm"]],
                    ["T0", "AI Interpolated", "T2"],
                    stem="sample_0000_triplet",
                    fps=2,
                )
            )

        out_json = self.out_dir / "visualization_manifest.json"
        with open(out_json, "w") as f:
            json.dump(saved, f, indent=2)
        logger.info("Manifest → %s", out_json)
        return saved


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ThermalIFNet Visualization CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--t0",         required=True,  help="T0 .nc path")
    p.add_argument("--t2",         required=True,  help="T2 .nc path")
    p.add_argument("--gt",         default=None,   help="Ground truth .nc path (optional)")
    p.add_argument("--checkpoint", required=True,  help="Model checkpoint .pth")
    p.add_argument("--out",        default="outputs/visualization")
    p.add_argument("--stem",       default="sample")
    p.add_argument("--size",       type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
    p.add_argument("--fps",        type=int, default=2, help="GIF frames per second")
    p.add_argument("--no_gif",     action="store_true")
    p.add_argument("--no_flow",    action="store_true")
    return p


def main() -> None:
    args      = build_parser().parse_args()
    predictor = ThermalPredictor(
        checkpoint_path=args.checkpoint,
        target_size=tuple(args.size),
    )
    vis = ThermalVisualizer(out_dir=args.out)

    result = predictor.predict(args.t0, args.t2, timestep=0.5)
    pred   = result["prediction_norm"]
    t0     = result["t0_norm"]
    t2     = result["t2_norm"]
    conf   = result["confidence"]
    flow   = result["flow"]

    gt = None
    if args.gt:
        gt_tensor, _, _ = preprocess_frame(args.gt, target_size=tuple(args.size))
        gt = gt_tensor if isinstance(gt_tensor, np.ndarray) else gt_tensor.squeeze().numpy()

    vis.difference_heatmap(pred, gt if gt is not None else t0, stem=args.stem)
    vis.side_by_side(t0, pred, t2, gt=gt, confidence=conf, stem=args.stem)
    vis.prediction_overlay(pred, gt if gt is not None else t0, stem=args.stem)

    if not args.no_flow:
        vis.flow_visualization(flow, stem=args.stem)

    if not args.no_gif:
        frames = [t0, pred, t2]
        labels = ["T0 (Input)", "AI Interpolated", "T2 (Input)"]
        if gt is not None:
            frames.insert(2, gt)
            labels.insert(2, "Ground Truth")
        vis.animated_gif(frames, labels, stem=args.stem, fps=args.fps)

    logger.info("All visualizations saved → %s", args.out)


if __name__ == "__main__":
    main()