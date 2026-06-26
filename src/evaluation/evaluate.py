# src/evaluation/evaluate.py
"""
Evaluation module for ThermalIFNet.
Evaluates full dataset, computes PSNR/SSIM/RMSE/MAE/BT-MAE/FSIM,
saves evaluation.json and summary plots.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.physics_metrics.metrics import compute_all_metrics
from src.data_loader.goes_dataset import GOESTripletDataset, build_triplets, discover_nc_files
from src.interpolation.rife_model import RIFEThermalInterpolator, DEVICE
from src.preprocessing.goes_preprocessor import TARGET_SIZE, BT_MIN, BT_MAX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate")


# ── Extra metrics not in metrics.py ───────────────────────────────────────────
def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def compute_full_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Combines existing metrics.py + RMSE + MAE."""
    base = compute_all_metrics(pred, gt)
    base["RMSE"] = rmse(pred, gt)
    base["MAE"]  = mae(pred, gt)
    return base


# ── Evaluator ─────────────────────────────────────────────────────────────────
class Evaluator:
    def __init__(
        self,
        checkpoint_path: str,
        data_root: str = "data/goes19/raw",
        out_dir: str = "outputs/evaluation",
        stride: int = 10,
        target_size: tuple = TARGET_SIZE,
        bt_min: float = BT_MIN,
        bt_max: float = BT_MAX,
        batch_size: int = 1,
        num_workers: int = 4,
        split: str = "test",          # "train" | "val" | "test" | "all"
        max_samples: Optional[int] = None,
        device: torch.device = DEVICE,
    ):
        self.checkpoint_path = checkpoint_path
        self.data_root       = data_root
        self.out_dir         = Path(out_dir)
        self.stride          = stride
        self.target_size     = target_size
        self.bt_min          = bt_min
        self.bt_max          = bt_max
        self.batch_size      = batch_size
        self.num_workers     = num_workers
        self.split           = split
        self.max_samples     = max_samples
        self.device          = device

        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.model = RIFEThermalInterpolator(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self.model.model.eval()
        logger.info("Checkpoint loaded: %s", checkpoint_path)
        logger.info("Device: %s | Split: %s", device, split)

    # ── Dataset ───────────────────────────────────────────────────────────────
    def _build_loader(self) -> DataLoader:
        files    = discover_nc_files(self.data_root)
        triplets = build_triplets(files, stride=self.stride,
                                  max_triplets=self.max_samples)

        n       = len(triplets)
        n_train = int(n * 0.80)
        n_val   = int(n * 0.10)

        split_map = {
            "train": triplets[:n_train],
            "val":   triplets[n_train:n_train + n_val],
            "test":  triplets[n_train + n_val:],
            "all":   triplets,
        }
        chosen = split_map.get(self.split)
        if chosen is None:
            raise ValueError(f"Unknown split '{self.split}'. Choose: train/val/test/all")
        if not chosen:
            raise RuntimeError(f"Split '{self.split}' is empty.")

        ds = GOESTripletDataset(chosen, target_size=self.target_size)
        logger.info("Evaluating %d samples from '%s' split", len(ds), self.split)

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    # ── Core eval loop ────────────────────────────────────────────────────────
    @torch.no_grad()
    def run(self) -> dict:
        loader  = self._build_loader()
        keys    = ["PSNR", "SSIM", "RMSE", "MAE", "MSE", "BT_MAE", "FSIM"]
        records = []                    # per-sample dicts
        accum   = {k: [] for k in keys}

        t_start = time.time()

        for batch in tqdm(loader, desc="Evaluating", dynamic_ncols=True):
            t0    = batch["t0"].to(self.device, non_blocking=True)
            t2    = batch["t2"].to(self.device, non_blocking=True)
            t1_gt = batch["t1_gt"].to(self.device, non_blocking=True)

            pred, _, _ = self.model.model(t0, t2)
            pred   = torch.clamp(pred, 0, 1)

            pred_np = pred.cpu().float().numpy()
            gt_np   = t1_gt.cpu().float().numpy()

            for i in range(pred_np.shape[0]):
                p = pred_np[i, 0]
                g = gt_np[i, 0]
                m = compute_full_metrics(p, g)

                for k in keys:
                    accum[k].append(m.get(k, float("nan")))

                records.append({
                    "t0_path":  batch["t0_path"][i],
                    "t1_path":  batch["t1_path"][i],
                    "t2_path":  batch["t2_path"][i],
                    "metrics":  {k: round(m.get(k, float("nan")), 6) for k in keys},
                })

        elapsed = time.time() - t_start

        # Aggregate
        summary = {}
        for k in keys:
            vals = [v for v in accum[k] if not np.isnan(v)]
            summary[k] = {
                "mean":   round(float(np.mean(vals)),   6),
                "std":    round(float(np.std(vals)),    6),
                "min":    round(float(np.min(vals)),    6),
                "max":    round(float(np.max(vals)),    6),
                "median": round(float(np.median(vals)), 6),
            }

        output = {
            "checkpoint":     self.checkpoint_path,
            "data_root":      self.data_root,
            "split":          self.split,
            "stride":         self.stride,
            "target_size":    list(self.target_size),
            "num_samples":    len(records),
            "inference_time_s": round(elapsed, 3),
            "summary":        summary,
            "per_sample":     records,
        }

        self._save_json(output)
        self._save_plots(accum, keys)
        self._print_summary(summary, elapsed, len(records))

        return output

    # ── Outputs ───────────────────────────────────────────────────────────────
    def _save_json(self, data: dict) -> None:
        p = self.out_dir / "evaluation.json"
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved → %s", p)

    def _save_plots(self, accum: dict, keys: list) -> None:
        n_keys = len(keys)
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        axes = axes.flatten()

        colors = ["#00e5ff", "#00ff88", "#ffaa00", "#ff6b6b", "#b388ff", "#ff8c42", "#a8dadc"]

        for idx, k in enumerate(keys):
            ax  = axes[idx]
            vals = [v for v in accum[k] if not np.isnan(v)]
            ax.hist(vals, bins=30, color=colors[idx % len(colors)], alpha=0.85, edgecolor="white", lw=0.4)
            ax.axvline(np.mean(vals), color="white", lw=1.5, linestyle="--", label=f"μ={np.mean(vals):.4f}")
            ax.set_title(k, fontsize=12, fontweight="bold", color="white")
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="gray")
            ax.legend(fontsize=9, labelcolor="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")

        # Hide unused subplot
        for j in range(n_keys, len(axes)):
            axes[j].set_visible(False)

        fig.patch.set_facecolor("#080c14")
        plt.suptitle("ThermalIFNet — Evaluation Metric Distributions",
                     fontsize=15, color="white", y=1.01)
        plt.tight_layout()
        p = self.out_dir / "metric_distributions.png"
        plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Saved → %s", p)

        # Per-sample line plot for PSNR and SSIM
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), facecolor="#080c14")
        for ax, k, color in [(ax1, "PSNR", "#00e5ff"), (ax2, "SSIM", "#00ff88")]:
            vals = accum[k]
            ax.plot(vals, color=color, lw=0.8, alpha=0.8)
            ax.axhline(np.mean(vals), color="white", lw=1.2, linestyle="--",
                       label=f"mean={np.mean(vals):.4f}")
            ax.fill_between(range(len(vals)), vals, alpha=0.15, color=color)
            ax.set_ylabel(k, color="white")
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="gray")
            ax.legend(fontsize=9, labelcolor="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
        ax2.set_xlabel("Sample Index", color="gray")
        plt.suptitle("PSNR & SSIM per Sample", color="white", fontsize=13)
        plt.tight_layout()
        p2 = self.out_dir / "psnr_ssim_per_sample.png"
        plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor="#080c14")
        plt.close(fig2)
        logger.info("Saved → %s", p2)

    def _print_summary(self, summary: dict, elapsed: float, n: int) -> None:
        logger.info("\n%s", "═" * 55)
        logger.info("  EVALUATION SUMMARY  |  samples=%d  |  %.1fs", n, elapsed)
        logger.info("═" * 55)
        for k, s in summary.items():
            unit = " K" if k == "BT_MAE" else (" dB" if k == "PSNR" else "")
            logger.info("  %-8s  mean=%-10.5f  std=%-10.5f  [%.5f – %.5f]%s",
                        k, s["mean"], s["std"], s["min"], s["max"], unit)
        logger.info("═" * 55)


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ThermalIFNet Evaluation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",   required=True,          help="Path to .pth checkpoint")
    p.add_argument("--data_root",    default="data/goes19/raw", help="Root directory of .nc files")
    p.add_argument("--out",          default="outputs/evaluation", help="Output directory")
    p.add_argument("--split",        default="test",         choices=["train","val","test","all"])
    p.add_argument("--stride",       type=int, default=10,   help="Triplet stride (minutes)")
    p.add_argument("--batch_size",   type=int, default=1)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--max_samples",  type=int, default=None, help="Cap number of triplets")
    p.add_argument("--size",         type=int, nargs=2, default=[512, 512], metavar=("H","W"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    evaluator = Evaluator(
        checkpoint_path = args.checkpoint,
        data_root       = args.data_root,
        out_dir         = args.out,
        stride          = args.stride,
        target_size     = tuple(args.size),
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        split           = args.split,
        max_samples     = args.max_samples,
    )
    evaluator.run()


if __name__ == "__main__":
    main()