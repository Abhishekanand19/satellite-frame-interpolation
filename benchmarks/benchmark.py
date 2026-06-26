# benchmarks/benchmark.py
"""
Benchmarking suite: Linear vs Optical Flow vs ThermalIFNet.
Computes PSNR, SSIM, RMSE, MAE, BT-MAE, FSIM, inference time.
Generates comparison tables, bar charts, box plots, scatter plots.
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
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.baseline_linear import LinearInterpolator
from benchmarks.baseline_optical_flow import OpticalFlowInterpolator
from src.interpolation.rife_model import RIFEThermalInterpolator, DEVICE
from src.data_loader.goes_dataset import GOESTripletDataset, build_triplets, discover_nc_files
from src.physics_metrics.metrics import compute_all_metrics
from src.preprocessing.goes_preprocessor import TARGET_SIZE, BT_MIN, BT_MAX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("benchmark")

BG     = "#080c14"
PANEL  = "#0d1117"
WHITE  = "#e0e0e0"
DPI    = 150
COLORS = ["#00e5ff", "#ffaa00", "#00ff88"]
METRICS_DISPLAY = ["PSNR", "SSIM", "RMSE", "MAE", "BT_MAE", "FSIM"]


# ── Extra metrics ─────────────────────────────────────────────────────────────
def rmse(p: np.ndarray, g: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - g) ** 2)))

def mae(p: np.ndarray, g: np.ndarray) -> float:
    return float(np.mean(np.abs(p - g)))

def full_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    base = compute_all_metrics(pred, gt)
    base["RMSE"] = rmse(pred, gt)
    base["MAE"]  = mae(pred, gt)
    return base


# ── Benchmark runner ──────────────────────────────────────────────────────────
class Benchmarker:

    def __init__(
        self,
        checkpoint_path: str,
        data_root: str = "data/goes19/raw",
        out_dir: str = "benchmarks/results",
        stride: int = 10,
        target_size: tuple = TARGET_SIZE,
        max_samples: Optional[int] = 100,
        num_workers: int = 4,
        device: torch.device = DEVICE,
        skip_flow: bool = False,
    ):
        self.data_root   = data_root
        self.out_dir     = Path(out_dir)
        self.stride      = stride
        self.target_size = target_size
        self.max_samples = max_samples
        self.device      = device
        self.skip_flow   = skip_flow
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Models
        self.rife   = RIFEThermalInterpolator(checkpoint_path=checkpoint_path, device=device)
        self.rife.model.eval()
        self.linear = LinearInterpolator()
        self.flow   = None if skip_flow else self._try_load_flow()

        logger.info("Benchmark init | device=%s | skip_flow=%s", device, skip_flow)

    def _try_load_flow(self):
        try:
            return OpticalFlowInterpolator()
        except ImportError:
            logger.warning("OpenCV not found — skipping optical flow baseline.")
            return None

    # ── Dataset ───────────────────────────────────────────────────────────────
    def _build_loader(self) -> DataLoader:
        files    = discover_nc_files(self.data_root)
        triplets = build_triplets(files, stride=self.stride,
                                  max_triplets=self.max_samples)
        n        = len(triplets)
        test_tri = triplets[int(n * 0.9):]  # use test split
        if not test_tri:
            test_tri = triplets          # fallback to all if tiny dataset
        ds = GOESTripletDataset(test_tri, target_size=self.target_size)
        logger.info("Benchmark samples: %d", len(ds))
        return DataLoader(ds, batch_size=1, shuffle=False,
                          num_workers=4, pin_memory=True)

    # ── Per-method evaluation ─────────────────────────────────────────────────
    def _eval_rife(self, t0: torch.Tensor, t2: torch.Tensor,
                   gt: torch.Tensor) -> tuple[dict, float]:
        t0g = t0.to(self.device)
        t2g = t2.to(self.device)
        t0_s = time.perf_counter()
        with torch.no_grad():
            pred, _, _ = self.rife.model(t0g, t2g)
            pred = torch.clamp(pred, 0, 1)
        elapsed = time.perf_counter() - t0_s
        p = pred.cpu().squeeze().numpy()
        g = gt.squeeze().numpy()
        return full_metrics(p, g), elapsed

    def _eval_linear(self, t0: np.ndarray, t2: np.ndarray,
                     gt: np.ndarray) -> tuple[dict, float]:
        t0_s = time.perf_counter()
        pred = self.linear.predict(t0, t2)
        elapsed = time.perf_counter() - t0_s
        return full_metrics(pred, gt), elapsed

    def _eval_flow(self, t0: np.ndarray, t2: np.ndarray,
                   gt: np.ndarray) -> tuple[dict, float]:
        t0_s = time.perf_counter()
        pred = self.flow.predict(t0, t2)
        elapsed = time.perf_counter() - t0_s
        return full_metrics(pred, gt), elapsed

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self) -> dict:
        loader = self._build_loader()

        method_names = ["ThermalIFNet (RIFE)", "Linear", "Optical Flow (Farneback)"]
        if self.flow is None:
            method_names = method_names[:2]

        accum = {
            m: {k: [] for k in METRICS_DISPLAY + ["inference_ms"]}
            for m in method_names
        }

        for batch in tqdm(loader, desc="Benchmarking", dynamic_ncols=True):
            t0_t  = batch["t0"]      # (1,1,H,W)
            t2_t  = batch["t2"]
            gt_t  = batch["t1_gt"]

            t0_np = t0_t.squeeze().numpy()
            t2_np = t2_t.squeeze().numpy()
            gt_np = gt_t.squeeze().numpy()

            # RIFE
            m, dt = self._eval_rife(t0_t, t2_t, gt_t)
            self._append(accum["ThermalIFNet (RIFE)"], m, dt)

            # Linear
            m, dt = self._eval_linear(t0_np, t2_np, gt_np)
            self._append(accum["Linear"], m, dt)

            # Optical Flow
            if self.flow is not None:
                try:
                    m, dt = self._eval_flow(t0_np, t2_np, gt_np)
                    self._append(accum["Optical Flow (Farneback)"], m, dt)
                except Exception as e:
                    logger.warning("Flow failed on sample: %s", e)

        results = self._aggregate(accum, method_names)
        self._save_json(results)
        self._print_table(results, method_names)
        self._plot_bar(results, method_names)
        self._plot_box(accum, method_names)
        self._plot_radar(results, method_names)
        self._plot_scatter(accum, method_names)
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _append(store: dict, metrics: dict, dt: float) -> None:
        for k in METRICS_DISPLAY:
            v = metrics.get(k)
            if v is not None and not np.isnan(v):
                store[k].append(v)
        store["inference_ms"].append(dt * 1000)

    @staticmethod
    def _aggregate(accum: dict, methods: list) -> dict:
        out = {}
        for method in methods:
            out[method] = {}
            for k, vals in accum[method].items():
                if vals:
                    out[method][k] = {
                        "mean":   round(float(np.mean(vals)),   5),
                        "std":    round(float(np.std(vals)),    5),
                        "median": round(float(np.median(vals)), 5),
                        "min":    round(float(np.min(vals)),    5),
                        "max":    round(float(np.max(vals)),    5),
                    }
        return out

    def _save_json(self, results: dict) -> None:
        p = self.out_dir / "benchmark_results.json"
        with open(p, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved → %s", p)

    # ── Table ─────────────────────────────────────────────────────────────────
    def _print_table(self, results: dict, methods: list) -> None:
        col_w = 24
        metric_w = 12
        header = f"{'Method':<{col_w}}" + "".join(f"{m:>{metric_w}}" for m in METRICS_DISPLAY + ["Time(ms)"])
        sep    = "─" * len(header)
        logger.info("\n%s\n%s\n%s", sep, header, sep)
        for method in methods:
            row = f"{method:<{col_w}}"
            for k in METRICS_DISPLAY + ["inference_ms"]:
                v = results[method].get(k, {}).get("mean", float("nan"))
                row += f"{v:>{metric_w}.4f}"
            logger.info(row)
        logger.info(sep)

        # Also save as txt
        lines = [sep, header, sep]
        for method in methods:
            row = f"{method:<{col_w}}"
            for k in METRICS_DISPLAY + ["inference_ms"]:
                v = results[method].get(k, {}).get("mean", float("nan"))
                row += f"{v:>{metric_w}.4f}"
            lines.append(row)
        lines.append(sep)
        p = self.out_dir / "benchmark_table.txt"
        p.write_text("\n".join(lines))
        logger.info("Table saved → %s", p)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    def _plot_bar(self, results: dict, methods: list) -> None:
        metrics_to_plot = METRICS_DISPLAY + ["inference_ms"]
        n_m = len(metrics_to_plot)
        fig, axes = plt.subplots(2, 4, figsize=(24, 10), facecolor=BG)
        axes = axes.flatten()
        fig.suptitle("Benchmark — Method Comparison (Mean ± Std)",
                     color=WHITE, fontsize=14, y=1.01)

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            ax.set_facecolor(PANEL)
            means = [results[m].get(metric, {}).get("mean", 0) for m in methods]
            stds  = [results[m].get(metric, {}).get("std",  0) for m in methods]
            x     = np.arange(len(methods))
            bars  = ax.bar(x, means, yerr=stds, color=COLORS[:len(methods)],
                           alpha=0.85, capsize=5, error_kw={"ecolor": "white", "lw": 1.2})
            ax.set_xticks(x)
            ax.set_xticklabels([m.split(" ")[0] for m in methods],
                               color="gray", fontsize=9, rotation=10)
            ax.set_title(metric, color=WHITE, fontsize=11, fontweight="bold")
            ax.tick_params(colors="gray")
            ax.set_facecolor(PANEL)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        f"{mean:.4f}", ha="center", va="bottom", color=WHITE, fontsize=8)

        for j in range(len(metrics_to_plot), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        p = str(self.out_dir / "benchmark_bar.png")
        fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info("Saved → %s", p)

    # ── Box plots ─────────────────────────────────────────────────────────────
    def _plot_box(self, accum: dict, methods: list) -> None:
        key_metrics = ["PSNR", "SSIM", "RMSE", "MAE"]
        fig, axes = plt.subplots(1, 4, figsize=(22, 7), facecolor=BG)
        fig.suptitle("Benchmark — Distribution per Method", color=WHITE, fontsize=13, y=1.01)

        for ax, metric in zip(axes, key_metrics):
            data   = [accum[m][metric] for m in methods if accum[m][metric]]
            labels = [m.split(" ")[0] for m in methods if accum[m][metric]]
            bp = ax.boxplot(data, labels=labels, patch_artist=True,
                            medianprops={"color": "white", "lw": 2},
                            whiskerprops={"color": "gray"},
                            capprops={"color": "gray"},
                            flierprops={"markerfacecolor": "#555", "markersize": 3})
            for patch, color in zip(bp["boxes"], COLORS):
                patch.set_facecolor(color + "55")
                patch.set_edgecolor(color)
            ax.set_title(metric, color=WHITE, fontsize=11, fontweight="bold")
            ax.set_facecolor(PANEL)
            ax.tick_params(colors="gray")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")

        plt.tight_layout()
        p = str(self.out_dir / "benchmark_boxplot.png")
        fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info("Saved → %s", p)

    # ── Radar chart ───────────────────────────────────────────────────────────
    def _plot_radar(self, results: dict, methods: list) -> None:
        radar_metrics = ["PSNR", "SSIM", "FSIM"]
        N = len(radar_metrics)

        # Normalize per metric for radar (0→worst, 1→best)
        raw = {m: [results[m].get(k, {}).get("mean", 0) for k in radar_metrics]
               for m in methods}
        col_vals = [[raw[m][i] for m in methods] for i in range(N)]
        norm_vals = {}
        for m in methods:
            norm_vals[m] = []
            for i, k in enumerate(radar_metrics):
                mn, mx = min(col_vals[i]), max(col_vals[i])
                v = raw[m][i]
                norm_vals[m].append((v - mn) / (mx - mn + 1e-9))

        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True}, facecolor=BG)
        ax.set_facecolor(PANEL)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics, color=WHITE, fontsize=11)
        ax.tick_params(colors="gray")
        ax.set_rlabel_position(0)
        ax.yaxis.set_tick_params(labelcolor="gray", labelsize=7)
        ax.grid(color="#333", linewidth=0.8)

        for method, color in zip(methods, COLORS):
            vals = norm_vals[method] + norm_vals[method][:1]
            ax.plot(angles, vals, color=color, lw=2, label=method.split(" ")[0])
            ax.fill(angles, vals, color=color, alpha=0.15)

        ax.set_title("Normalized Performance Radar\n(higher = better)",
                     color=WHITE, fontsize=12, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
                  labelcolor=WHITE, facecolor=PANEL, edgecolor="#333", fontsize=9)

        p = str(self.out_dir / "benchmark_radar.png")
        fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info("Saved → %s", p)

    # ── Scatter: PSNR vs inference time ──────────────────────────────────────
    def _plot_scatter(self, accum: dict, methods: list) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
        fig.suptitle("Quality vs Speed Trade-off", color=WHITE, fontsize=13, y=1.01)

        for ax, (xk, yk, xl, yl) in zip(axes, [
            ("inference_ms", "PSNR", "Inference Time (ms)", "PSNR (dB)"),
            ("inference_ms", "SSIM", "Inference Time (ms)", "SSIM"),
        ]):
            ax.set_facecolor(PANEL)
            for method, color in zip(methods, COLORS):
                xs = accum[method].get(xk, [])
                ys = accum[method].get(yk, [])
                n  = min(len(xs), len(ys))
                if n:
                    ax.scatter(xs[:n], ys[:n], color=color, alpha=0.55, s=20,
                               label=method.split(" ")[0])
                    ax.scatter([np.mean(xs[:n])], [np.mean(ys[:n])],
                               color=color, s=120, marker="*", zorder=5)
            ax.set_xlabel(xl, color="gray", fontsize=10)
            ax.set_ylabel(yl, color="gray", fontsize=10)
            ax.tick_params(colors="gray")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            ax.legend(labelcolor=WHITE, facecolor=PANEL, edgecolor="#333", fontsize=9)

        plt.tight_layout()
        p = str(self.out_dir / "benchmark_scatter.png")
        fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info("Saved → %s", p)


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ThermalIFNet Benchmark CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--data_root",    default="data/goes19/raw")
    p.add_argument("--out",          default="benchmarks/results")
    p.add_argument("--stride",       type=int, default=10)
    p.add_argument("--max_samples",  type=int, default=100)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--skip_flow",    action="store_true",
                   help="Skip optical flow baseline (if OpenCV unavailable)")
    p.add_argument("--size",         type=int, nargs=2, default=[512, 512], metavar=("H","W"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    bench = Benchmarker(
        checkpoint_path = args.checkpoint,
        data_root       = args.data_root,
        out_dir         = args.out,
        stride          = args.stride,
        target_size     = tuple(args.size),
        max_samples     = args.max_samples,
        num_workers     = args.num_workers,
        skip_flow       = args.skip_flow,
    )
    bench.run()


if __name__ == "__main__":
    main()