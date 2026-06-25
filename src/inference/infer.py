# src/inference/infer.py
"""
CLI inference script for ThermalIFNet.

Usage examples:
  # Single pair
  python src/inference/infer.py \
      --t0 data/goes19/raw/day001/frame_00.nc \
      --t2 data/goes19/raw/day001/frame_20.nc \
      --checkpoint models/checkpoints/best_model.pth \
      --out outputs/inference

  # Batch from CSV  (columns: t0_path,t2_path)
  python src/inference/infer.py \
      --batch_csv scripts/pairs.csv \
      --checkpoint models/checkpoints/best_model.pth \
      --out outputs/batch

  # Auto-build pairs from a directory (every Nth file)
  python src/inference/infer.py \
      --dir data/goes19/raw \
      --stride 10 \
      --checkpoint models/checkpoints/best_model.pth \
      --out outputs/batch
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.predictor import ThermalPredictor
from src.data_loader.goes_dataset import discover_nc_files
from src.physics_metrics.metrics import compute_all_metrics, print_metrics
from src.preprocessing.goes_preprocessor import preprocess_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("infer")


# ── Arg parser ─────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ThermalIFNet Inference CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input modes (mutually exclusive)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--t0",        type=str, help="Path to T0 .nc file (single mode)")
    mode.add_argument("--batch_csv", type=str, help="CSV with columns: t0_path,t2_path")
    mode.add_argument("--dir",       type=str, help="Directory to auto-scan for .nc files")

    p.add_argument("--t2",         type=str,   default=None,
                   help="Path to T2 .nc file (required with --t0)")
    p.add_argument("--gt",         type=str,   default=None,
                   help="Path to ground truth .nc file (optional, computes metrics)")
    p.add_argument("--stride",     type=int,   default=10,
                   help="Frame stride for --dir mode")
    p.add_argument("--checkpoint", type=str,   required=True,
                   help="Path to trained .pth checkpoint")
    p.add_argument("--out",        type=str,   default="outputs/inference",
                   help="Output directory")
    p.add_argument("--timestep",   type=float, default=0.5,
                   help="Interpolation timestep [0,1]")
    p.add_argument("--no_png",     action="store_true", help="Skip PNG output")
    p.add_argument("--no_npy",     action="store_true", help="Skip .npy output")
    p.add_argument("--no_compare", action="store_true", help="Skip comparison figure")
    p.add_argument("--size",       type=int,   nargs=2, default=[512, 512],
                   metavar=("H", "W"), help="Target spatial resolution")
    return p


# ── Helpers ───────────────────────────────────────────────────────────────────
def pairs_from_csv(csv_path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["t0_path"].strip(), row["t2_path"].strip()))
    logger.info("Loaded %d pairs from %s", len(pairs), csv_path)
    return pairs


def pairs_from_dir(data_dir: str, stride: int) -> list[tuple[str, str]]:
    files = discover_nc_files(data_dir)
    pairs = [(str(files[i]), str(files[i + stride]))
             for i in range(len(files) - stride)]
    logger.info("Auto-built %d pairs from %s (stride=%d)", len(pairs), data_dir, stride)
    return pairs


def run_single(
    predictor: ThermalPredictor,
    t0: str,
    t2: str,
    out_dir: str,
    stem: str,
    args: argparse.Namespace,
    gt_path: str = None,
) -> None:
    t_start = time.time()
    result = predictor.predict(t0, t2, timestep=args.timestep)
    elapsed = time.time() - t_start

    saved = predictor.save_result(
        result,
        out_dir=out_dir,
        stem=stem,
        save_npy=not args.no_npy,
        save_png=not args.no_png,
        save_comparison=not args.no_compare,
    )

    logger.info("Inference time: %.3fs", elapsed)
    for k, v in saved.items():
        logger.info("  [%s] → %s", k, v)

    # Optional metrics vs ground truth
    if gt_path:
        logger.info("Computing metrics vs ground truth: %s", gt_path)
        gt_tensor, _, _ = preprocess_frame(gt_path,
                                           target_size=tuple(args.size),
                                           return_tensor=True)
        gt_np = gt_tensor.squeeze().numpy()
        m = compute_all_metrics(result["prediction_norm"], gt_np)
        print_metrics(m, prefix=f"Metrics vs GT [{stem}]")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    predictor = ThermalPredictor(
        checkpoint_path=args.checkpoint,
        target_size=tuple(args.size),
    )

    out_dir = args.out

    # ── Single mode ──
    if args.t0:
        if not args.t2:
            parser.error("--t2 is required when using --t0")
        stem = f"{Path(args.t0).stem}__to__{Path(args.t2).stem}"
        run_single(predictor, args.t0, args.t2, out_dir, stem, args, gt_path=args.gt)

    # ── Batch CSV mode ──
    elif args.batch_csv:
        pairs = pairs_from_csv(args.batch_csv)
        for i, (t0, t2) in enumerate(pairs):
            stem = f"pair_{i:04d}"
            logger.info("─── Pair %d/%d ───", i + 1, len(pairs))
            try:
                run_single(predictor, t0, t2, out_dir, stem, args)
            except Exception as e:
                logger.error("Failed pair %d: %s", i, e)

    # ── Dir mode ──
    elif args.dir:
        pairs = pairs_from_dir(args.dir, args.stride)
        for i, (t0, t2) in enumerate(pairs):
            stem = f"pair_{i:04d}"
            logger.info("─── Pair %d/%d ───", i + 1, len(pairs))
            try:
                run_single(predictor, t0, t2, out_dir, stem, args)
            except Exception as e:
                logger.error("Failed pair %d: %s", i, e)

    logger.info("All done. Outputs → %s", out_dir)


if __name__ == "__main__":
    main()