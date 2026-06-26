"""
Production trainer for ThermalIFNet — dataset-agnostic temporal interpolation.
Supports GOES now; INSAT later via config/dataloader swap only.
"""

import logging
import os
import random
import sys
import time
import argparse
import yaml
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_loader.goes_dataset import GOESTripletDataset, build_triplets, discover_nc_files
from src.interpolation.rife_model import RIFEThermalInterpolator, DEVICE
from src.physics_metrics.metrics import compute_all_metrics
from src.config.config_loader import load_config

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("trainer")


# ── Loss ──────────────────────────────────────────────────────────────────────
class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps))


class CombinedLoss(nn.Module):
    """
    Weighted sum of Charbonnier + SSIM loss.
    Charbonnier is robust to outliers; SSIM preserves structural detail.
    """
    def __init__(self, alpha: float = 0.84, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.charb = CharbonnierLoss(eps)

    def _ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_p = torch.nn.functional.avg_pool2d(pred,   3, 1, 1)
        mu_t = torch.nn.functional.avg_pool2d(target, 3, 1, 1)
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sig_p  = torch.nn.functional.avg_pool2d(pred   ** 2, 3, 1, 1) - mu_p2
        sig_t  = torch.nn.functional.avg_pool2d(target ** 2, 3, 1, 1) - mu_t2
        sig_pt = torch.nn.functional.avg_pool2d(pred * target, 3, 1, 1) - mu_pt
        ssim_map = ((2 * mu_pt + c1) * (2 * sig_pt + c2)) / \
                   ((mu_p2 + mu_t2 + c1) * (sig_p + sig_t + c2))
        return 1.0 - ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self._ssim(pred, target) + (1 - self.alpha) * self.charb(pred, target)


# ── Config ────────────────────────────────────────────────────────────────────
class TrainerConfig:
    def __init__(self, **kwargs):
        # Data
        self.data_root:        str   = kwargs.get("data_root",        "data/goes19/raw")
        self.stride:           int   = kwargs.get("stride",           10)
        self.target_size:      tuple = kwargs.get("target_size",      (512, 512))
        self.train_ratio:      float = kwargs.get("train_ratio",      0.80)
        self.val_ratio:        float = kwargs.get("val_ratio",        0.10)
        self.max_triplets:     Optional[int] = kwargs.get("max_triplets", None)

        # Training
        self.epochs:           int   = kwargs.get("epochs",           100)
        self.batch_size:       int   = kwargs.get("batch_size",       4)
        self.num_workers:      int   = kwargs.get("num_workers",      4)
        self.seed:             int   = kwargs.get("seed",             42)
        self.amp:              bool  = kwargs.get("amp",              True)
        self.grad_clip:        float = kwargs.get("grad_clip",        1.0)

        # Optimizer
        self.lr:               float = kwargs.get("lr",               1e-4)
        self.weight_decay:     float = kwargs.get("weight_decay",     1e-4)
        self.betas:            tuple = kwargs.get("betas",            (0.9, 0.999))

        # Scheduler
        self.scheduler:        str   = kwargs.get("scheduler",        "cosine")   # cosine | step | plateau
        self.lr_step_size:     int   = kwargs.get("lr_step_size",     20)
        self.lr_gamma:         float = kwargs.get("lr_gamma",         0.5)
        self.eta_min:          float = kwargs.get("eta_min",          1e-7)

        # Loss
        self.loss_alpha:       float = kwargs.get("loss_alpha",       0.84)

        # Checkpointing
        self.checkpoint_dir:   str   = kwargs.get("checkpoint_dir",   "models/checkpoints")
        self.resume:           Optional[str] = kwargs.get("resume",   None)
        self.save_every:       int   = kwargs.get("save_every",       10)

        # Early stopping
        self.early_stop_patience: int  = kwargs.get("early_stop_patience", 15)
        self.early_stop_delta:    float = kwargs.get("early_stop_delta",   1e-5)

        # Logging
        self.log_dir:          str   = kwargs.get("log_dir",          "runs")
        self.log_every:        int   = kwargs.get("log_every",        10)

    def __repr__(self) -> str:
        return "\n".join(f"  {k}: {v}" for k, v in vars(self).items())


# ── Early Stopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 15, delta: float = 1e-5):
        self.patience  = patience
        self.delta     = delta
        self.best      = float("inf")
        self.counter   = 0
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.delta:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ── Trainer ───────────────────────────────────────────────────────────────────
class Trainer:
    """
    Dataset-agnostic trainer for ThermalIFNet.
    Switch from GOES to INSAT by swapping the DataLoader only.
    """

    def __init__(self, cfg: TrainerConfig):
        self.cfg    = cfg
        self.device = DEVICE
        self._fix_seed(cfg.seed)

        # Dirs
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

        # Components
        self.interpolator = RIFEThermalInterpolator(checkpoint_path=cfg.resume, device=self.device)
        self.model        = self.interpolator.model
        self.criterion    = CombinedLoss(alpha=cfg.loss_alpha).to(self.device)
        self.optimizer    = self._build_optimizer()
        self.scheduler    = self._build_scheduler()
        self.scaler       = GradScaler(enabled=cfg.amp)
        self.stopper      = EarlyStopping(cfg.early_stop_patience, cfg.early_stop_delta)
        self.writer       = SummaryWriter(log_dir=cfg.log_dir)

        self.start_epoch  = 0
        self.best_val     = float("inf")
        self.global_step  = 0

        if cfg.resume and Path(cfg.resume).exists():
            self._load_checkpoint(cfg.resume)

        logger.info("Device      : %s", self.device)
        logger.info("AMP enabled : %s", cfg.amp)
        logger.info("Config:\n%s", cfg)

    # ── Setup helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _fix_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    def _build_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            betas=self.cfg.betas,
            weight_decay=self.cfg.weight_decay,
        )

    def _build_scheduler(self):
        s = self.cfg.scheduler
        if s == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.epochs, eta_min=self.cfg.eta_min
            )
        if s == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.cfg.lr_step_size, gamma=self.cfg.lr_gamma
            )
        if s == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=5, factor=self.cfg.lr_gamma
            )
        raise ValueError(f"Unknown scheduler: {s}")

    # ── DataLoaders ─────────────────────────────────────────────────────────────
    def _build_dataloaders(self):
        """
        Dataset-agnostic boundary point:
        Replace discover_nc_files + build_triplets + GOESTripletDataset
        with INSAT equivalents here when INSAT data is available.
        Everything downstream (training loop, metrics, logging) stays unchanged.
        """
        files   = discover_nc_files(self.cfg.data_root)
        triplets = build_triplets(files, stride=self.cfg.stride,
                                  max_triplets=self.cfg.max_triplets)

        n       = len(triplets)
        n_train = int(n * self.cfg.train_ratio)
        n_val   = int(n * self.cfg.val_ratio)

        train_ds = GOESTripletDataset(triplets[:n_train],           self.cfg.target_size)
        val_ds   = GOESTripletDataset(triplets[n_train:n_train+n_val], self.cfg.target_size)
        test_ds  = GOESTripletDataset(triplets[n_train+n_val:],     self.cfg.target_size)

        logger.info("Triplets — train:%d  val:%d  test:%d", len(train_ds), len(val_ds), len(test_ds))

        loader_kwargs = dict(
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )
        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size,
                                  shuffle=True,  **loader_kwargs)
        val_loader   = DataLoader(val_ds,   batch_size=self.cfg.batch_size,
                                  shuffle=False, **loader_kwargs)
        test_loader  = DataLoader(test_ds,  batch_size=1,
                                  shuffle=False, **loader_kwargs)
        return train_loader, val_loader, test_loader

    # ── Checkpoint ────────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, tag: str, metrics: dict) -> None:
        path = Path(self.cfg.checkpoint_dir) / f"{tag}.pth"
        torch.save({
            "epoch":            epoch,
            "global_step":      self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "scheduler_state":  self.scheduler.state_dict(),
            "scaler_state":     self.scaler.state_dict(),
            "best_val":         self.best_val,
            "metrics":          metrics,
            "config":           vars(self.cfg),
        }, path)
        logger.info("Checkpoint saved → %s", path)

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.start_epoch  = ckpt.get("epoch", 0)
        self.global_step  = ckpt.get("global_step", 0)
        self.best_val     = ckpt.get("best_val", float("inf"))
        logger.info("Resumed from %s  (epoch %d)", path, self.start_epoch)

    # ── One epoch ─────────────────────────────────────────────────────────────
    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Train E{epoch:03d}", leave=False, dynamic_ncols=True)

        for batch in pbar:
            t0    = batch["t0"].to(self.device, non_blocking=True)
            t2    = batch["t2"].to(self.device, non_blocking=True)
            t1_gt = batch["t1_gt"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.cfg.amp):
                pred, _, _ = self.model(t0, t2)
                loss       = self.criterion(pred, t1_gt)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_val = loss.item()
            total_loss += loss_val
            self.global_step += 1

            if self.global_step % self.cfg.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/loss_step", loss_val, self.global_step)
                self.writer.add_scalar("train/lr",        lr,       self.global_step)

            pbar.set_postfix(loss=f"{loss_val:.5f}")

        return total_loss / len(loader)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, epoch: int) -> tuple[float, dict]:
        self.model.eval()
        total_loss   = 0.0
        all_metrics  = {k: [] for k in ("MSE", "PSNR", "SSIM", "BT_MAE", "FSIM")}
        pbar = tqdm(loader, desc=f"Val   E{epoch:03d}", leave=False, dynamic_ncols=True)

        for batch in pbar:
            t0    = batch["t0"].to(self.device, non_blocking=True)
            t2    = batch["t2"].to(self.device, non_blocking=True)
            t1_gt = batch["t1_gt"].to(self.device, non_blocking=True)

            with autocast(enabled=self.cfg.amp):
                pred, _, _ = self.model(t0, t2)
                loss       = self.criterion(pred, t1_gt)

            total_loss += loss.item()

            pred_np = pred.cpu().float().numpy()
            gt_np   = t1_gt.cpu().float().numpy()

            for i in range(pred_np.shape[0]):
                m = compute_all_metrics(pred_np[i, 0], gt_np[i, 0])
                for k, v in m.items():
                    all_metrics[k].append(v)

            pbar.set_postfix(loss=f"{loss.item():.5f}")

        avg_loss    = total_loss / len(loader)
        avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items() if v}
        return avg_loss, avg_metrics

    # ── Main train loop ───────────────────────────────────────────────────────
    def fit(self) -> None:
        train_loader, val_loader, _ = self._build_dataloaders()
        logger.info("Starting training — %d epochs on %s", self.cfg.epochs, self.device)

        for epoch in range(self.start_epoch + 1, self.cfg.epochs + 1):
            t_start = time.time()

            train_loss            = self._train_epoch(train_loader, epoch)
            val_loss, val_metrics = self._val_epoch(val_loader, epoch)

            # Scheduler step
            if self.cfg.scheduler == "plateau":
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            elapsed = time.time() - t_start
            lr      = self.optimizer.param_groups[0]["lr"]

            # ── TensorBoard ──
            self.writer.add_scalar("train/loss_epoch", train_loss, epoch)
            self.writer.add_scalar("val/loss",         val_loss,   epoch)
            self.writer.add_scalar("val/SSIM",         val_metrics.get("SSIM",   0), epoch)
            self.writer.add_scalar("val/PSNR",         val_metrics.get("PSNR",   0), epoch)
            self.writer.add_scalar("val/BT_MAE",       val_metrics.get("BT_MAE", 0), epoch)
            self.writer.add_scalar("val/FSIM",         val_metrics.get("FSIM",   0), epoch)
            self.writer.add_scalar("train/lr_epoch",   lr,                           epoch)

            logger.info(
                "Epoch %03d/%03d | train=%.5f | val=%.5f | "
                "SSIM=%.4f | PSNR=%.2f | BT-MAE=%.3fK | lr=%.2e | %.1fs",
                epoch, self.cfg.epochs,
                train_loss, val_loss,
                val_metrics.get("SSIM", 0),
                val_metrics.get("PSNR", 0),
                val_metrics.get("BT_MAE", 0),
                lr, elapsed,
            )

            # ── Save best ──
            if val_loss < self.best_val:
                self.best_val = val_loss
                self._save_checkpoint(epoch, "best_model", val_metrics)

            # ── Save periodic ──
            if epoch % self.cfg.save_every == 0:
                self._save_checkpoint(epoch, f"epoch_{epoch:03d}", val_metrics)

            # ── Early stopping ──
            if self.stopper.step(val_loss):
                logger.info("Early stopping triggered at epoch %d (patience=%d)",
                            epoch, self.cfg.early_stop_patience)
                break

        self._save_checkpoint(epoch, "final_model", val_metrics)
        self.writer.close()
        logger.info("Training complete. Best val loss: %.6f", self.best_val)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self) -> dict:
        _, _, test_loader = self._build_dataloaders()
        best_ckpt = Path(self.cfg.checkpoint_dir) / "best_model.pth"
        if best_ckpt.exists():
            self._load_checkpoint(str(best_ckpt))
            logger.info("Loaded best checkpoint for evaluation")

        self.model.eval()
        all_metrics = {k: [] for k in ("MSE", "PSNR", "SSIM", "BT_MAE", "FSIM")}
        pbar = tqdm(test_loader, desc="Evaluating", dynamic_ncols=True)

        for batch in pbar:
            t0    = batch["t0"].to(self.device)
            t2    = batch["t2"].to(self.device)
            t1_gt = batch["t1_gt"].to(self.device)

            with autocast(enabled=self.cfg.amp):
                pred, _, _ = self.model(t0, t2)

            m = compute_all_metrics(pred.cpu().float().numpy().squeeze(),
                                    t1_gt.cpu().float().numpy().squeeze())
            for k, v in m.items():
                all_metrics[k].append(v)

        results = {k: float(np.mean(v)) for k, v in all_metrics.items() if v}
        logger.info("Test metrics: %s", results)
        return results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--set", nargs="*", help="key=value overrides (e.g., --set dataset=insat train.batch_size=8)")
    args = p.parse_args()

    # Process command line overrides dynamically
    overrides = {}
    for kv in (args.set or []):
        if "=" in kv:
            k, v = kv.split("=", 1)
            if "." in k:
                parent, child = k.split(".", 1)
                if parent not in overrides:
                    overrides[parent] = {}
                overrides[parent][child] = yaml.safe_load(v)
            else:
                overrides[k] = yaml.safe_load(v)

    # Load the unified config object from the YAML files
    cfg = load_config(overrides=overrides)

    try:
        dataset_name = cfg.dataset.name.upper()
    except AttributeError:
        dataset_name = "GOES (Default)"
        
    logger.info(f"--- Initializing Trainer for dataset tier: {dataset_name} ---")

    # Pass the loaded YAML configurations directly into the TrainerConfig object
    # If a variable is missing from YAML, it falls back to a safe default.
    trainer_cfg = TrainerConfig(
        data_root      = getattr(cfg.dataset, 'data_root', "data/goes19/raw"),
        stride         = getattr(cfg.dataset, 'stride', 10),
        target_size    = tuple(getattr(cfg.dataset, 'target_size', [512, 512])),
        epochs         = getattr(cfg.train, 'epochs', 100) if hasattr(cfg, 'train') else 100,
        batch_size     = getattr(cfg.train, 'batch_size', 4) if hasattr(cfg, 'train') else 4,
        lr             = getattr(cfg.optimizer, 'lr', 1e-4) if hasattr(cfg, 'optimizer') else 1e-4,
        checkpoint_dir = getattr(cfg.paths, 'checkpoint_dir', "models/checkpoints") if hasattr(cfg, 'paths') else "models/checkpoints",
        log_dir        = getattr(cfg.paths, 'log_dir', "runs/thermal_ifnet") if hasattr(cfg, 'paths') else "runs/thermal_ifnet"
    )

    trainer = Trainer(trainer_cfg)
    trainer.fit()
    trainer.evaluate()