# src/interpolation/trainer.py
"""
Fine-tune ThermalIFNet on GOES BT data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Optional
import time
import json

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.interpolation.rife_model import RIFEThermalInterpolator, DEVICE
from src.data_loader.goes_dataset import get_dataloaders


class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.84):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def ssim_loss(self, pred, target):
        from torchmetrics.functional import structural_similarity_index_measure as ssim
        return 1 - ssim(pred, target, data_range=1.0)

    def forward(self, pred, target):
        l1 = self.l1(pred, target)
        try:
            ssim = self.ssim_loss(pred, target)
            return self.alpha * ssim + (1 - self.alpha) * l1
        except Exception:
            return l1


def train(
    data_root: str = "data/goes19/raw",
    checkpoint_dir: str = "models/checkpoints",
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    stride: int = 10,
    max_triplets: Optional[int] = None,
    resume: Optional[str] = None,
):
    print(f"Device: {DEVICE}")
    
    # Data
    train_loader, val_loader, _ = get_dataloaders(
        data_root=data_root,
        stride=stride,
        batch_size=batch_size,
        max_triplets=max_triplets,
    )
    
    # Model
    interpolator = RIFEThermalInterpolator(checkpoint_path=resume)
    model = interpolator.model
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    criterion = CombinedLoss().to(DEVICE)
    
    start_epoch = 0
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=DEVICE)
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")
    
    history = {'train_loss': [], 'val_loss': []}
    best_val = float('inf')
    
    for epoch in range(start_epoch, epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        t0_epoch = time.time()
        
        for i, batch in enumerate(train_loader):
            t0 = batch['t0'].to(DEVICE)
            t2 = batch['t2'].to(DEVICE)
            t1_gt = batch['t1_gt'].to(DEVICE)
            
            optimizer.zero_grad()
            with autocast():
                pred, _, _ = model(t0, t2)
                loss = criterion(pred, t1_gt)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if i % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {i}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}")
        
        # ── Val ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                t0 = batch['t0'].to(DEVICE)
                t2 = batch['t2'].to(DEVICE)
                t1_gt = batch['t1_gt'].to(DEVICE)
                with autocast():
                    pred, _, _ = model(t0, t2)
                    loss = criterion(pred, t1_gt)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step()
        
        elapsed = time.time() - t0_epoch
        print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train:.4f} | "
              f"Val: {avg_val:.4f} | Time: {elapsed:.1f}s")
        
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        
        # Save best
        if avg_val < best_val:
            best_val = avg_val
            interpolator.save(
                f"{checkpoint_dir}/best_model.pth",
                epoch=epoch + 1,
                metrics={'val_loss': avg_val}
            )
        
        # Save periodic
        if (epoch + 1) % 10 == 0:
            interpolator.save(
                f"{checkpoint_dir}/epoch_{epoch+1:03d}.pth",
                epoch=epoch + 1
            )
        
        # Save history
        with open(f"{checkpoint_dir}/history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    return history


if __name__ == "__main__":
    train(
        data_root="data/goes19/raw",
        epochs=50,
        batch_size=4,
        stride=10,
        max_triplets=500,  # limit for quick test; remove for full training
    )