# src/data_loader/goes_dataset.py
"""
Builds (t0, t2, t1) triplets from GOES .nc files.
t0 = frame at time T
t2 = frame at time T+20min (or T+2 files)
t1 = frame at time T+10min (ground truth midframe)

For GOES-19 ABI M6 mesoscale: 1 file per minute.
Triplet stride = 10 files apart (10 min gap).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
import json

from src.preprocessing.goes_preprocessor import (
    preprocess_frame, TARGET_SIZE, BT_MIN, BT_MAX
)


def discover_nc_files(data_root: str = "data/goes19/raw") -> List[Path]:
    """Recursively find and sort all .nc files."""
    root = Path(data_root)
    files = sorted(root.rglob("*.nc"))
    print(f"Discovered {len(files)} .nc files under {root}")
    return files


def build_triplets(
    files: List[Path],
    stride: int = 10,       # files between t0→t1, t1→t2
    max_triplets: Optional[int] = None,
) -> List[Tuple[Path, Path, Path]]:
    """
    Build list of (t0, t1, t2) path triplets.
    stride=10 means 10-minute gap for 1-min cadence data.
    """
    triplets = []
    n = len(files)
    for i in range(0, n - 2 * stride, 1):
        t0 = files[i]
        t1 = files[i + stride]
        t2 = files[i + 2 * stride]
        triplets.append((t0, t1, t2))
    
    if max_triplets:
        triplets = triplets[:max_triplets]
    
    print(f"Built {len(triplets)} triplets (stride={stride})")
    return triplets


class GOESTripletDataset(Dataset):
    """
    PyTorch Dataset that returns (t0_tensor, t2_tensor, t1_tensor).
    t0, t2 = inputs to interpolation model
    t1     = ground truth midframe
    All tensors: shape (1, H, W), float32, range [0,1]
    """
    
    def __init__(
        self,
        triplets: List[Tuple[Path, Path, Path]],
        target_size: Tuple[int, int] = TARGET_SIZE,
        bt_min: float = BT_MIN,
        bt_max: float = BT_MAX,
        cache: bool = False,
    ):
        self.triplets = triplets
        self.target_size = target_size
        self.bt_min = bt_min
        self.bt_max = bt_max
        self.cache = cache
        self._cache = {}
    
    def __len__(self):
        return len(self.triplets)
    
    def _load(self, path: Path) -> torch.Tensor:
        key = str(path)
        if self.cache and key in self._cache:
            return self._cache[key]
        
        tensor, _, _ = preprocess_frame(
            str(path), self.target_size, self.bt_min, self.bt_max
        )
        t = tensor.squeeze(0)  # (1, H, W)
        
        if self.cache:
            self._cache[key] = t
        return t
    
    def __getitem__(self, idx: int):
        t0_path, t1_path, t2_path = self.triplets[idx]
        
        t0 = self._load(t0_path)
        t1 = self._load(t1_path)   # ground truth
        t2 = self._load(t2_path)
        
        return {
            't0': t0,           # (1, H, W)
            't2': t2,           # (1, H, W)
            't1_gt': t1,        # (1, H, W) ground truth
            't0_path': str(t0_path),
            't1_path': str(t1_path),
            't2_path': str(t2_path),
        }


def get_dataloaders(
    data_root: str = "data/goes19/raw",
    stride: int = 10,
    target_size: Tuple[int, int] = TARGET_SIZE,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    batch_size: int = 4,
    num_workers: int = 4,
    max_triplets: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    files = discover_nc_files(data_root)
    triplets = build_triplets(files, stride, max_triplets)
    
    n = len(triplets)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    
    train_triplets = triplets[:n_train]
    val_triplets = triplets[n_train:n_train + n_val]
    test_triplets = triplets[n_train + n_val:]
    
    print(f"Split → train:{len(train_triplets)} val:{len(val_triplets)} test:{len(test_triplets)}")
    
    train_ds = GOESTripletDataset(train_triplets, target_size)
    val_ds   = GOESTripletDataset(val_triplets, target_size)
    test_ds  = GOESTripletDataset(test_triplets, target_size)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader


# ── CLI test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    files = discover_nc_files("data/goes19/raw")
    
    if len(files) < 21:
        print(f"Need at least 21 files for stride=10 triplets. Found {len(files)}.")
        exit(1)
    
    triplets = build_triplets(files, stride=10, max_triplets=50)
    dataset = GOESTripletDataset(triplets, cache=False)
    
    sample = dataset[0]
    print(f"\nSample keys: {list(sample.keys())}")
    print(f"t0   shape: {sample['t0'].shape}, range: [{sample['t0'].min():.3f}, {sample['t0'].max():.3f}]")
    print(f"t2   shape: {sample['t2'].shape}")
    print(f"t1gt shape: {sample['t1_gt'].shape}")
    print(f"t0 path: {sample['t0_path']}")
    print(f"t1 path: {sample['t1_path']}")
    print(f"t2 path: {sample['t2_path']}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, key, title in zip(axes, ['t0', 't1_gt', 't2'],
                               ['T0 (Input)', 'T1 (Ground Truth)', 'T2 (Input)']):
        ax.imshow(sample[key].squeeze().numpy(), cmap='inferno_r', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
    plt.suptitle("Sample Triplet")
    plt.tight_layout()
    out = Path("outputs/triplet_sample.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"\nSaved → {out}")
    
    # Test DataLoader
    print("\nTesting DataLoader...")
    loader = DataLoader(dataset, batch_size=2, num_workers=0)
    batch = next(iter(loader))
    print(f"Batch t0 shape: {batch['t0'].shape}")
    print(f"Batch t2 shape: {batch['t2'].shape}")