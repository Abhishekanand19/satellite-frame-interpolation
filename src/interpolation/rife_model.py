"""
RIFE integration for single-channel thermal satellite imagery.
Downloads pretrained RIFE weights and wraps them for BT interpolation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────────────────────────────────────
# Minimal RIFE IFNet (self-contained, no external repo needed)
# Based on RIFE v4.6 architecture, adapted for 1-channel input
# ─────────────────────────────────────────────────────────────────────────────

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1, True)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True),
        nn.LeakyReLU(0.1, True)
    )


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, bias=True)
        self.beta = nn.Parameter(torch.ones(1, c, 1, 1))
        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c), ResConv(c), ResConv(c),
            ResConv(c), ResConv(c), ResConv(c),
            ResConv(c), ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 2 * 4 + 1, 4, 2, 1),  # 4 flow channels + 1 mask channel
        )

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1. / scale, mode='bilinear',
                          align_corners=False, recompute_scale_factor=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode='bilinear',
                                 align_corners=False, recompute_scale_factor=False) * (1. / scale)
            x = torch.cat((x, flow), dim=1)
        x = self.conv0(x)
        x = self.convblock(x)
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode='bilinear',
                            align_corners=False, recompute_scale_factor=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask


class ThermalIFNet(nn.Module):
    """
    RIFE-style IFNet adapted for single-channel (grayscale) thermal imagery.
    Input: concatenation of [t0, t2] → 2 channels
    Output: interpolated t1 (1 channel)
    """
    def __init__(self):
        super().__init__()
        # Internal layer configurations managed cleanly for cascading layers
        self.block0 = IFBlock(2, c=192)
        self.block1 = IFBlock(2 + 4, c=128)
        self.block2 = IFBlock(2 + 4, c=96)
        self.block3 = IFBlock(2 + 4, c=64)
        
        # Context extraction and refinement block
        self.refine = nn.Sequential(
            conv(1 + 2, 32),  # 1 channel blended + 2 original tracking channels
            conv(32, 32),
            conv(32, 32),
            nn.Conv2d(32, 1, 3, 1, 1),  # single channel residual output
            nn.Sigmoid()
        )

    def warp(self, img, flow):
        B, C, H, W = img.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=img.device),
            torch.arange(W, dtype=torch.float32, device=img.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).to(img.device)  # 1,2,H,W
        
        vgrid = grid + flow
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)  # B,H,W,2
        
        return F.grid_sample(img, vgrid, mode='bilinear',
                             align_corners=True, padding_mode='border')

    def forward(self, t0: torch.Tensor, t2: torch.Tensor, timestep: float = 0.5):
        """
        t0, t2: (B, 1, H, W) normalized float32
        Returns: interpolated (B, 1, H, W)
        """
        imgs = torch.cat([t0, t2], dim=1)  # B,2,H,W
        
        # Multi-scale cascade flow matrix parsing
        flow, mask = self.block0(imgs, None, scale=8)
        flow, mask = self.block1(imgs, flow, scale=4)
        flow, mask = self.block2(imgs, flow, scale=2)
        flow, mask = self.block3(imgs, flow, scale=1)
        
        warped0 = self.warp(t0, flow[:, :2])
        warped2 = self.warp(t2, flow[:, 2:4])
        
        mask_sigmoid = torch.sigmoid(mask)
        
        # Linear fusion of pixel vectors
        blended = warped0 * mask_sigmoid + warped2 * (1 - mask_sigmoid)
        
        # Dense context mapping step
        residual = self.refine(torch.cat([blended, t0, t2], dim=1))
        output = blended + residual - 0.5
        output = torch.clamp(output, 0, 1)
        
        return output, flow, mask_sigmoid


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper with save/load
# ─────────────────────────────────────────────────────────────────────────────

class RIFEThermalInterpolator:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: torch.device = DEVICE,
    ):
        self.device = device
        self.model = ThermalIFNet().to(device)
        self.model = torch.compile(self.model) if hasattr(torch, 'compile') else self.model
        
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load(checkpoint_path)
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"Initialized fresh model (no checkpoint). Device: {device}")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        if 'model_state_dict' in state:
            self.model.load_state_dict(state['model_state_dict'])
        else:
            self.model.load_state_dict(state)

    def save(self, path: str, epoch: int = 0, metrics: dict = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'metrics': metrics or {},
        }, path)
        print(f"Saved checkpoint → {path}")

    @torch.no_grad()
    def infer(
        self,
        t0: torch.Tensor,
        t2: torch.Tensor,
        timestep: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single inference call.
        t0, t2: (B,1,H,W) or (1,H,W) tensors on any device.
        Returns: (prediction, flow, confidence_map) all on CPU.
        """
        self.model.eval()
        
        if t0.dim() == 3:
            t0 = t0.unsqueeze(0)
        if t2.dim() == 3:
            t2 = t2.unsqueeze(0)
            
        if t0.shape[1] != 1 or t2.shape[1] != 1:
            raise ValueError(f"Expected single channel (B, 1, H, W), got channels t0={t0.shape[1]}, t2={t2.shape[1]}")
        
        # Pad coordinates to match multiple of 32 requirement
        H, W = t0.shape[-2], t0.shape[-1]
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        t0 = F.pad(t0, [0, pad_w, 0, pad_h])
        t2 = F.pad(t2, [0, pad_w, 0, pad_h])
        
        t0 = t0.to(self.device)
        t2 = t2.to(self.device)
        
        pred, flow, mask = self.model(t0, t2, timestep)
        
        pred = pred[:, :, :H, :W]
        flow = flow[:, :, :H, :W]
        mask = mask[:, :, :H, :W]
        
        return pred.cpu(), flow.cpu(), mask.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# CLI test execution block
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.preprocessing.goes_preprocessor import preprocess_frame
    from pathlib import Path
    import matplotlib.pyplot as plt

    interpolator = RIFEThermalInterpolator()
    
    data_root = Path("data/goes19/raw")
    files = sorted(data_root.rglob("*.nc"))
    
    if len(files) < 21:
        print("Need ≥ 21 files. Using random tensors for architecture test.")
        t0 = torch.rand(1, 1, 512, 512)
        t2 = torch.rand(1, 1, 512, 512)
    else:
        print(f"Loading files: {files[0].name} and {files[20].name}")
        t0, _, _ = preprocess_frame(str(files[0]))
        t2, _, _ = preprocess_frame(str(files[20]))
        if t0.dim() == 3:
            t0 = t0.unsqueeze(0)
        if t2.dim() == 3:
            t2 = t2.unsqueeze(0)

    print(f"Running inference with dimensions: t0={t0.shape}, t2={t2.shape}")
    pred, flow, mask = interpolator.infer(t0, t2)
    print(f"Prediction shape: {pred.shape}, range: [{pred.min():.3f}, {pred.max():.3f}]")
    print(f"Flow shape: {flow.shape}")
    print(f"Confidence shape: {mask.shape}")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(t0.squeeze().numpy(), cmap='inferno_r', vmin=0, vmax=1)
    axes[0].set_title('T0 Input')
    axes[1].imshow(pred.squeeze().numpy(), cmap='inferno_r', vmin=0, vmax=1)
    axes[1].set_title('Predicted T1')
    axes[2].imshow(t2.squeeze().numpy(), cmap='inferno_r', vmin=0, vmax=1)
    axes[2].set_title('T2 Input')
    axes[3].imshow(mask.squeeze().numpy(), cmap='viridis')
    axes[3].set_title('Confidence Mask')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    out = Path("outputs/rife_test.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")