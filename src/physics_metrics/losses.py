import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLoss(nn.Module):
    def __init__(self, l1_weight=1.0, thermal_weight=0.5):
        super(PhysicsInformedLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.thermal_weight = thermal_weight
        self.l1_weight = l1_weight

    def forward(self, pred_frame, gt_frame):
        # 1. Pixel-level spatial reconstruction loss
        base_loss = self.l1(pred_frame, gt_frame)
        
        # 2. Meteorological Constraint: Spatial gradients (fluid movement stabilization)
        # Ensures cloud movement boundary transitions are smooth rather than blurry
        pred_grad_x = torch.abs(pred_frame[:, :, :, :-1] - pred_frame[:, :, :, 1:])
        gt_grad_x = torch.abs(gt_frame[:, :, :, :-1] - gt_frame[:, :, :, 1:])
        thermal_gradient_loss = F.mse_loss(pred_grad_x, gt_grad_x)
        
        total_loss = (self.l1_weight * base_loss) + (self.thermal_weight * thermal_gradient_loss)
        return total_loss, base_loss, thermal_gradient_loss
