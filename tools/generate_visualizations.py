import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Create visual asset dump directory
OUTPUT_DIR = os.path.join(ROOT, "data", "visuals")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_assets():
    print("🎨 Initializing Day 2 Spatial Presentation Utilities...")
    
    # 1. Simulate a standard satellite image canvas grid (256x256)
    np.random.seed(42)
    x = np.linspace(-3, 3, 256)
    y = np.linspace(-3, 3, 256)
    X, Y = np.meshgrid(x, y)
    
    # Mocking standard cloud structural patterns (Thermal Infrared spectrum representation)
    gt_cloud = np.exp(-X**2 - Y**2) 
    pred_cloud = np.exp(-(X-0.08)**2 - (Y+0.05)**2) # Model prediction with tiny offset
    
    # --- ASSET 1: Absolute Error Heatmap ---
    error_heatmap = np.abs(gt_cloud - pred_cloud)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(error_heatmap, cmap='inferno')
    plt.colorbar(label='Thermal Deviation Delta (Kelvin Residuals)')
    plt.title('Absolute Spatial Error Map (Physics Residue Analysis)')
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, "residual_heatmap.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print("📸 Asset 1 Generated: data/visuals/residual_heatmap.png")
    
    # --- ASSET 2: Cloud Motion Vector Fields (Quiver Arrow Overlay) ---
    plt.figure(figsize=(6, 6))
    plt.imshow(pred_cloud, cmap='bone')
    
    # Generate sub-sampled arrow points so grid isn't overcrowded
    skip = 16
    Y_sub, X_sub = np.mgrid[0:256:skip, 0:256:skip]
    U = np.ones_like(X_sub) * 2.5   # X velocity component (Fixed function call)
    V = np.ones_like(Y_sub) * -1.2  # Y velocity component (Fixed function call)
    
    plt.quiver(X_sub, Y_sub, U, V, color='cyan', scale=40, width=0.005, label='Cloud Motion Vector')
    plt.title('Dynamic Convective Cloud Motion Vectors (USP Layer)')
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, "motion_vectors.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print("📸 Asset 2 Generated: data/visuals/motion_vectors.png")
    
    print("\n✅ Visualization Engine Complete. Presentation assets ready for slide generation.")

if __name__ == "__main__":
    generate_assets()
