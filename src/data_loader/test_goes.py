# src/data_loader/test_goes.py
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys

def explore_goes_file(nc_path: str):
    print(f"\n{'='*60}")
    print(f"FILE: {nc_path}")
    print(f"{'='*60}")
    
    ds = xr.open_dataset(nc_path, engine='netcdf4')
    
    print("\n--- DIMENSIONS ---")
    for dim, size in ds.dims.items():
        print(f"  {dim}: {size}")
    
    print("\n--- VARIABLES ---")
    for var in ds.data_vars:
        v = ds[var]
        print(f"  {var}: shape={v.shape}, dtype={v.dtype}, units={v.attrs.get('units','N/A')}")
    
    print("\n--- GLOBAL ATTRS ---")
    for k, v in list(ds.attrs.items())[:10]:
        print(f"  {k}: {v}")
    
    # Extract brightness temperature
    bt = None
    for candidate in ['CMI', 'Rad', 'BT', 'brightness_temperature']:
        if candidate in ds.data_vars:
            bt = ds[candidate].values.astype(np.float32)
            print(f"\nUsing variable: {candidate}")
            break
    
    if bt is None:
        var_name = list(ds.data_vars)[0]
        bt = ds[var_name].values.astype(np.float32)
        print(f"\nFallback to first variable: {var_name}")
    
    print(f"BT shape: {bt.shape}")
    print(f"BT min: {np.nanmin(bt):.2f}, max: {np.nanmax(bt):.2f}, mean: {np.nanmean(bt):.2f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Raw BT
    im0 = axes[0].imshow(bt, cmap='gray_r', interpolation='nearest')
    axes[0].set_title('Brightness Temperature (Raw)', fontsize=13)
    plt.colorbar(im0, ax=axes[0], label='BT (K)')
    
    # Thermal colormap
    bt_clipped = np.clip(bt, np.nanpercentile(bt, 2), np.nanpercentile(bt, 98))
    im1 = axes[1].imshow(bt_clipped, cmap='inferno_r', interpolation='nearest')
    axes[1].set_title('Brightness Temperature (Enhanced)', fontsize=13)
    plt.colorbar(im1, ax=axes[1], label='BT (K)')
    
    plt.suptitle(f"GOES-19 ABI L2 CMIP M6C13\n{Path(nc_path).name}", fontsize=11)
    plt.tight_layout()
    
    out_path = Path("outputs/test_goes_preview.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved preview → {out_path}")
    plt.close()
    
    ds.close()
    return bt

if __name__ == "__main__":
    if len(sys.argv) > 1:
        nc_file = sys.argv[1]
    else:
        # Auto-find first .nc file
        search_dirs = [
            "data/goes19/raw/day001",
            "data/goes19/raw/day002",
            "data/goes19/raw/day003",
        ]
        nc_file = None
        for d in search_dirs:
            files = sorted(Path(d).glob("*.nc"))
            if files:
                nc_file = str(files[0])
                break
        
        if nc_file is None:
            print("ERROR: No .nc files found. Pass path as argument.")
            sys.exit(1)
    
    print(f"Loading: {nc_file}")
    bt = explore_goes_file(nc_file)
    print("\nDone.")