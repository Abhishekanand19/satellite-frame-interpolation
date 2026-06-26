import os
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Gracefully handle imports to see what exists
try:
    from src.data_loader.goes_dataset import discover_nc_files, build_triplets
except ImportError:
    # Fallback dummy logic if modules aren't matching perfectly yet
    def discover_nc_files(d): return [str(p) for p in Path(d).rglob("*.nc")]
    def build_triplets(files): return [(files[i], files[i+1], files[i+2]) for i in range(len(files)-2)] if len(files) >= 3 else []

def main():
    data_dir = os.path.join(ROOT, "data", "goes19")
    print(f"🔍 Scanning directory: {data_dir}")
    
    # Try finding any .nc files anywhere in data/goes19
    nc_files = [str(p) for p in Path(data_dir).rglob("*.nc")]
    print(f"📊 Found {len(nc_files)} NetCDF (.nc) files.")
    
    if len(nc_files) == 0:
        print("❌ ERROR: No files found! Let's check where they are.")
        print("Current existing items in data/goes19:")
        os.system(f"ls -R {data_dir}")
        return
        
    nc_files.sort()
    triplets = []
    if len(nc_files) >= 3:
        for i in range(len(nc_files) - 2):
            triplets.append((nc_files[i], nc_files[i+1], nc_files[i+2]))
            
    print(f"🎬 Generated {len(triplets)} interpolation triplets (t0, t1, t2).")
    
    n = len(triplets)
    tr, va = int(n * 0.8), int(n * 0.9)
    train_triplets = triplets[:tr]
    val_triplets = triplets[tr:va]
    test_triplets = triplets[va:]
    
    output_dir = os.path.join(ROOT, "data", "splits")
    os.makedirs(output_dir, exist_ok=True)
    
    for name, data in [("train.txt", train_triplets), ("val.txt", val_triplets), ("test.txt", test_triplets)]:
        with open(os.path.join(output_dir, name), "w") as f:
            for item in data:
                f.write(f"{item[0]},{item[1]},{item[2]}\n")
                
    stats = {
        "total_files": len(nc_files),
        "total_triplets": len(triplets),
        "train_count": len(train_triplets),
        "val_count": len(val_triplets),
        "test_count": len(test_triplets)
    }
    
    with open(os.path.join(output_dir, "dataset_statistics.json"), "w") as f:
        json.dump(stats, f, indent=4)
        
    print("✅ Phase 1 Deliverables saved to data/splits/")

if __name__ == "__main__":
    main()
