import s3fs
import os
import re
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
YEAR = "2026"
DAYS = ["001", "002", "003"]
TARGET_MINUTES = [0, 10, 20, 30, 40, 50]
BASE_DIR = Path("data/goes19/raw/")
LOG_FILE = "download_log.txt"

# ==========================================
# LOGGING SETUP
# ==========================================
def log_to_file(message):
    """Appends a message to the download log file with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_closest_files(file_list):
    """
    Finds the closest M6C13 files to the target minutes.
    Prevents downloading all 120 files in an hour.
    """
    target_seconds = {m: m * 60 for m in TARGET_MINUTES}
    selected_files = set() # Using a set to prevent accidental duplicate downloads
    
    # Filter only Thermal IR (M6C13)
    ch13_files = [f for f in file_list if "M6C13" in f]
    
    for target_m, t_sec in target_seconds.items():
        best_file = None
        min_diff = float('inf')
        
        for file in ch13_files:
            # Extract the Start Time: _sYYYYDDDHHMMSS
            # \d{9} matches YYYY(4) + DDD(3) + HH(2)
            # (\d{4}) captures MMSS
            match = re.search(r'_s\d{9}(\d{4})', file)
            if match:
                mmss = match.group(1)
                file_sec = int(mmss[:2]) * 60 + int(mmss[2:])
                diff = abs(file_sec - t_sec)
                
                if diff < min_diff:
                    min_diff = diff
                    best_file = file
        
        if best_file:
            selected_files.add(best_file)
            
    return list(selected_files)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # Initialize connection to anonymous public bucket
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={'region_name': 'us-east-1'})
    
    # Statistics Tracking
    stats = {
        "total_downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "days": {day: 0 for day in DAYS}
    }
    
    log_to_file("=== NEW DOWNLOAD SESSION STARTED ===")
    
    for day in DAYS:
        day_dir = BASE_DIR / f"day{day}"
        day_dir.mkdir(parents=True, exist_ok=True)
        
        # We expect exactly 144 files per day (24 hours * 6 files)
        pbar = tqdm(total=144, desc=f"Day {day}", unit="file", leave=True)
        
        for hour in range(24):
            hour_str = f"{hour:02d}"
            bucket_path = f"noaa-goes19/ABI-L2-CMIPM/{YEAR}/{day}/{hour_str}/"
            
            try:
                # Get all files in the bucket for this hour
                all_files = fs.ls(bucket_path)
                target_files = get_closest_files(all_files)
                
                for remote_file in target_files:
                    filename = os.path.basename(remote_file)
                    local_path = day_dir / filename
                    
                    # Extract minute for logging
                    match = re.search(r'_s\d{9}(\d{2})', filename)
                    minute = match.group(1) if match else "XX"
                    
                    if local_path.exists():
                        stats["skipped"] += 1
                        log_to_file(f"SKIPPED (Exists): {filename}")
                        pbar.update(1)
                    else:
                        try:
                            # Execute Download
                            fs.get(remote_file, str(local_path))
                            stats["total_downloaded"] += 1
                            stats["days"][day] += 1
                            log_to_file(f"DOWNLOADED: {filename}")
                            pbar.update(1)
                        except Exception as e:
                            stats["failed"] += 1
                            err_msg = f"FAILED: Day {day} Hour {hour_str} Minute {minute}"
                            tqdm.write(err_msg) # Write above progress bar
                            log_to_file(f"{err_msg} | Error: {e}")
                            pbar.update(1) # Update bar even on failure so it completes
                            
                tqdm.write(f"Day {day} Hour {hour_str} completed.")
                
            except FileNotFoundError:
                tqdm.write(f"FAILED: Day {day} Hour {hour_str} (Directory missing on AWS)")
                log_to_file(f"FAILED: AWS Directory missing for Day {day} Hour {hour_str}")
                pbar.update(6) # Skip 6 files for this hour
            except Exception as e:
                tqdm.write(f"FAILED: Day {day} Hour {hour_str} (Connection Error)")
                log_to_file(f"FAILED: Connection error for Day {day} Hour {hour_str} | {e}")
                pbar.update(6)
                
        pbar.close()
        tqdm.write(f"Day {day} successfully downloaded.\n{stats['days'][day]} files downloaded.")
        log_to_file(f"--- Day {day} Complete ---")

    # ==========================================
    # FINAL REPORT
    # ==========================================
    report = f"""
===================================
DOWNLOAD COMPLETE
Total files downloaded: {stats['total_downloaded']}
Downloaded:
Day001: {stats['days']['001']} files
Day002: {stats['days']['002']} files
Day003: {stats['days']['003']} files
Skipped existing files: {stats['skipped']}
Failed downloads: {stats['failed']}
===================================
"""
    print(report)
    log_to_file(report.strip())

if __name__ == "__main__":
    main()
    