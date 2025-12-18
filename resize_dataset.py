import os
import time
import shutil
from PIL import Image
from multiprocessing import Pool, cpu_count

# --- CONFIGURATION ---
SOURCE_DIR = "nuscenes_data"       # Where your massive images are now
TARGET_DIR = "nuscenes_320x192"    # Where the tiny fast images will go
TARGET_W = 320
TARGET_H = 192
# ---------------------

def resize_file(file_info):
    """Worker function to resize a single file"""
    src_path, dest_path = file_info
    
    # Skip if already exists (good for resuming interrupted runs)
    if os.path.exists(dest_path):
        return

    try:
        # Open and Resize
        # Using LANCZOS for best quality when downscaling
        img = Image.open(src_path)
        img = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        
        # Save (JPEG quality 90 is a good balance)
        img.save(dest_path, quality=90)
    except Exception as e:
        print(f"Error resizing {src_path}: {e}")

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Could not find source directory '{SOURCE_DIR}'")
        return

    print(f"--- Starting Resize Job ---")
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}")
    print(f"Resolution: {TARGET_W}x{TARGET_H}")
    
    # 1. Collect all image files
    print("Scanning files (this might take a minute)...")
    tasks = []
    
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(root, file)
                
                # Determine destination path
                # Replaces 'nuscenes_data' with 'nuscenes_320x192' in the path
                rel_path = os.path.relpath(src_path, SOURCE_DIR)
                dest_path = os.path.join(TARGET_DIR, rel_path)
                
                # Create subfolder in target if it doesn't exist
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
                tasks.append((src_path, dest_path))

    total_files = len(tasks)
    print(f"Found {total_files} images to resize.")

    # 2. Copy Metadata (Crucial for nuScenes database!)
    print("Copying metadata (json files)...")
    for item in os.listdir(SOURCE_DIR):
        src_item = os.path.join(SOURCE_DIR, item)
        # We look for folders like v1.0-trainval or v1.0-mini
        if os.path.isdir(src_item) and item.startswith("v1.0"):
            dest_item = os.path.join(TARGET_DIR, item)
            if not os.path.exists(dest_item):
                print(f"Copying {item}...")
                shutil.copytree(src_item, dest_item)
            else:
                print(f"Metadata {item} already exists, skipping copy.")
        # Also copy 'maps' if it exists
        elif os.path.isdir(src_item) and item == "maps":
             dest_item = os.path.join(TARGET_DIR, item)
             if not os.path.exists(dest_item):
                 print(f"Copying {item}...")
                 shutil.copytree(src_item, dest_item)

    # 3. Run Multiprocessing Resize
    # We use roughly 80% of available cores to avoid freezing the system
    num_processes = max(1, int(cpu_count() * 0.8))
    print(f"Spawning {num_processes} workers...")
    
    start_time = time.time()
    
    with Pool(num_processes) as p:
        # p.imap_unordered is faster than map for this
        for i, _ in enumerate(p.imap_unordered(resize_file, tasks), 1):
            if i % 2000 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (total_files - i) / rate
                print(f"Processed {i}/{total_files} ({i/total_files:.1%}) - {rate:.1f} img/sec - ETA: {remaining/60:.1f} min")

    print("Done! Dataset resize complete.")

if __name__ == "__main__":
    main()
