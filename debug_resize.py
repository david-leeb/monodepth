import os
from PIL import Image

# CONFIG
SOURCE_DIR = "nuscenes_data"
TARGET_DIR = "nuscenes_320x192"

def main():
    print(f"Checking {SOURCE_DIR}...")
    
    count = 0
    # Walk through the source folders
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith('.jpg'):
                src_path = os.path.join(root, file)
                
                # Calculate new path
                rel_path = os.path.relpath(src_path, SOURCE_DIR)
                dest_path = os.path.join(TARGET_DIR, rel_path)
                
                print(f"\n[Image {count+1}]")
                print(f"  Source: {src_path}")
                print(f"  Target: {dest_path}")
                
                try:
                    # 1. Create Folder
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    
                    # 2. Open
                    img = Image.open(src_path)
                    print(f"  Opened successfully: {img.size}")
                    
                    # 3. Resize
                    img = img.resize((320, 192), Image.LANCZOS)
                    print(f"  Resized successfully.")
                    
                    # 4. Save
                    img.save(dest_path, quality=90)
                    print(f"  SAVED! Check this file: {dest_path}")
                    
                except Exception as e:
                    print(f"  CRITICAL FAILURE: {e}")
                
                count += 1
                if count >= 10:
                    print("\n--- Test Complete (Stopped after 10 images) ---")
                    return

    if count == 0:
        print("ERROR: No .jpg files found in nuscenes_data!")

if __name__ == "__main__":
    main()
