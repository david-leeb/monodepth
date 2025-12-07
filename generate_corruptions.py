import os
import numpy as np
import argparse
from tqdm import tqdm
import PIL.Image as pil
from imagecorruptions import corrupt
from utils import readlines

def generate(opt):
    # 1. Define Corruptions to Generate
    # You can add more: 'gaussian_noise', 'snow', 'pixelate', 'zoom_blur', etc.
    corruption_types = ["fog", "motion_blur", "defocus_blur", "brightness", "contrast"]
    severities = [1, 3, 5]  # Low, Medium, High

    # 2. Get List of Test Files
    # We use the standard Eigen split file list
    split_path = os.path.join(os.path.dirname(__file__), "splits", "eigen", "test_files.txt")
    filenames = readlines(split_path)

    print(f"-> Generating corruptions for {len(filenames)} images...")

    for c_name in corruption_types:
        for sev in severities:
            print(f"\nProcessing: {c_name} (Severity {sev})")
            
            # Create a specific root folder for this corruption combo
            # e.g., "corrupted_kitti/fog_3/"
            out_root = os.path.join(opt.output_path, f"{c_name}_{sev}")
            
            for line in tqdm(filenames):
                # Parse KITTI path
                line_parts = line.split()
                folder = line_parts[0]
                frame_index = int(line_parts[1])
                side = line_parts[2] if len(line_parts) > 2 else "l"
                
                # Construct Input Path
                side_map = {"l": "image_02", "r": "image_03"}
                img_subpath = os.path.join(folder, side_map.get(side, "image_02"), "data", "{:010d}.png".format(frame_index))
                
                # Handle jpg vs png
                full_in_path = os.path.join(opt.data_path, img_subpath)
                if not os.path.exists(full_in_path):
                    full_in_path = full_in_path.replace(".png", ".jpg")
                
                # 3. Load & Corrupt
                image = pil.open(full_in_path).convert('RGB')
                image_np = np.array(image)
                
                # Apply corruption
                corrupted_np = corrupt(image_np, corruption_name=c_name, severity=sev)
                corrupted_pil = pil.fromarray(corrupted_np)

                # 4. Save (Mirroring Directory Structure)
                full_out_path = os.path.join(out_root, img_subpath)
                
                # Create parent directories if they don't exist
                os.makedirs(os.path.dirname(full_out_path), exist_ok=True)
                
                # Save as PNG to avoid compression artifacts messing up results further
                corrupted_pil.save(full_out_path.replace(".jpg", ".png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Corrupted KITTI Test Set")
    parser.add_argument("--data_path", type=str, default="kitti_data", help="Path to original KITTI data")
    parser.add_argument("--output_path", type=str, default="corrupted_kitti", help="Where to save generated images")
    args = parser.parse_args()
    
    generate(args)
