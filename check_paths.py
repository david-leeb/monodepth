# Save this as check_paths.py and run it: python check_paths.py
import os
from utils import readlines

# CONFIG
root = "/mnt/shared/home/b31xs2/b31xs/monodepth2/corrupted_kitti/brightness_1"  # Update this to your full path if needed
test_file_line = "2011_09_26/2011_09_26_drive_0002_sync 0000000006 l"

# LOGIC
folder, frame_index, side = test_file_line.split()
side_map = {"l": "image_02", "r": "image_03"}
f_str = "{:010d}.png".format(int(frame_index)) # Check PNG first

full_path = os.path.join(root, folder, side_map[side], "data", f_str)

print(f"Looking for: {full_path}")
if os.path.isfile(full_path):
    print("FOUND!")
else:
    print("MISSING")
    # Check JPG
    jpg_path = full_path.replace(".png", ".jpg")
    if os.path.isfile(jpg_path):
        print(f"Found as JPG instead: {jpg_path}")
