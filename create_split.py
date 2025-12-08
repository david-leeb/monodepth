import os
from nuscenes.nuscenes import NuScenes

# CONFIG
dataroot = "nuscenes_data" 
version = "v1.0-trainval"
split_name = "nuscenes_full"

print(f"Loading nuScenes {version} database... this takes time!")
nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

os.makedirs(f"splits/{split_name}", exist_ok=True)

train_lines = []
val_lines = []

# --- DEFINING THE CAMERAS (This was missing!) ---
cameras = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]

print(f"Processing {len(nusc.scene)} scenes...")

for i, scene in enumerate(nusc.scene):
    # 1. Split Strategy: Every 5th scene goes to Validation
    is_train = (i % 5 != 0)
    
    sample_token = scene['first_sample_token']
    while sample_token:
        sample = nusc.get('sample', sample_token)
        
        # 2. Loop through ALL 6 cameras
        for cam_name in cameras:
            # Check if this sample has data for this camera
            if cam_name in sample['data']:
                cam_token = sample['data'][cam_name]
                cam_data = nusc.get('sample_data', cam_token)
                
                # 3. Verify the file actually exists on disk (Blob 01 check)
                if os.path.exists(os.path.join(dataroot, cam_data['filename'])):
                    line = f"{cam_name} {sample_token} l\n"
                    
                    if is_train:
                        train_lines.append(line)
                    else:
                        val_lines.append(line)
        
        sample_token = sample['next']

# Save files
with open(f"splits/{split_name}/train_files.txt", "w") as f:
    f.writelines(train_lines)
with open(f"splits/{split_name}/val_files.txt", "w") as f:
    f.writelines(val_lines)

print(f"Done! {len(train_lines)} training frames, {len(val_lines)} validation frames.")
