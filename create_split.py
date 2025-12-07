import os
from nuscenes.nuscenes import NuScenes

# CONFIG
dataroot = "nuscenes_data" 
version = "v1.0-trainval"  # <--- CHANGED FROM MINI
split_name = "nuscenes_full"

print(f"Loading nuScenes {version} database... this takes time!")
nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

os.makedirs(f"splits/{split_name}", exist_ok=True)

train_lines = []
val_lines = []

# NuScenes officially separates scenes into 'train' and 'val'
# We should respect that to avoid data leakage
total_scenes = len(nusc.scene)
print(f"Processing {total_scenes} scenes...")

for i, scene in enumerate(nusc.scene):
    # Determine if this scene is in the official train or val split
    # (Checking against official lists is safer, but for now we can infer 
    # based on scene name or just split by index if you want a custom split.
    # NuScenes doesn't explicitly flag 'train' vs 'val' in the scene dict easily 
    # without looking up the split spec. 
    # SIMPLE HACK: Use the first 80% for train, 20% for val)
    
    is_train = i < (total_scenes * 0.8)
    
    sample_token = scene['first_sample_token']
    while sample_token:
        sample = nusc.get('sample', sample_token)
        
        # Check if we actually have the image file (in case you didn't download all blobs)
        cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        if os.path.exists(os.path.join(dataroot, cam_data['filename'])):
            line = f"CAM_FRONT {sample_token} l\n"
            if is_train:
                train_lines.append(line)
            else:
                val_lines.append(line)
        
        sample_token = sample['next']

with open(f"splits/{split_name}/train_files.txt", "w") as f:
    f.writelines(train_lines)
with open(f"splits/{split_name}/val_files.txt", "w") as f:
    f.writelines(val_lines)

print(f"Done! {len(train_lines)} training frames, {len(val_lines)} validation frames.")
