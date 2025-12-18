import os
import argparse
import csv
import torch
import numpy as np
from layers import disp_to_depth
from utils import readlines
import networks
from torch.utils.data import DataLoader
import datasets
import cv2
from tqdm import tqdm

# === CONFIGURATION ===
DATA_ROOT = "kitti_data"            # Path to Clean KITTI
# Path to your Corrupted KITTI folder
CORRUPT_ROOT = "/mnt/shared/home/b31xs2/b31xs/monodepth2/corrupted_kitti"
SPLIT = "eigen"
# =====================

def compute_errors(gt, pred):
    """Standard Metric Computation"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def run_eval(data_path, encoder, depth_decoder, filenames, gt_depths, desc="Eval"):
    """
    Runs evaluation on a specific data path.
    CRITICAL UPDATE: Filters out filenames that do not exist on disk 
    (handling the case where corrupted data has skipped frames).
    """
    
    # --- 1. Filter Filenames ---
    valid_filenames = []
    valid_gt_depths = []
    
    # We construct the expected path for every file in the test list
    # If it exists, we keep it. If not, we skip it.
    for i, line in enumerate(filenames):
        folder, frame_index, side = line.split()
        f_str = "{:010d}.png".format(int(frame_index))
        side_map = {"l": "image_02", "r": "image_03"}
        
        # Check if file exists
        full_path = os.path.join(data_path, folder, side_map[side], "data", f_str)
        
        if os.path.isfile(full_path):
            valid_filenames.append(line)
            valid_gt_depths.append(gt_depths[i])
            
    # If no files matched, warn the user and return None
    if len(valid_filenames) == 0:
        # Only print warning if we are deeper in the loop (avoid spamming for empty folders)
        if "Clean" not in desc:
            print(f"  [Warning] No matching files found in {data_path}. (Dataset mismatch?)")
        return None

    # --- 2. Create Dataset with Valid Files Only ---
    dataset = datasets.KITTIRAWDataset(
        data_path, valid_filenames, 192, 640, [0], 4, is_train=False, img_ext=".png")
    
    dataloader = DataLoader(dataset, 8, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    pred_disps = []
    
    # --- 3. Inference Loop ---
    with torch.no_grad():
        for data in tqdm(dataloader, desc=desc, leave=False):
            input_color = data[("color", 0, 0)].cuda()
            output = depth_decoder(encoder(input_color))
            pred_disp, _ = disp_to_depth(output[("disp", 0)], 1e-3, 80.0)
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disps.append(pred_disp)

    if not pred_disps:
        return None

    pred_disps = np.concatenate(pred_disps)
    errors = []
    
    # --- 4. Metric Computation ---
    for i in range(pred_disps.shape[0]):
        # CRITICAL: Use valid_gt_depths[i] instead of gt_depths[i] 
        # to ensure the Ground Truth matches the Filtered Image.
        gt_depth = valid_gt_depths[i]
        
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = cv2.resize(pred_disps[i], (gt_width, gt_height))
        pred_depth = 1 / pred_disp
        
        mask = np.logical_and(gt_depth > 1e-3, gt_depth < 80.0)
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        ratio = np.median(gt_depth) / np.median(pred_depth)
        pred_depth *= ratio
        pred_depth[pred_depth < 1e-3] = 1e-3
        pred_depth[pred_depth > 80.0] = 80.0

        errors.append(compute_errors(gt_depth, pred_depth))

    return np.array(errors).mean(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robustness Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to weights folder')
    parser.add_argument('--model_name', type=str, required=True, help='Name for CSV entry')
    parser.add_argument('--csv_path', type=str, default='robustness_results.csv', help='Output CSV file')
    args = parser.parse_args()

    # 1. Setup Model
    device = torch.device("cuda")
    encoder = networks.ResnetEncoder(18, False)
    
    # Safe Load for Encoder (Fixes "Unexpected key" error)
    loaded_dict = torch.load(os.path.join(args.model_path, "encoder.pth"), map_location=device)
    filtered_dict = {k: v for k, v in loaded_dict.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict)
    encoder.to(device).eval()

    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
    depth_decoder.load_state_dict(torch.load(os.path.join(args.model_path, "depth.pth")))
    depth_decoder.to(device).eval()

    # 2. Setup Data
    filenames = readlines(os.path.join("splits", SPLIT, "test_files.txt"))
    gt_path = os.path.join("splits", SPLIT, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    # 3. CSV Setup
    file_exists = os.path.isfile(args.csv_path)
    with open(args.csv_path, mode='a', newline='') as csv_file:
        fieldnames = ['Model', 'Corruption', 'Severity', 'AbsRel', 'SqRel', 'RMSE', 'RMSE_log', 'a1', 'a2', 'a3']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        # 4. Evaluate Clean
        print(f"[{args.model_name}] Evaluating Clean...")
        res = run_eval(DATA_ROOT, encoder, depth_decoder, filenames, gt_depths, desc="Clean")
        if res is not None:
            writer.writerow({
                'Model': args.model_name, 'Corruption': 'Clean', 'Severity': 0,
                'AbsRel': res[0], 'SqRel': res[1], 'RMSE': res[2], 'RMSE_log': res[3], 'a1': res[4], 'a2': res[5], 'a3': res[6]
            })

        # 5. Evaluate Corruptions
        if os.path.exists(CORRUPT_ROOT):
            # Get all folders inside corrupted_kitti
            all_folders = sorted([d for d in os.listdir(CORRUPT_ROOT) if os.path.isdir(os.path.join(CORRUPT_ROOT, d))])
            
            for folder_name in all_folders:
                # Expecting format: "brightness_1", "contrast_5" etc.
                parts = folder_name.rsplit('_', 1) 
                
                if len(parts) == 2 and parts[1].isdigit():
                    corruption_type = parts[0]
                    severity = int(parts[1])
                    curr_path = os.path.join(CORRUPT_ROOT, folder_name)
                    
                    print(f"[{args.model_name}] Evaluating {corruption_type} (Sev {severity})...")
                    res = run_eval(curr_path, encoder, depth_decoder, filenames, gt_depths, desc=folder_name)
                    
                    if res is not None:
                        writer.writerow({
                            'Model': args.model_name, 'Corruption': corruption_type, 'Severity': severity,
                            'AbsRel': res[0], 'SqRel': res[1], 'RMSE': res[2], 'RMSE_log': res[3], 'a1': res[4], 'a2': res[5], 'a3': res[6]
                        })
                    else:
                         print(f"  -> Skipping {folder_name}: No matching frames found.")
                else:
                    pass
        else:
            print(f"Error: {CORRUPT_ROOT} does not exist.")

    print(f"Results appended to {args.csv_path}")
