import os
import torch
import numpy as np
from layers import disp_to_depth
from utils import readlines
import networks
from torch.utils.data import DataLoader
import datasets
import cv2
from tqdm import tqdm

# Hardcode options to simplify usage
DATA_ROOT = "kitti_data"      # Original data (needed for Clean baseline)
CORRUPT_ROOT = "corrupted_kitti" # Where generated images live
MODEL_PATH = "models/weights_19" # Your model path
SPLIT = "eigen"

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

def run_eval(data_path, encoder, depth_decoder, filenames, gt_depths):
    """Runs evaluation on a specific data path."""
    
    # Setup Dataloader pointing to the SPECIFIC corrupted folder
    # We assume images are saved as .png in generation
    dataset = datasets.KITTIRAWDataset(
        data_path, filenames, 192, 640, [0], 4, is_train=False, img_ext=".png")
    
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    pred_disps = []
    
    # Inference Loop
    with torch.no_grad():
        for data in dataloader:
            input_color = data[("color", 0, 0)].cuda()
            output = depth_decoder(encoder(input_color))
            pred_disp, _ = disp_to_depth(output[("disp", 0)], 1e-3, 80.0)
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)
    
    # Metric Loop
    errors = []
    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = cv2.resize(pred_disps[i], (gt_width, gt_height))
        pred_depth = 1 / pred_disp
        
        # Standard Eigen Crop
        mask = np.logical_and(gt_depth > 1e-3, gt_depth < 80.0)
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        # Median Scaling
        ratio = np.median(gt_depth) / np.median(pred_depth)
        pred_depth *= ratio
        pred_depth[pred_depth < 1e-3] = 1e-3
        pred_depth[pred_depth > 80.0] = 80.0

        errors.append(compute_errors(gt_depth, pred_depth))

    return np.array(errors).mean(0)

if __name__ == "__main__":
    # 1. Load Model Once
    device = torch.device("cuda")
    print(f"-> Loading model from {MODEL_PATH}")
    
    enc_path = os.path.join(MODEL_PATH, "encoder.pth")
    dec_path = os.path.join(MODEL_PATH, "depth.pth")
    
    encoder = networks.ResnetEncoder(18, False)
    encoder.load_state_dict(torch.load(enc_path))
    encoder.to(device).eval()
    
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
    depth_decoder.load_state_dict(torch.load(dec_path))
    depth_decoder.to(device).eval()

    # 2. Load Metadata
    filenames = readlines(os.path.join("splits", SPLIT, "test_files.txt"))
    gt_path = os.path.join("splits", SPLIT, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    # 3. Evaluate Baseline (Clean)
    print("-> Evaluating CLEAN Baseline...")
    base_res = run_eval(DATA_ROOT, encoder, depth_decoder, filenames, gt_depths)
