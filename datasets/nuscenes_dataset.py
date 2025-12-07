import os
import random
import numpy as np
import torch
import PIL.Image as pil
from nuscenes.nuscenes import NuScenes
from .mono_dataset import MonoDataset

class NuScenesDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(NuScenesDataset, self).__init__(*args, **kwargs)
        
        # Load nuScenes database
        version = "v1.0-mini" if "mini" in self.data_path else "v1.0-trainval"
        self.nusc = NuScenes(version=version, dataroot=self.data_path, verbose=False)

    def get_image_path(self, cam_name, sample_token):
        sample = self.nusc.get('sample', sample_token)
        cam_data = self.nusc.get('sample_data', sample['data'][cam_name])
        return os.path.join(self.data_path, cam_data['filename'])

    def get_intrinsic(self, cam_name, sample_token):
        # 1. Get original K
        sample = self.nusc.get('sample', sample_token)
        cam_data = self.nusc.get('sample_data', sample['data'][cam_name])
        cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        K = np.array(cs_record['camera_intrinsic'], dtype=np.float32)
        
        # 2. Scale K to match target resolution
        orig_w = cam_data['width']
        orig_h = cam_data['height']
        scale_x = self.width / orig_w
        scale_y = self.height / orig_h
        
        K[0, :] *= scale_x
        K[1, :] *= scale_y
        return K

    def __getitem__(self, index):
        """Override standard __getitem__ to handle string tokens and variable intrinsics"""
        inputs = {}
        
        # 1. Parse line (Camera, Token)
        line = self.filenames[index].split()
        folder = line[0]        # Camera Name (e.g. CAM_FRONT)
        frame_index = line[1]   # Sample Token (String)
        side = None

        # Augmentation toggles
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        # 2. Temporal Loop (Load 0, -1, 1)
        for i in self.opt.frame_ids:
            # Find the token for the requested frame (0, -1, or 1)
            if i == 0:
                target_token = frame_index
            else:
                # Use nuScenes graph to find prev/next
                curr_sample = self.nusc.get('sample', frame_index)
                curr_cam_data = self.nusc.get('sample_data', curr_sample['data'][folder])
                
                if i == -1:
                    target_token = curr_cam_data['prev'] # Previous frame token
                elif i == 1:
                    target_token = curr_cam_data['next'] # Next frame token
                
                # Handle missing frames (start/end of scene) by repeating the center frame
                if target_token == "":
                    target_token = frame_index

            # Load Image
            # Note: We cannot use self.get_color here because we need to bypass MonoDataset's logic
            # We implement the loading directly:
            
            # Resolve the sample token from the sensor token if needed
            # (NuScenes 'prev'/'next' point to sample_data tokens, not sample tokens. 
            #  We need to be careful. For simplicity, we stick to sample_data tokens here.)
            
            # Actually, standard nuscenes 'prev' gives a sample_data token.
            # We can get the filename directly from that.
            sd_record = self.nusc.get('sample_data', target_token) if target_token != frame_index else self.nusc.get('sample_data', self.nusc.get('sample', frame_index)['data'][folder])
            
            img_path = os.path.join(self.data_path, sd_record['filename'])
            img = pil.open(img_path).convert('RGB')

            # Resize/Crop/Flip (Standard MonoDataset logic replicated)
            if do_flip:
                img = img.transpose(pil.Image.FLIP_LEFT_RIGHT)
            
            img = img.resize((self.width, self.height), pil.Image.ANTIALIAS)
            
            # Convert to Tensor
            inputs[("color", i, -1)] = self.to_tensor(img)

        # 3. Handle Intrinsics (K)
        # We only need K for the center frame (0) usually, or for all if using advanced losses.
        # Monodepth2 mainly uses K for frame 0.
        K = self.get_intrinsic(folder, frame_index)
        if do_flip:
            K[0, 2] = self.width - 1 - K[0, 2]
        
        inv_K = np.linalg.pinv(K)
        
        # Add to inputs for all scales
        for scale in self.opt.scales:
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        # 4. Multi-scale color adjustment (Standard)
        for scale in self.opt.scales:
            for i in self.opt.frame_ids:
                inputs[("color", i, scale)] = self.resize[scale](inputs[("color", i, -1)])

        return inputs
