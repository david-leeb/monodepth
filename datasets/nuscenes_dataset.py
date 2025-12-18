import os
import random
import numpy as np
import torch
import PIL.Image as pil
from torchvision import transforms
from nuscenes.nuscenes import NuScenes
from .mono_dataset import MonoDataset

# Global cache to prevent reloading NuScenes database on every worker
_NUSC_CACHE = {} 

class NuScenesDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(NuScenesDataset, self).__init__(*args, **kwargs)
        
        version = "v1.0-mini" if "mini" in self.data_path else "v1.0-trainval"
        
        # CHECK CACHE BEFORE LOADING
        global _NUSC_CACHE
        if version not in _NUSC_CACHE:
            print("Loading NuScenes {} into cache...".format(version))
            _NUSC_CACHE[version] = NuScenes(version=version, dataroot=self.data_path, verbose=False)
        
        self.nusc = _NUSC_CACHE[version]
        
        self.full_res_shape = (1600, 900) # (Width, Height)


    def get_image_path(self, cam_name, sample_token):
        sample = self.nusc.get('sample', sample_token)
        cam_data = self.nusc.get('sample_data', sample['data'][cam_name])
        return os.path.join(self.data_path, cam_data['filename'])

    def get_intrinsic(self, cam_name, sample_token):
        # 1. Get original K (3x3)
        sample = self.nusc.get('sample', sample_token)
        cam_data = self.nusc.get('sample_data', sample['data'][cam_name])
        cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        
        # 2. Initialize 4x4 Identity Matrix (Required by Monodepth2)
        #    [ f 0 c 0 ]
        #    [ 0 f c 0 ]
        #    [ 0 0 1 0 ]
        #    [ 0 0 0 1 ]
        K = np.eye(4, dtype=np.float32)
        
        # 3. Fill in the 3x3 Intrinsics
        K[:3, :3] = np.array(cs_record['camera_intrinsic'], dtype=np.float32)
        
        # 4. Scale K to match target resolution
        orig_w = cam_data['width']
        orig_h = cam_data['height']
        scale_x = self.width / orig_w
        scale_y = self.height / orig_h
        
        K[0, :] *= scale_x
        K[1, :] *= scale_y
        
        return K



    def get_depth(self, folder, frame_index, side, do_flip):
        # 1. Get tokens for the Camera and the closest Lidar scan
        sample = self.nusc.get('sample', frame_index)
        cam_token = sample['data'][folder]
        lidar_token = sample['data']['LIDAR_TOP']

        # 2. Project Lidar to Camera using the standard SDK function
        # This returns the 3D points in the camera's image coordinates
        points, _, _ = self.nusc.explorer.map_pointcloud_to_image(
            pointsensor_token=lidar_token,
            camera_token=cam_token,
            render_intensity=False,
            show_lidarseg=False,
            filter_lidarseg_labels=None,
            lidarseg_preds_bin_path=None,
            show_panoptic=False
        )

        # 3. Create the blank depth map
        # NuScenes images are typically 1600x900
        cam_record = self.nusc.get('sample_data', cam_token)
        orig_h, orig_w = cam_record['height'], cam_record['width']
        
        depth_gt = np.zeros((orig_h, orig_w), dtype=np.float32)
        
        # 4. Fill the depth map
        # points[0] = x, points[1] = y, points[2] = depth
        pts_x = points[0, :].astype(np.int32)
        pts_y = points[1, :].astype(np.int32)
        pts_d = points[2, :]

        # Filter out points that fall outside the image bounds
        valid_mask = (pts_x >= 0) & (pts_x < orig_w) & (pts_y >= 0) & (pts_y < orig_h)
        
        depth_gt[pts_y[valid_mask], pts_x[valid_mask]] = pts_d[valid_mask]

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


    def check_depth(self):
        return True

    def __getitem__(self, index):
        inputs = {}
        
        line = self.filenames[index].split()
        folder = line[0]        # Camera Name
        frame_index = line[1]   # Sample Token
        side = None

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        for i in self.frame_idxs:
            if i == 0:
                target_token = frame_index
            else:
                curr_sample = self.nusc.get('sample', frame_index)
                curr_cam_data = self.nusc.get('sample_data', curr_sample['data'][folder])
                if i == -1:
                    target_token = curr_cam_data['prev']
                elif i == 1:
                    target_token = curr_cam_data['next']
                if target_token == "":
                    target_token = frame_index

            if target_token == frame_index:
                 sd_record_token = self.nusc.get('sample', frame_index)['data'][folder]
                 sd_record = self.nusc.get('sample_data', sd_record_token)
            else:
                 sd_record = self.nusc.get('sample_data', target_token)
                 
            img_path = os.path.join(self.data_path, sd_record['filename'])
            img = pil.open(img_path).convert('RGB')

            if do_flip:
                img = img.transpose(pil.FLIP_LEFT_RIGHT)
            
            img = img.resize((self.width, self.height), pil.LANCZOS)
            inputs[("color", i, -1)] = img

        K = self.get_intrinsic(folder, frame_index)
        if do_flip:
            K[0, 2] = self.width - 1 - K[0, 2]
        inv_K = np.linalg.pinv(K)
        
        for scale in range(self.num_scales):
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        for scale in range(self.num_scales):
            for i in self.frame_idxs:
                inputs[("color", i, scale)] = self.resize[scale](inputs[("color", i, -1)])

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        for scale in range(self.num_scales):
            for i in self.frame_idxs:
                raw_img = inputs[("color", i, scale)]
                inputs[("color", i, scale)] = self.to_tensor(raw_img)
                inputs[("color_aug", i, scale)] = self.to_tensor(color_aug(raw_img))

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]

        return inputs
