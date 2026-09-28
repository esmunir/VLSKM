import os
import glob
import csv
import math
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

def load_image_to_tensor(img_path, size=(504, 504)):
    """Loads and normalizes an image for the neural network."""
    img = Image.open(img_path).convert('RGB').resize(size, Image.Resampling.LANCZOS)
    tensor = TF.to_tensor(img)
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalizer(tensor)

class TaipeiCrossViewDataset(Dataset):
    def __init__(self, dataset_dir, split="train"):
        """
        Args:
            dataset_dir: Path to the Taipei dataset root
            split: "train" or "test"
        """
        self.split_dir = os.path.join(dataset_dir, f"pairing_retrieval-google_uav-100_150_200-sat-27982_56102_1052", split)
        self.metadata_dir = os.path.join(dataset_dir, "metadata")
        
        # Depending on train/test, the folder names change slightly in your dataset
        if split == "train":
            self.sat_dir = os.path.join(self.split_dir, "satellite")
            self.uav_dir = os.path.join(self.split_dir, "drone")
        else:
            self.sat_dir = os.path.join(self.split_dir, "gallery_satellite")
            self.uav_dir = os.path.join(self.split_dir, "query_drone")
            
        # 1. Load the CSVs
        self.sat_metadata, self.uav_metadata = self._load_csvs()
        
        # 2. Find valid Class IDs
        self.class_ids = [d for d in os.listdir(self.uav_dir) if os.path.isdir(os.path.join(self.uav_dir, d))]
        
        # 3. Create combinations
        self.valid_pairs = []
        for class_id in self.class_ids:
            uav_images = glob.glob(os.path.join(self.uav_dir, class_id, "*.png"))
            sat_images = glob.glob(os.path.join(self.sat_dir, class_id, "*.png"))
            
            for uav_img in uav_images:
                for sat_img in sat_images:
                    self.valid_pairs.append({
                        "uav_path": uav_img,
                        "sat_path": sat_img,
                        "class_id": class_id
                    })
        
        print(f"✅ Created Dataset for {split}: {len(self.valid_pairs)} positive pairs ready for training.")

    def _load_csvs(self):
        sat_dict = {}
        uav_dict = {}
        
        sat_csv = os.path.join(self.metadata_dir, "sat_data.csv")
        if os.path.exists(sat_csv):
            with open(sat_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sat_dict[os.path.basename(row['filename'])] = {'lat': float(row['center_lat']), 'lon': float(row['center_lon'])}
        
        # Load all UAV CSVs (train0, test0, test1, etc.)
        uav_csvs = glob.glob(os.path.join(self.metadata_dir, "uav_*.csv"))
        for csv_file in uav_csvs:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    uav_dict[os.path.basename(row['filename'])] = {
                        'lat': float(row['center_lat']), 'lon': float(row['center_lon']),
                        'yaw': float(row['rotation_degrees']), 'height': float(row['height'])
                    }
        return sat_dict, uav_dict

    def _calculate_ground_truth_homography(self, uav_meta, sat_meta):
        """ Calculates the 3x3 Ground Truth Affine Transformation Matrix """
        # 1. Scale factor (Drone Coverage / Satellite Coverage)
        # Satellite is always 250m. Drone coverage = Drone height.
        scale = uav_meta['height'] / 250.0
        
        # 2. Translation offset (Meters to Pixels)
        R_earth = 6378137.0
        d_north_m = (uav_meta['lat'] - sat_meta['lat']) * (math.pi / 180.0) * R_earth
        d_east_m = (uav_meta['lon'] - sat_meta['lon']) * (math.pi / 180.0) * R_earth * math.cos(sat_meta['lat'] * math.pi / 180.0)
        
        # GSD of Satellite after resizing to 504x504
        sat_gsd = 250.0 / 504.0 
        
        # Convert meter offset to pixel offset on the 504x504 grid
        sat_offset_x = d_east_m / sat_gsd
        sat_offset_y = -d_north_m / sat_gsd # +Y is South in images
        
        # 3. Rotation (CCW Yaw to Radians)
        theta = uav_meta['yaw'] * (math.pi / 180.0)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        # 4. Construct the 3x3 Affine Matrix H
        # This maps a pixel [xd, yd, 1] in the Drone image to [xs, ys, 1] in the Sat image.
        H = np.zeros((3, 3), dtype=np.float32)
        
        # Rotation and Scale parameters
        H[0, 0] = scale * cos_t
        H[0, 1] = scale * sin_t
        H[1, 0] = -scale * sin_t
        H[1, 1] = scale * cos_t
        
        # Translation parameters (accounting for the 252,252 center origin)
        H[0, 2] = -252.0 * scale * cos_t - 252.0 * scale * sin_t + 252.0 + sat_offset_x
        H[1, 2] = 252.0 * scale * sin_t - 252.0 * scale * cos_t + 252.0 + sat_offset_y
        
        # Homogeneous row
        H[2, 2] = 1.0 
        
        return H

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        
        uav_tensor = load_image_to_tensor(pair["uav_path"])
        sat_tensor = load_image_to_tensor(pair["sat_path"])
        
        uav_name = os.path.basename(pair["uav_path"])
        sat_name = os.path.basename(pair["sat_path"])
        
        uav_meta = self.uav_metadata[uav_name]
        sat_meta = self.sat_metadata[sat_name]
        
        H_gt = self._calculate_ground_truth_homography(uav_meta, sat_meta)
        
        return {
            "uav_image": uav_tensor,
            "sat_image": sat_tensor,
            "H_gt": torch.tensor(H_gt, dtype=torch.float32),
            "uav_name": uav_name,
            "sat_name": sat_name
        }