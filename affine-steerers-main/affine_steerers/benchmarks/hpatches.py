from PIL import Image
import numpy as np

import os

from tqdm import tqdm
import torch
from affine_steerers.utils import pose_auc, to_homogeneous, from_homogeneous
import cv2

class HPatchesHomogMNN:
    """Hpatches grid goes from [0,n-1] instead of [0.5,n-0.5]"""

    def __init__(self, dataset_path) -> None:
        seqs_dir = "hpatches-sequences-release"
        self.seqs_path = os.path.join(dataset_path, seqs_dir)
        self.seq_names = sorted(os.listdir(self.seqs_path))
        # Ignore seqs is same as LoFTR.
        self.ignore_seqs = set(
            [
                "i_contruction",
                "i_crownnight",
                "i_dc",
                "i_pencils",
                "i_whitebuilding",
                "v_artisans",
                "v_astronautis",
                "v_talent",
            ]
        )
    
    def benchmark(self, detector, descriptor, matcher, model_name = None):
        n_matches = []
        homog_dists = []
        for seq_idx, seq_name in tqdm(
            enumerate(self.seq_names), total=len(self.seq_names), mininterval = 10.,
        ):
            if seq_name in self.ignore_seqs:
                continue
            im1_path = os.path.join(self.seqs_path, seq_name, "1.ppm")
            w1, h1 = Image.open(im1_path).size
            keypoints_A = detector.detect_from_path(im1_path)["keypoints"]
            description_A = descriptor.describe_keypoints_from_path(im1_path, keypoints_A)["descriptions"]
            
            for im_idx in range(2, 7):
                im2_path = os.path.join(self.seqs_path, seq_name, f"{im_idx}.ppm")
                w2, h2 = Image.open(im2_path).size
                H = np.loadtxt(
                    os.path.join(self.seqs_path, seq_name, "H_1_" + str(im_idx))
                )
                keypoints_B = detector.detect_from_path(im2_path)["keypoints"]
                description_B = descriptor.describe_keypoints_from_path(im2_path, keypoints_B)["descriptions"]
                matches_A, matches_B, batch_ids = matcher.match(keypoints_A, description_A, 
                                                                     keypoints_B, description_B,)

                matches_A, matches_B = matcher.to_pixel_coords(matches_A, matches_B, h1, w1, h2, w2)
                matches_A, matches_B = matches_A - 0.5, matches_B - 0.5
                try:
                    H_pred, inliers = cv2.findHomography(
                        matches_A.cpu().numpy(),
                        matches_B.cpu().numpy(),
                        method = cv2.RANSAC,
                        confidence = 0.99999,
                        ransacReprojThreshold = 3 * min(w2, h2) / 480,
                    )
                except:
                    H_pred = None
                if H_pred is None:
                    H_pred = np.zeros((3, 3))
                    H_pred[2, 2] = 1.0
                corners = np.array(
                    [[0, 0, 1], [0, h1 - 1, 1], [w1 - 1, 0, 1], [w1 - 1, h1 - 1, 1]]
                )
                real_warped_corners = np.dot(corners, np.transpose(H))
                real_warped_corners = (
                    real_warped_corners[:, :2] / real_warped_corners[:, 2:]
                )
                warped_corners = np.dot(corners, np.transpose(H_pred))
                warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
                mean_dist = np.mean(
                    np.linalg.norm(real_warped_corners - warped_corners, axis=1)
                ) / (min(w2, h2) / 480.0)
                homog_dists.append(mean_dist)
        n_matches = np.array(n_matches)
        thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        auc = pose_auc(np.array(homog_dists), thresholds)
        print(f"HPatches AUC: {auc}")
        return {
            "hpatches_homog_auc_3": auc[2],
            "hpatches_homog_auc_5": auc[4],
            "hpatches_homog_auc_10": auc[9],
        }