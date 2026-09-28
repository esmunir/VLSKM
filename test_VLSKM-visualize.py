import matplotlib.pyplot as plt
import os
import sys
import torch
import numpy as np
import cv2
import math
import csv
import time
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import glob
import argparse
import itertools

# ==========================================
# 1. PATH MANAGEMENT
# ==========================================
CURRENT_DIR = os.path.abspath(os.getcwd())
STEERERS_DIR = os.path.join(CURRENT_DIR, "affine-steerers-main")
GLUE_FACTORY_DIR = os.path.join(CURRENT_DIR, "glue-factory-main")
ROOT_DIR = os.path.dirname(CURRENT_DIR) 
DATASET_DIR = os.path.join(ROOT_DIR, "Taipei")
TEST_DIR = os.path.join(DATASET_DIR, "pairing_retrieval-google_uav-100_150_200-sat-27982_56102_1052", "test")
METADATA_DIR = os.path.join(DATASET_DIR, "metadata")

sys.path.append(STEERERS_DIR)
sys.path.append(GLUE_FACTORY_DIR)

# ==========================================
# 2. IMPORTS & MODEL INIT
# ==========================================
from omegaconf import OmegaConf
from gluefactory.models.matchers.mambaglue import MambaGlue

try:
    from affine_steerers import dedode_descriptor_G, dedode_detector_L, dedode_descriptor_B
    from affine_steerers.matchers.dual_softmax_matcher import MaxSimilarityMatcher
    from affine_steerers.utils import negative_distance_similarity
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit()

def load_models(device="cuda"):
    print(f"🧊 Loading Affine models into {device}...")
    weights_dir = os.path.join(CURRENT_DIR, "weights")
    detector_weights = torch.load(os.path.join(weights_dir, "dedode_detector_C4.pth"), map_location=device)
    detector = dedode_detector_L(device=device, weights=detector_weights).eval()

    descriptor_weights = torch.load(os.path.join(weights_dir, "descriptor_aff_equi_B.pth"), map_location=device)
    descriptor = dedode_descriptor_B(device=device, weights=descriptor_weights).eval()
    
    from affine_steerers.steerers import SteererSpread
    steerer = SteererSpread(
        max_order=4, normalize=True, normalize_only_higher=False, fix_order_1_scalings=False,
        max_determinant_scaling=None, block_diag_rot=False, block_diag_optimal_scalings=False,
        learnable_determinant_scaling=True, learnable_basis=True,               
        learnable_reference_direction=False, learnable_lstsq_weights=False,
    ).to(device)
    steerer.load_state_dict(torch.load(os.path.join(weights_dir, "steerer_aff_equi_B.pth"), map_location=device))
    steerer.eval()
    
    print(f"🧠 Loading Custom Trained MambaGlue (5 Layers)...")
    mamba_conf = {
        "name": "direction_aware_mambaglue",
        "input_dim": 256,       
        "descriptor_dim": 256, 
        "n_layers": 5,          
        "num_heads": 4,
        "flash": True,
        "mp": True,             
        "filter_threshold": 0.1,
        "weights": None         
    }
    
    mambaglue = MambaGlue(features=None, **mamba_conf).to(device)
    ckpt_path = os.path.join(CURRENT_DIR, "checkpoints", args.exp_name, "checkpoint1_9.tar")
    if not os.path.exists(ckpt_path):
        print(f"⚠️ WARNING: Could not find checkpoint at {ckpt_path}.")
        sys.exit()
    else:
        mambaglue.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("✅ MambaGlue weights loaded successfully.")
    
    mambaglue.eval()
    return detector, descriptor, steerer, mambaglue

def load_image_to_tensor(img_path, device, size=(504, 504)):
    img = Image.open(img_path).convert('RGB').resize(size, Image.Resampling.LANCZOS)
    tensor = TF.to_tensor(img)
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalizer(tensor).unsqueeze(0).to(device)

def load_all_metadata(metadata_dir):
    sat_dict = {}
    uav_dict = {}
    sat_csv = os.path.join(metadata_dir, "sat_data.csv")
    if os.path.exists(sat_csv):
        with open(sat_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sat_dict[os.path.basename(row['filename'])] = {'lat': float(row['center_lat']), 'lon': float(row['center_lon'])}
    
    uav_csvs = glob.glob(os.path.join(metadata_dir, "uav_test*.csv"))
    for csv_file in uav_csvs:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                uav_dict[os.path.basename(row['filename'])] = {
                    'lat': float(row['center_lat']), 'lon': float(row['center_lon']),
                    'yaw': float(row['rotation_degrees']), 'height': float(row['height'])
                }
    return sat_dict, uav_dict

def estimate_pose_ccw(H, gt_uav, gt_sat):
    center_uav = np.array([252.0, 252.0, 1.0])
    mapped_center = H @ center_uav
    est_sat_x = mapped_center[0] / mapped_center[2]
    est_sat_y = mapped_center[1] / mapped_center[2]

    m_per_px = 250.0 / 504.0
    dx_meters = (est_sat_x - 252.0) * m_per_px
    dy_meters = (est_sat_y - 252.0) * m_per_px
    d_east = dx_meters
    d_north = -dy_meters

    R_earth = 6378137.0
    sat_lat = gt_sat['lat']
    sat_lon = gt_sat['lon']
    
    est_yaw_cw = math.degrees(math.atan2(H[1, 0], H[0, 0])) % 360
    est_yaw_ccw = (360.0 - est_yaw_cw) % 360.0

    gt_d_north = (gt_uav['lat'] - sat_lat) * (math.pi / 180.0) * R_earth
    gt_d_east = (gt_uav['lon'] - sat_lon) * (math.pi / 180.0) * R_earth * math.cos(sat_lat * math.pi / 180.0)
    loc_error_m = math.hypot(d_north - gt_d_north, d_east - gt_d_east)
    
    yaw_error = abs(est_yaw_ccw - gt_uav['yaw'])
    yaw_error = min(yaw_error, 360.0 - yaw_error)

    return {"loc_err_m": loc_error_m, "yaw_err_deg": yaw_error}

# ==========================================
# 7. PIPELINE LOGIC (TOP-1 vs TOP-5 HYPOTHESIS)
# ==========================================
def process_and_verify(uav_path, sat_path, detector, descriptor, steerer, matcher, mambaglue, device, sat_dict, uav_dict):
    uav_tensor = load_image_to_tensor(uav_path, device)
    sat_tensor = load_image_to_tensor(sat_path, device)

    if device == "cuda": torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        uav_kpts = detector.detect({"image": uav_tensor}, num_keypoints=2000)
        sat_kpts = detector.detect({"image": sat_tensor}, num_keypoints=2000)
        uav_desc = descriptor.describe_keypoints({"image": uav_tensor}, uav_kpts["keypoints"])
        sat_desc = descriptor.describe_keypoints({"image": sat_tensor}, sat_kpts["keypoints"])

        # 1. CONVERT TO PIXEL SCALE ONCE (Unrotated)
        kpts0_pix = (uav_kpts["keypoints"] + 1.0) * 504.0 / 2.0
        kpts1_pix = (sat_kpts["keypoints"] + 1.0) * 504.0 / 2.0
        
        scores = []
        inv_temp = 5.0
        
        for r_idx in range(len(steerer.prototype_affines)):
            R = steerer.prototype_affines[r_idx].unsqueeze(0)
            steered_desc = steerer(uav_desc["descriptions"], R)
            
            sim = torch.einsum("bnd,bmd->bnm", steered_desc, sat_desc["descriptions"]) / inv_temp
            dual_softmax_sim = sim.softmax(dim=1) * sim.softmax(dim=2)
            score = dual_softmax_sim.max(dim=-1)[0].sum().item()
            scores.append((score, r_idx, steered_desc))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        top_5_hypotheses = scores[:5]

        # ⚡ Tracking both Top-1 and Top-5
        top1_inlier_count = 0
        top1_H = None
        
        best_top5_inlier_count = 0
        best_top5_H = None

        for rank, (score, r_idx, steered_desc) in enumerate(top_5_hypotheses):

            # ⚡ MATHEMATICAL FIX: Steer coordinates dynamically per hypothesis
            R_hyp = steerer.prototype_affines[r_idx].unsqueeze(0)
            centered_kpts = kpts0_pix - 252.0
            rotated_centered_kpts = torch.bmm(centered_kpts, R_hyp.transpose(1, 2))
            steered_kpts0_pix = rotated_centered_kpts + 252.0
            
            # ⚡ Match the exact dictionary structure from training
            mg_data = {
                "image0": {
                    "keypoints": steered_kpts0_pix, #uav_kpts["keypoints"],
                    "descriptors": steered_desc,
                    "image_size": torch.tensor([[504, 504]], device=device)
                },
                "image1": {
                    "keypoints": kpts1_pix, #sat_kpts["keypoints"],
                    "descriptors": sat_desc["descriptions"],
                    "image_size": torch.tensor([[504, 504]], device=device)
                }
            }
            
            # ⚡ FIX: bfloat16 prevents Mamba SSM state collapse
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                mg_pred = mambaglue(mg_data)
            
            m0 = mg_pred["matches0"][0] 
            valid_matches = m0 > -1
            idx_uav = torch.nonzero(valid_matches).squeeze(-1)
            idx_sat = m0[valid_matches]
            
            pts_uav = uav_kpts["keypoints"][0, idx_uav].cpu().numpy()
            pts_sat = sat_kpts["keypoints"][0, idx_sat].cpu().numpy()
            
            if len(pts_uav) >= 4:
                # Convert back to pixels for OpenCV math
                pts_uav_pix = (pts_uav + 1.0) * 504 / 2.0
                pts_sat_pix = (pts_sat + 1.0) * 504 / 2.0
                
                H, mask = cv2.estimateAffinePartial2D(pts_uav_pix, pts_sat_pix, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                
                # ... (rest of logging logic stays the same) ...
                
                if mask is not None:
                    inlier_count = np.sum(mask.ravel() == 1)
                    
                    if rank == 0:
                        top1_inlier_count = inlier_count
                        top1_H = H
                        
                    if inlier_count > best_top5_inlier_count:
                        best_top5_inlier_count = inlier_count
                        best_top5_H = H

    if device == "cuda": torch.cuda.synchronize()
    latency = time.perf_counter() - start_time

    # Evaluate Metrics
    gt_uav = uav_dict.get(os.path.basename(uav_path))
    gt_sat = sat_dict.get(os.path.basename(sat_path))
    
    top1_metrics = None
    if top1_H is not None and top1_inlier_count >= 4:
        top1_metrics = estimate_pose_ccw(np.vstack([top1_H, [0, 0, 1]]), gt_uav, gt_sat)

    top5_metrics = None
    if best_top5_H is not None and best_top5_inlier_count >= 4:
        top5_metrics = estimate_pose_ccw(np.vstack([best_top5_H, [0, 0, 1]]), gt_uav, gt_sat)

    return top1_metrics, top5_metrics, latency

# ==========================================
# SINGLE-PAIR MATCH VISUALIZATION
# ==========================================
def load_image_for_display(img_path, size=(504, 504)):
    """Loads a raw (non-normalized) uint8 RGB image for drawing/visualization."""
    img = Image.open(img_path).convert('RGB').resize(size, Image.Resampling.LANCZOS)
    return np.array(img)

def visualize_single_pair(uav_path, sat_path, detector, descriptor, steerer, mambaglue,
                           device, sat_dict, uav_dict, save_path):
    """
    Runs the same detect -> describe -> rotate-and-steer -> MambaGlue-match pipeline
    as process_and_verify(), but for ONE UAV/satellite pair, and renders the matched
    keypoints (green = RANSAC inlier, red = outlier) on a side-by-side image saved to disk.
    """
    uav_tensor = load_image_to_tensor(uav_path, device)
    sat_tensor = load_image_to_tensor(sat_path, device)

    with torch.no_grad():
        uav_kpts = detector.detect({"image": uav_tensor}, num_keypoints=2000)
        sat_kpts = detector.detect({"image": sat_tensor}, num_keypoints=2000)
        uav_desc = descriptor.describe_keypoints({"image": uav_tensor}, uav_kpts["keypoints"])
        sat_desc = descriptor.describe_keypoints({"image": sat_tensor}, sat_kpts["keypoints"])

        kpts0_pix = (uav_kpts["keypoints"] + 1.0) * 504.0 / 2.0
        kpts1_pix = (sat_kpts["keypoints"] + 1.0) * 504.0 / 2.0

        # Pick the single best rotation hypothesis (top-1), same scoring as the benchmark loop
        scores = []
        inv_temp = 5.0
        for r_idx in range(len(steerer.prototype_affines)):
            R = steerer.prototype_affines[r_idx].unsqueeze(0)
            steered_desc = steerer(uav_desc["descriptions"], R)
            sim = torch.einsum("bnd,bmd->bnm", steered_desc, sat_desc["descriptions"]) / inv_temp
            dual_softmax_sim = sim.softmax(dim=1) * sim.softmax(dim=2)
            score = dual_softmax_sim.max(dim=-1)[0].sum().item()
            scores.append((score, r_idx, steered_desc))
        scores.sort(key=lambda x: x[0], reverse=True)
        _, r_idx, steered_desc = scores[0]

        R_hyp = steerer.prototype_affines[r_idx].unsqueeze(0)
        centered_kpts = kpts0_pix - 252.0
        rotated_centered_kpts = torch.bmm(centered_kpts, R_hyp.transpose(1, 2))
        steered_kpts0_pix = rotated_centered_kpts + 252.0

        mg_data = {
            "image0": {
                "keypoints": steered_kpts0_pix,
                "descriptors": steered_desc,
                "image_size": torch.tensor([[504, 504]], device=device)
            },
            "image1": {
                "keypoints": kpts1_pix,
                "descriptors": sat_desc["descriptions"],
                "image_size": torch.tensor([[504, 504]], device=device)
            }
        }

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            mg_pred = mambaglue(mg_data)

        m0 = mg_pred["matches0"][0]
        valid_matches = m0 > -1
        idx_uav = torch.nonzero(valid_matches).squeeze(-1)
        idx_sat = m0[valid_matches]

        pts_uav = uav_kpts["keypoints"][0, idx_uav].cpu().numpy()
        pts_sat = sat_kpts["keypoints"][0, idx_sat].cpu().numpy()

    pts_uav_pix = (pts_uav + 1.0) * 504.0 / 2.0
    pts_sat_pix = (pts_sat + 1.0) * 504.0 / 2.0
    n_matches = len(pts_uav_pix)

    inlier_mask = np.zeros(n_matches, dtype=bool)
    H = None
    if n_matches >= 4:
        H, mask = cv2.estimateAffinePartial2D(pts_uav_pix, pts_sat_pix, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if mask is not None:
            inlier_mask = mask.ravel().astype(bool)

    metrics = None
    gt_uav = uav_dict.get(os.path.basename(uav_path))
    gt_sat = sat_dict.get(os.path.basename(sat_path))
    if H is not None and inlier_mask.sum() >= 4 and gt_uav and gt_sat:
        metrics = estimate_pose_ccw(np.vstack([H, [0, 0, 1]]), gt_uav, gt_sat)

    # ---- Draw ----
    uav_img = load_image_for_display(uav_path)
    sat_img = load_image_for_display(sat_path)
    canvas = np.concatenate([uav_img, sat_img], axis=1)  # satellite image sits to the right, +504px in x
    x_offset = uav_img.shape[1]

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
    ax.imshow(canvas)
    ax.axis('off')

    for i in range(n_matches):
        x0, y0 = pts_uav_pix[i]
        x1, y1 = pts_sat_pix[i]
        color = 'lime' if inlier_mask[i] else 'red'
        ax.plot([x0, x1 + x_offset], [y0, y1], color=color, linewidth=0.6, alpha=0.6, zorder=1)
        ax.scatter([x0, x1 + x_offset], [y0, y1], s=8, color=color, zorder=2)

    n_inliers = int(inlier_mask.sum())
    title = (f"{os.path.basename(uav_path)}  ↔  {os.path.basename(sat_path)}\n"
              f"Matches: {n_matches} | Inliers: {n_inliers}")
    if metrics is not None:
        title += f" | Loc err: {metrics['loc_err_m']:.2f} m | Yaw err: {metrics['yaw_err_deg']:.2f}°"
    elif gt_uav is None or gt_sat is None:
        title += " | (no GT metadata found for this pair)"
    else:
        title += " | Pose estimation failed (too few inliers)"
    ax.set_title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved match visualization to {save_path}")
    return metrics

# ==========================================
# MAIN BATCH EXECUTION & AACHEN METRICS
# ==========================================
def calculate_academic_metrics(loc_list, yaw_list):
    """Calculates Mean (of successes), Median, and strict Aachen-style Pose Recalls"""
    locs = np.array(loc_list)
    yaws = np.array(yaw_list)
    
    # Calculate Mean ONLY on successful predictions (ignore inf) to see how accurate it is when it works
    valid_mask = locs != np.inf
    mean_loc = np.mean(locs[valid_mask]) if np.any(valid_mask) else np.inf
    mean_yaw = np.mean(yaws[valid_mask]) if np.any(valid_mask) else np.inf
    
    # Calculate Median across EVERYTHING (including failures)
    median_loc = np.median(locs)
    median_yaw = np.median(yaws)
    
    # Similar with Aachen Pose Recall Benchmarks (Both Loc AND Yaw must be satisfied)
    recall_1m_1d = np.mean((locs <= 1.0) & (yaws <= 1.0)) * 100
    recall_5m_5d  = np.mean((locs <= 5.0)  & (yaws <= 5.0)) * 100
    recall_10m_10d  = np.mean((locs <= 10.0)  & (yaws <= 10.0)) * 100
    
    return mean_loc, median_loc, mean_yaw, median_yaw, recall_1m_1d, recall_5m_5d, recall_10m_10d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="sp-lightglue-baseline", help="Name of the experiment")
    parser.add_argument("--mode", type=str, choices=["benchmark", "visualize"], default="benchmark",
                         help="'benchmark': run the full evaluation over every UAV/satellite pair (slow). "
                              "'visualize': generate one match-visualization image for a single chosen "
                              "UAV image (and its class-matching satellite image) and exit.")
    parser.add_argument("--class_id", type=str, default=None,
                         help="[visualize mode] Class/folder ID under query_drone/ and gallery_satellite/ to pick from.")
    parser.add_argument("--uav_name", type=str, default=None,
                         help="[visualize mode] Specific UAV image filename inside --class_id. "
                              "Defaults to the first .png found if omitted.")
    parser.add_argument("--sat_name", type=str, default=None,
                         help="[visualize mode] Specific satellite image filename inside --class_id. "
                              "Defaults to the first .png found if omitted.")
    parser.add_argument("--out_name", type=str, default=None,
                         help="[visualize mode] Output PNG filename. Defaults to an auto-generated name.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("\n--- Academic Benchmark Pipeline (Top-1 vs Top-5) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sat_dict, uav_dict = load_all_metadata(METADATA_DIR)
    detector, descriptor, steerer, mambaglue = load_models(device) 
    
    num_rotations = 16
    angles = torch.linspace(0, 2 * np.pi, num_rotations + 1)[:-1]
    steerer.prototype_affines = torch.stack([
        torch.tensor([[torch.cos(a), -torch.sin(a)], [torch.sin(a), torch.cos(a)]], device=device, dtype=torch.float32)
        for a in angles
    ])
    matcher = MaxSimilarityMatcher(steerer=steerer, inv_temp=5, threshold=0.01, similarity=negative_distance_similarity).to(device)
    
    output_dir = os.path.join(CURRENT_DIR, "checkpoints", args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # VISUALIZE MODE: process every UAV x satellite pair within the ONE
    # class you pick, then exit — skips the full (slow) gallery loop
    # across all other classes entirely.
    # ------------------------------------------------------------------
    if args.mode == "visualize":
        if not args.class_id:
            print("❌ --class_id is required for --mode visualize (e.g. --class_id 0001)")
            sys.exit()

        uav_class_dir = os.path.join(TEST_DIR, "query_drone", args.class_id)
        sat_class_dir = os.path.join(TEST_DIR, "gallery_satellite", args.class_id)

        # If a specific filename is given, use just that one image; otherwise take
        # every .png in the class folder (there can be more than one on either side).
        if args.uav_name:
            uav_paths = [os.path.join(uav_class_dir, args.uav_name)]
        else:
            uav_paths = sorted(glob.glob(os.path.join(uav_class_dir, "*.png")))

        if args.sat_name:
            sat_paths = [os.path.join(sat_class_dir, args.sat_name)]
        else:
            sat_paths = sorted(glob.glob(os.path.join(sat_class_dir, "*.png")))

        if not uav_paths:
            print(f"❌ No UAV images found in {uav_class_dir}")
            sys.exit()
        if not sat_paths:
            print(f"❌ No satellite images found in {sat_class_dir}")
            sys.exit()
        for p in uav_paths + sat_paths:
            if not os.path.exists(p):
                print(f"❌ Image not found: {p}")
                sys.exit()

        pairs = list(itertools.product(uav_paths, sat_paths))
        print(f"🖼️  Class '{args.class_id}': {len(uav_paths)} UAV image(s) x "
              f"{len(sat_paths)} satellite image(s) = {len(pairs)} pair(s) to visualize")

        # A custom --out_name only makes sense for a single pair; for multiple
        # pairs each gets its own auto-generated, non-colliding filename.
        use_custom_name = args.out_name and len(pairs) == 1

        for i, (uav_path, sat_path) in enumerate(pairs, start=1):
            if use_custom_name:
                out_name = args.out_name
            else:
                out_name = (
                    f"match_{args.class_id}_"
                    f"{os.path.splitext(os.path.basename(uav_path))[0]}_vs_"
                    f"{os.path.splitext(os.path.basename(sat_path))[0]}.png"
                )
            save_path = os.path.join(output_dir, out_name)

            print(f"  [{i}/{len(pairs)}] {os.path.basename(uav_path)} ↔ {os.path.basename(sat_path)}")
            visualize_single_pair(uav_path, sat_path, detector, descriptor, steerer, mambaglue,
                                   device, sat_dict, uav_dict, save_path)

        sys.exit()

    # ------------------------------------------------------------------
    # BENCHMARK MODE (default, unchanged): loops over the entire gallery
    # ------------------------------------------------------------------
    # Tracking Lists
    t1_loc, t1_yaw = [], []
    t5_loc, t5_yaw = [], []
    all_latencies = []
    
    print("🔥 Warming up CUDA kernels...")
    gallery_dirs = sorted(glob.glob(os.path.join(TEST_DIR, "gallery_satellite", "*")))
    _ = process_and_verify(
        glob.glob(os.path.join(TEST_DIR, "query_drone", "*", "*.png"))[0],
        glob.glob(os.path.join(TEST_DIR, "gallery_satellite", "*", "*.png"))[0],
        detector, descriptor, steerer, matcher, mambaglue, device, sat_dict, uav_dict
    )
    
    for class_dir in gallery_dirs:
        class_id = os.path.basename(class_dir)
        sat_img_paths = sorted(glob.glob(os.path.join(TEST_DIR, "gallery_satellite", class_id, "*.png")))
        uav_img_paths = sorted(glob.glob(os.path.join(TEST_DIR, "query_drone", class_id, "*.png")))
        
        print(f"📁 Processing Class {class_id}: {len(uav_img_paths)}x{len(sat_img_paths)} pairs.")
        
        for uav_path in uav_img_paths:
            for sat_path in sat_img_paths:
                t1_met, t5_met, latency = process_and_verify(uav_path, sat_path, detector, descriptor, steerer, matcher, mambaglue, device, sat_dict, uav_dict)
                all_latencies.append(latency)
                
                # TOP 1 LOGGING (np.inf if failed)
                if t1_met is not None:
                    t1_loc.append(t1_met["loc_err_m"]); t1_yaw.append(t1_met["yaw_err_deg"])
                else:
                    t1_loc.append(np.inf); t1_yaw.append(np.inf)
                    
                # TOP 5 LOGGING (np.inf if failed)
                if t5_met is not None:
                    t5_loc.append(t5_met["loc_err_m"]); t5_yaw.append(t5_met["yaw_err_deg"])
                else:
                    t5_loc.append(np.inf); t5_yaw.append(np.inf)

    # Compute Final Benchmarks
    t1_m_loc, t1_med_loc, t1_m_yaw, t1_med_yaw, t1_r1, t1_r2, t1_r3 = calculate_academic_metrics(t1_loc, t1_yaw)
    t5_m_loc, t5_med_loc, t5_m_yaw, t5_med_yaw, t5_r1, t5_r2, t5_r3 = calculate_academic_metrics(t5_loc, t5_yaw)
    
    mean_time = np.mean(all_latencies)
    
    summary_text = (
        f"\n{'='*60}\n"
        f"🎯 ACADEMIC BENCHMARK: VLSKM\n"
        f"{'='*60}\n"
        f"Total Pairs Evaluated : {len(all_latencies)}\n"
        f"Average Latency       : {mean_time:.3f} s/pair ({1.0/mean_time:.1f} FPS)\n"
        f"{'-'*60}\n"
        f"🏆 TOP-1 HYPOTHESIS\n"
        f"  Mean Error (Inliers): {t1_m_loc:.2f}m, {t1_m_yaw:.2f}°\n"
        f"  Median Error (All)  : {t1_med_loc:.2f}m, {t1_med_yaw:.2f}°\n"
        f"  Recall (1.0m, 1°)  : {t1_r1:.2f} %\n"
        f"  Recall (5.0m, 5°)  : {t1_r2:.2f} %\n"
        f"  Recall (10.0m, 10°) : {t1_r3:.2f} %\n"
        f"{'-'*60}\n"
        f"🏆 TOP-5 HYPOTHESIS\n"
        f"  Mean Error (Inliers): {t5_m_loc:.2f}m, {t5_m_yaw:.2f}°\n"
        f"  Median Error (All)  : {t5_med_loc:.2f}m, {t5_med_yaw:.2f}°\n"
        f"  Recall (1.0m, 1°)  : {t5_r1:.2f} %\n"
        f"  Recall (5.0m, 5°)  : {t5_r2:.2f} %\n"
        f"  Recall (10.0m, 10°) : {t5_r3:.2f} %\n"
        f"{'='*60}\n"
    )
    
    print(summary_text)
    results_file_path = os.path.join(output_dir, "benchmark_summary-new.txt")
    with open(results_file_path, "w") as text_file:
        text_file.write(summary_text)