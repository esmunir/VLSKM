import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import os
import sys
import logging
import random
import math
import time
import kornia

# Import your custom dataset
from uav_dataset import TaipeiCrossViewDataset

# Setup Paths
CURRENT_DIR = os.path.abspath(os.getcwd())
ROOT_DIR = os.path.dirname(CURRENT_DIR) 
STEERERS_DIR = os.path.join(CURRENT_DIR, "affine-steerers-main")
GLUE_FACTORY_DIR = os.path.join(CURRENT_DIR, "glue-factory-main")
sys.path.append(STEERERS_DIR)
sys.path.append(GLUE_FACTORY_DIR)

from affine_steerers import dedode_descriptor_G, dedode_detector_L, dedode_descriptor_B
from affine_steerers.steerers import SteererSpread

# --- NEW MAMBA IMPORTS ---
from gluefactory.models.matchers.mambaglue_newstructure_multidir import MambaGlue
from gluefactory.models.utils.losses import NLLLoss

# ==========================================
# CUSTOM LOGGER
# ==========================================
class GlueLogger:
    def __init__(self):
        self.logger = logging.getLogger("gluefactory")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            formatter = logging.Formatter('[%(asctime)s gluefactory INFO] %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def info(self, msg):
        self.logger.info(msg)

    def log_loss(self, epoch, it, losses):
        clean_losses = {k: v.mean().item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        loss_str = f"loss {{total {clean_losses.get('total', 0):.3E}, "
        loss_str += f"last {clean_losses.get('last', 0):.3E}, "
        loss_str += f"spatial {clean_losses.get('spatial_loss', 0):.3E}, " # <-- ADDED THIS LINE
        loss_str += f"assignment_nll {clean_losses.get('assignment_nll', 0):.3E}, "
        loss_str += f"nll_pos {clean_losses.get('nll_pos', 0):.3E}, "
        loss_str += f"nll_neg {clean_losses.get('nll_neg', 0):.3E}, "
        loss_str += f"num_matchable {clean_losses.get('num_matchable', 0):.3E}, "
        loss_str += f"num_unmatchable {clean_losses.get('num_unmatchable', 0):.3E}, "
        loss_str += f"row_norm {clean_losses.get('row_norm', 0):.3E}}}"
        self.logger.info(f"[E {epoch} | it {it}] {loss_str}")

log = GlueLogger()

# ==========================================
# STANDALONE MAMBA LOSS FUNCTION (SVD-FREE) -- UPDATED
# ==========================================
def compute_mamba_loss(matcher, pred, data, loss_module):
    """Calculates Deep Supervision Loss across all Mamba layers with SVD-Free Spatial Reprojection"""
    def loss_params(pred, i):
        la, _ = matcher.log_assignment[i](
            pred["ref_descriptors0"][:, i], pred["ref_descriptors1"][:, i]
        )
        return {"log_assignment": la}

    N_layers = pred["ref_descriptors0"].shape[1]

    # 1. Base NLL Loss for the final layer (compute log_assignment ONCE, reuse below)
    final_layer_params = loss_params(pred, -1)
    nll, gt_weights, loss_metrics = loss_module(final_layer_params, data)

    # ---------------------------------------------------------
    # 🚀 SVD-FREE SPATIAL REPROJECTION LOSS
    # ---------------------------------------------------------
    # Get Log-Assignment Probabilities (Shape: [B, M, N])
    P = final_layer_params["log_assignment"][:, :-1, :-1].exp()

    kpts1 = data["image1"]["keypoints"]         # [B, N, 2]
    gt_matches0 = data["gt_matches0"]           # [B, M], -1 = unmatched

    valid_mask = gt_matches0 > -1               # [B, M] bool

    if valid_mask.any():
        b_idx, m_idx = valid_mask.nonzero(as_tuple=True)

        # Only compute soft-matches / targets for rows that actually have a GT match.
        # Avoids dividing near-zero dustbin-routed rows (NaN/Inf risk) entirely.
        P_valid = P[b_idx, m_idx]                        # [K, N]
        P_norm = P_valid / (P_valid.sum(dim=-1, keepdim=True) + 1e-6)

        kpts1_valid = kpts1[b_idx]                        # [K, N, 2]
        soft_kpts1 = torch.bmm(P_norm.unsqueeze(1), kpts1_valid).squeeze(1)  # [K, 2]

        # Exact GT target: gather the matched keypoint directly by index.
        # (No homography reprojection here — data["image0"]["keypoints"] may be
        # rotated/augmented (steered_kpts0_pix), so re-projecting through H_gt
        # would NOT land on the correct target. Using gt_matches0 index is
        # rotation-invariant and always correct.)
        safe_idx = gt_matches0[b_idx, m_idx]               # [K], all >= 0 here
        gt_kpts1 = kpts1[b_idx, safe_idx]                  # [K, 2]

        spatial_diff = torch.nn.functional.smooth_l1_loss(
            soft_kpts1, gt_kpts1, reduction='none'
        )  # [K, 2]
        spatial_loss = spatial_diff.mean()
    else:
        # No valid matches in this batch — contribute zero without breaking the graph.
        spatial_loss = P.sum() * 0.0

    # Combine NLL and Spatial Loss
    # Alpha dictates how heavily the network cares about pixel-perfect alignment
    alpha = 0.01  # 0.005
    total_loss = nll.clone() + (alpha * spatial_loss)
    loss_metrics["spatial_loss"] = spatial_loss.detach()
    # ---------------------------------------------------------

    sum_weights = 1.0

    # Deep supervision for intermediate layers (pure NLL)
    for i in range(N_layers - 1):
        params_i = loss_params(pred, i)
        layer_nll, _, _ = loss_module(params_i, data, weights=gt_weights)
        # NOTE: 1.0 ** anything == 1.0, so this currently weights every
        # intermediate layer equally regardless of depth. If you intended
        # a real decay (e.g. gamma < 1.0 discounting shallower layers),
        # replace `weight = 1.0 ** (...)` with your actual gamma value.
        weight = 1.0 ** (N_layers - i - 1)  # gamma
        sum_weights += weight
        total_loss = total_loss + layer_nll * weight

    total_loss /= sum_weights

    losses = {"total": total_loss, "last": nll.detach(), **loss_metrics}
    losses["row_norm"] = pred["log_assignment"].exp()[:, :-1].sum(2).mean(1)

    return losses

def compute_stage2_loss(pred, gt_matches0, gt_matches1):
    """Calculates Binary Cross-Entropy loss for the Deep Confidence Regressors"""
    # Create binary ground truth labels (1 if the point has a valid match > -1, else 0)
    gt_labels0 = (gt_matches0 > -1).float()
    gt_labels1 = (gt_matches1 > -1).float()
    
    total_loss = 0.0
    num_layers = len(pred["token0"]) # Should be N-1 layers
    
    # Temporarily disable autocast for the loss calculation to prevent float16 underflow
    with torch.autocast(device_type='cuda', enabled=False):
        for i in range(num_layers):
            # Explicitly cast predictions back to float32 before calculating BCE
            p0 = pred["token0"][i].float()
            p1 = pred["token1"][i].float()
            
            # BCE Loss between predicted confidence [0, 1] and binary label {0, 1}
            loss0 = torch.nn.functional.binary_cross_entropy(p0, gt_labels0)
            loss1 = torch.nn.functional.binary_cross_entropy(p1, gt_labels1)
            
            # Average the loss for this layer
            total_loss += (loss0 + loss1) / 2.0
        
    # Average across all intermediate layers
    return {"total": total_loss / num_layers}

# ==========================================
# UTILS & DATA
# ==========================================
def load_frozen_affine_models(device):
    weights_dir = os.path.join(CURRENT_DIR, "weights")
    detector = dedode_detector_L(device=device, weights=torch.load(os.path.join(weights_dir, "dedode_detector_C4.pth"), map_location=device))
    descriptor = dedode_descriptor_B(device=device, weights=torch.load(os.path.join(weights_dir, "descriptor_aff_equi_B.pth"), map_location=device))
    steerer = SteererSpread(
        max_order=4, normalize=True, normalize_only_higher=False, fix_order_1_scalings=False,
        max_determinant_scaling=None, block_diag_rot=False, block_diag_optimal_scalings=False,
        learnable_determinant_scaling=True, learnable_basis=True,               
        learnable_reference_direction=False, learnable_lstsq_weights=False,
    ).to(device)
    steerer.load_state_dict(torch.load(os.path.join(weights_dir, "steerer_aff_equi_B.pth"), map_location=device))
    
    for model in [detector, descriptor, steerer]:
        for param in model.parameters(): param.requires_grad = False
        model.eval()
    return detector, descriptor, steerer

def generate_ground_truth(kpts0, kpts1, H_gt, threshold_px=5.0):
    B, M, _ = kpts0.shape
    B, N, _ = kpts1.shape
    device = kpts0.device
    
    kpts0_h = torch.cat([kpts0, torch.ones(B, M, 1, device=device)], dim=-1)
    warped_kpts0_h = torch.bmm(kpts0_h, H_gt.transpose(1, 2))
    warped_kpts0 = warped_kpts0_h[..., :2] / warped_kpts0_h[..., 2:]
    
    dist = torch.cdist(warped_kpts0, kpts1)
    min_dist0, nn0 = dist.min(dim=2) 
    min_dist1, nn1 = dist.min(dim=1) 
    
    gt_matches0 = torch.full((B, M), -1, dtype=torch.long, device=device)
    gt_matches1 = torch.full((B, N), -1, dtype=torch.long, device=device)
    gt_assignment = torch.zeros((B, M, N), dtype=torch.bool, device=device)
    
    for b in range(B):
        mutual = (nn1[b][nn0[b]] == torch.arange(M, device=device))
        valid = mutual & (min_dist0[b] < threshold_px)
        
        valid_indices_0 = torch.nonzero(valid).squeeze(-1)
        valid_indices_1 = nn0[b][valid_indices_0]
        
        if len(valid_indices_0) > 0:
            gt_matches0[b, valid_indices_0] = valid_indices_1
            gt_matches1[b, valid_indices_1] = valid_indices_0
            gt_assignment[b, valid_indices_0, valid_indices_1] = True
            
    return gt_matches0, gt_matches1, gt_assignment

# ==========================================
# MAIN ARGPARSE & TRAINING LOOP
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="mamba-smoke-test", help="Experiment name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--num_kpts", type=int, default=2000, help="Number of keypoints to extract")
    parser.add_argument("--noise_deg", type=float, default=11.25, help="Artificial rotational noise")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1, help="1: Train matching, 2: Train early-stop regressors")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    ckpt_dir = os.path.join(CURRENT_DIR, "checkpoints", args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    log.info(f"🐍 Starting MAMBAGLUE Smoke Test: {args.exp_name}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch.backends.cudnn.benchmark = True 
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    dataset_path = os.path.join(ROOT_DIR, "Taipei") 
    train_dataset = TaipeiCrossViewDataset(dataset_dir=dataset_path, split="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True)
    
    detector, descriptor, steerer = load_frozen_affine_models(device)
    
    # --- MAMBAGLUE SETUP ---
    mamba_conf = {
        "name": "mambaglue",
        "input_dim": 256,       # Match DeDoDe output
        "descriptor_dim": 256, 
        "n_layers": 5,
        "num_heads": 4,
        "flash": True,
        "mp": True,             # Automatic Mixed Precision handled inside Mamba
        "depth_confidence": -1, # CRITICAL: Disable early stopping during training
        "width_confidence": -1, # CRITICAL: Disable pruning during training
        "filter_threshold": 0.1,
        "weights": None         # Train from scratch!
    }
    matcher = MambaGlue(features=None, **mamba_conf).to(device)

    # --- STAGE 2 PREPARATION ---
    if args.stage == 2:
        log.info("Transitioning to STAGE 2: Freezing backbone and training confidence regressors.")
        # Load your best Stage 1 checkpoint
        stage1_ckpt = os.path.join(CURRENT_DIR, "checkpoints", args.exp_name, "checkpoint_9.tar")
        matcher.load_state_dict(torch.load(stage1_ckpt, map_location=device))
        
        # Freeze everything
        for param in matcher.parameters():
            param.requires_grad = False
            
        # Unfreeze ONLY the TokenConfidence modules
        for module in matcher.token_confidence:
            for param in module.parameters():
                param.requires_grad = True

    matcher.train()
    
    # Base NLL Loss Module
    nll_loss_module = NLLLoss({"gamma": 1.0, "fn": "nll", "nll_balancing": 0.5}).to(device)
    
    # 🧠 THE FIX: Separate parameters to protect Mamba's internal states
    decay_params = []
    no_decay_params = []
    
    for name, param in matcher.named_parameters():
        if not param.requires_grad:
            continue
        # Don't decay biases, 1D parameters (like LayerNorm weights), or Mamba's explicit no_decay tags
        if hasattr(param, "_no_weight_decay") or param.ndim <= 1 or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optim_groups = [
        {"params": decay_params, "weight_decay": 1e-4},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    
    # Initialize AdamW with the protected groups
    optimizer = optim.AdamW(optim_groups, lr=args.lr)


    #🧠 THE FIX: Cosine Annealing Learning Rate Scheduler
    # This gradually reduces the LR to prevent massive activation spikes in the later epochs.
    grad_accum_steps = 4

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs * steps_per_epoch, 
        eta_min=1e-6
    )

    scaler = torch.amp.GradScaler('cuda') 
    
    static_img_size = torch.tensor([[504, 504]], device=device)
    total_iterations = 0
    
    for epoch in range(args.epochs):
        log.info(f"Starting epoch {epoch}")
        # 🌟 CONFIG: Accumulation steps for stable gradients
        grad_accum_steps = 4
        
        for batch_idx, batch in enumerate(train_loader):
            uav_img = batch["uav_image"].to(device, non_blocking=True)
            sat_img = batch["sat_image"].to(device, non_blocking=True)
            H_gt = batch["H_gt"].to(device, non_blocking=True)
            
            #optimizer.zero_grad(set_to_none=True) 
            
            # --- 1. FROZEN EXTRACTION ---
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                with torch.no_grad():
                    uav_kpts = detector.detect({"image": uav_img}, num_keypoints=args.num_kpts)
                    sat_kpts = detector.detect({"image": sat_img}, num_keypoints=args.num_kpts)
                    uav_desc = descriptor.describe_keypoints({"image": uav_img}, uav_kpts["keypoints"])
                    sat_desc = descriptor.describe_keypoints({"image": sat_img}, sat_kpts["keypoints"])
                    
                    R_gt = H_gt[:, :2, :2] 
                    B = uav_desc["descriptions"].shape[0]
                    
                    # 🔴 FLOAT32 SAFE ZONE (For Steerer)
                    with torch.autocast(device_type='cuda', enabled=False):
                        desc_f32 = uav_desc["descriptions"].float()
                        R_gt_f32 = R_gt.float()
                        s = torch.sqrt(R_gt_f32[:, 0, 0]**2 + R_gt_f32[:, 0, 1]**2).view(B, 1, 1)

                        R_gt_pure = R_gt_f32 / s
                        
                        noise_deg = (torch.rand(B, device=device) * 2 * args.noise_deg) - args.noise_deg
                        noise_rad = noise_deg * (math.pi / 180.0)
                        
                        cos_n = torch.cos(noise_rad)
                        sin_n = torch.sin(noise_rad)
                        R_noise = torch.stack([
                            torch.stack([cos_n, -sin_n], dim=-1),
                            torch.stack([sin_n,  cos_n], dim=-1)
                        ], dim=1).float()
                        
                        R_noisy_batch = torch.bmm(R_noise, R_gt_pure)
                        
                        steered_list = []
                        for b in range(B):
                            desc_b = desc_f32[b:b+1]
                            R_noisy_b = R_noisy_batch[b].unsqueeze(0)
                            steered_list.append(steerer(desc_b, R_noisy_b))
                            
                        steered_uav_desc = torch.cat(steered_list, dim=0)
                    
                    kpts0_pix = (uav_kpts["keypoints"] + 1.0) * 504 / 2.0
                    kpts1_pix = (sat_kpts["keypoints"] + 1.0) * 504 / 2.0
                    gt_m0, gt_m1, gt_assign = generate_ground_truth(kpts0_pix, kpts1_pix, H_gt, threshold_px=5.0)

                    # 2. Compute Steered Coordinates for the Model Input
                    # (Ensuring you center them around 252.0 before applying R_noisy_batch)
                    centered_kpts = kpts0_pix - 252.0
                    rotated_centered_kpts = torch.bmm(centered_kpts, R_noisy_batch.transpose(1, 2))
                    steered_kpts0_pix = rotated_centered_kpts + 252.0

                    # -----------------------------------------------------------------
                    # 🚀 FIX: Adjust H_gt matrix to account for rotated keypoint coords
                    # -----------------------------------------------------------------
                    H_gt_adj = H_gt.clone()
                    for b in range(B):
                        R_inv = R_noisy_batch[b].T
                        t_shift = torch.tensor([252.0, 252.0], device=device) - R_inv @ torch.tensor([252.0, 252.0], device=device)
                        
                        T_noise_inv = torch.eye(3, device=device)
                        T_noise_inv[:2, :2] = R_inv
                        T_noise_inv[:2, 2] = t_shift
                        
                        H_gt_adj[b] = torch.mm(H_gt[b], T_noise_inv)
                    # -----------------------------------------------------------------
                
                if gt_assign.sum().item() == 0:
                    optimizer.zero_grad(set_to_none=True)
                    continue
                    
                data = {
                    "image0": {
                        "keypoints": steered_kpts0_pix, #kpts0_pix,
                        "descriptors": steered_uav_desc,
                        "image_size": static_img_size
                    },
                    "image1": {
                        "keypoints": kpts1_pix,
                        "descriptors": sat_desc["descriptions"],
                        "image_size": static_img_size
                    },
                    "gt_matches0": gt_m0,
                    "gt_matches1": gt_m1,
                    "gt_assignment": gt_assign,
                    "H_gt": H_gt_adj  # 🚀 ADDED THIS LINE
                }

                # --- 2. MAMBAGLUE FORWARD PASS ---
                pred = matcher(data)
                
                # --- 3. CUSTOM MAMBA LOSS ---
                # --- 3. LOSS CALCULATION BASED ON STAGE ---
                if args.stage == 1:
                    losses = compute_mamba_loss(matcher, pred, data, nll_loss_module)
                else:
                    losses = compute_stage2_loss(pred, gt_m0, gt_m1)
                total_loss = losses["total"].mean()

             # --- 4. BACKPROPAGATION ---
            loss_scaled = total_loss / grad_accum_steps
            scaler.scale(loss_scaled).backward()

            # 🌟 STEP ONLY AFTER ACCUMULATION
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(matcher.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step() # Step only once per accumulated batch
                
            if total_iterations % 1000 == 0:
                log.log_loss(epoch, total_iterations, losses)
                
            total_iterations += 1
            #time.sleep(0.05) # Tiny leash to keep desktop usable
            
        ckpt_name = f"checkpoint{args.stage}_{epoch}.tar"
        torch.save(matcher.state_dict(), os.path.join(ckpt_dir, ckpt_name))
        log.info(f"💾 Saving checkpoint: {os.path.join(ckpt_dir, ckpt_name)}")