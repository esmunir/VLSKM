import torch
import torch.nn as nn
from omegaconf import OmegaConf


def weight_loss(log_assignment, weights, warped_kpts0=None, kpts1=None, gamma_d=0.0):
    b, m, n = log_assignment.shape
    m -= 1
    n -= 1

    loss_sc = log_assignment * weights

    num_neg0 = weights[:, :m, -1].sum(-1).clamp(min=1.0)
    num_neg1 = weights[:, -1, :n].sum(-1).clamp(min=1.0)
    num_pos = weights[:, :m, :n].sum((-1, -2)).clamp(min=1.0)

    # 1. Standard Assignment NLL Losses
    nll_pos = -loss_sc[:, :m, :n].sum((-1, -2))
    nll_pos /= num_pos.clamp(min=1.0)

    nll_neg0 = -loss_sc[:, :m, -1].sum(-1)
    nll_neg1 = -loss_sc[:, -1, :n].sum(-1)
    nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)

    # # 2. UAV Localization Center-Distance Regularization Loss (L_dist)
    # nll_dist = torch.zeros_like(nll_pos)
    # if gamma_d > 0.0 and warped_kpts0 is not None and kpts1 is not None:
    #     # Compute pair-wise pixel distance matrix between true warped targets and satellite candidates
    #     # warped_kpts0: (B, M, 2), kpts1: (B, N, 2)
    #     pixel_dists = torch.cdist(warped_kpts0[..., :2], kpts1[..., :2], p=2) # Shape: (B, M, N)
        
    #     # Extract the network predicted probability matrix
    #     assignment_prob = torch.exp(log_assignment[:, :m, :n])
        
    #     # Penalize predicted probabilities based on physical pixel distance error
    #     dist_error_matrix = assignment_prob * pixel_dists
        
    #     # Sum and normalize by the number of matchable keypoint centers
    #     nll_dist = dist_error_matrix.sum((-1, -2)) / num_pos.clamp(min=1.0)

    # 2. UAV Localization SQUARED Center-Distance Regularization Loss (L_dist)
    nll_dist = torch.zeros_like(nll_pos)
    if gamma_d > 0.0 and warped_kpts0 is not None and kpts1 is not None:
        # Compute pair-wise physical Euclidean distances in pixel space: (B, M, N)
        pixel_dists = torch.cdist(warped_kpts0[..., :2], kpts1[..., :2], p=2) # Shape: (B, M, N)
        
        # 🧠 THE QUADRATIC SHIFT: Square the distance elements.
        # This softly suppresses small near-misses but multiplies extreme outliers exponentially.
        squared_pixel_dists = torch.pow(pixel_dists, 2)
        
        # Extract the network predicted probability matrix
        assignment_prob = torch.exp(log_assignment[:, :m, :n])
        
        # Penalize assigned probabilities based on the squared pixel distance matrix
        dist_error_matrix = assignment_prob * squared_pixel_dists
        
        # Sum and normalize by the number of matchable keypoint centers
        nll_dist = dist_error_matrix.sum((-1, -2)) / num_pos.clamp(min=1.0)

    return nll_pos, nll_neg, nll_dist, num_pos, (num_neg0 + num_neg1) / 2.0


class NLLLoss(nn.Module):
    default_conf = {
        "nll_balancing": 0.5,
        "gamma_f": 0.0,  
        "gamma_d": 0.0,  # Localization Distance Weight (Set > 0 to activate, e.g., 0.01)
    }

    def __init__(self, conf):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.loss_fn = self.nll_loss

    def forward(self, pred, data, weights=None):
        log_assignment = pred["log_assignment"]
        
        # Safely extract the custom warped projection tensor and the satellite coordinates
        warped_kpts0 = data.get("warped_keypoints0")
        kpts1 = data["image1"].get("keypoints")
        
        if weights is None:
            weights = self.loss_fn(log_assignment, data)
            
        nll_pos, nll_neg, nll_dist, num_pos, num_neg = weight_loss(
            log_assignment, 
            weights, 
            warped_kpts0=warped_kpts0, 
            kpts1=kpts1, 
            gamma_d=self.conf.gamma_d
        )
        
        nll = (
            self.conf.nll_balancing * nll_pos + (1 - self.conf.nll_balancing) * nll_neg
        )
        
        # Total loss calculation incorporating our new regularization component
        total_loss = nll + (self.conf.gamma_d * nll_dist)

        return (
            total_loss,
            weights,
            {
                "total_loss": total_loss,      # Added so training logger reads the complete step gradient
                "assignment_nll": nll,
                "nll_pos": nll_pos,
                "nll_neg": nll_neg,
                "nll_dist": nll_dist,          # Metrics pass-through
                "num_matchable": num_pos,
                "num_unmatchable": num_neg,
            },
        )

    def nll_loss(self, log_assignment, data):
        m = data["image0"]["keypoints"].size(1)
        n = data["image1"]["keypoints"].size(1)
        positive = data["gt_assignment"].float()
        neg0 = (data["gt_matches0"] == -1).float()
        neg1 = (data["gt_matches1"] == -1).float()

        weights = torch.zeros_like(log_assignment)
        weights[:, :m, :n] = positive

        weights[:, :m, -1] = neg0
        weights[:, -1, :n] = neg1
        return weights