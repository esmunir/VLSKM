import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from affine_steerers.utils import *
from affine_steerers.matchers.dual_softmax_matcher import DualSoftMaxMatcher
import affine_steerers

class DescriptorLoss(nn.Module):
    def __init__(self,
                 detector,
                 num_keypoints = 5000,
                 steerer = None,
                 kptwise_affine_loss = False,
                 filter_kptwise_affine_loss = False,
                 affine_loss_factor = 1.,
                 nll_loss_factor = 1.,
                 steer_kptwise = False,
                 steer_with_estimated_affine = False,
                 ordinary_steer_and_estimated = False,
                 sing_value_cutoff = 5.,
                 matcher = DualSoftMaxMatcher(),
                 filter_inliers_with_mask = False,
                 artificial_incorrect_scale = False,
                 steer_base_affine = True,
                 nbr_random_prototypes = 2,
                 steer_closest_prototype = False,
                 use_equiv_loss = False,
                 equiv_loss_factor = 1.,
                 device = get_best_device()) -> None:
        super().__init__()
        if filter_kptwise_affine_loss and not steer_with_estimated_affine:
            raise ValueError()
        if ordinary_steer_and_estimated and not steer_with_estimated_affine:
            raise ValueError()
        self.detector = detector
        self.tracked_metrics = {}
        self.num_keypoints = num_keypoints
        self.steerer = steerer
        self.kptwise_affine_loss = kptwise_affine_loss
        self.filter_kptwise_affine_loss = filter_kptwise_affine_loss
        self.affine_loss_factor = affine_loss_factor
        self.nll_loss_factor = nll_loss_factor
        self.steer_kptwise = steer_kptwise
        self.steer_with_estimated_affine = steer_with_estimated_affine
        self.ordinary_steer_and_estimated = ordinary_steer_and_estimated
        self.sing_value_cutoff = sing_value_cutoff
        self.matcher = matcher
        self.filter_inliers_with_mask = filter_inliers_with_mask
        self.artificial_incorrect_scale = artificial_incorrect_scale
        self.steer_base_affine = steer_base_affine
        self.nbr_random_prototypes = nbr_random_prototypes
        self.steer_closest_prototype = steer_closest_prototype
        self.use_equiv_loss = use_equiv_loss
        self.equiv_loss_factor = equiv_loss_factor
    
    def warp_from_depth(self, batch, kpts_A, kpts_B):
        mask_A_to_B, kpts_A_to_B = warp_kpts(kpts_A, 
                    batch["im_A_depth"],
                    batch["im_B_depth"],
                    batch["T_1to2"],
                    batch["K1"],
                    batch["K2"],)
        mask_B_to_A, kpts_B_to_A = warp_kpts(kpts_B, 
                    batch["im_B_depth"],
                    batch["im_A_depth"],
                    batch["T_1to2"].inverse(),
                    batch["K2"],
                    batch["K1"],)
        return (mask_A_to_B, kpts_A_to_B), (mask_B_to_A, kpts_B_to_A)

    def affine_from_depth(self, batch, kpts_A, kpts_B):
        affine_A_to_B, _, kpts_A_to_B, mask_A_to_B = local_affine(
            kpts_A,
            batch["im_A_depth"],
            batch["im_B_depth"],
            batch["T_1to2"],
            batch["K1"],
            batch["K2"],
            default=batch["affine_AtoB"],
            sing_value_cutoff=self.sing_value_cutoff,
        )
        mask_B_to_A, kpts_B_to_A = warp_kpts(
            kpts_B, 
            batch["im_B_depth"],
            batch["im_A_depth"],
            batch["T_1to2"].inverse(),
            batch["K2"],
            batch["K1"],
        )
        return (mask_A_to_B, kpts_A_to_B, affine_A_to_B), (mask_B_to_A, kpts_B_to_A)
    
    def warp_from_homog(self, batch, kpts_A, kpts_B):
        kpts_A_to_B = homog_transform(batch["Homog_A_to_B"], kpts_A)
        kpts_B_to_A = homog_transform(batch["Homog_A_to_B"].inverse(), kpts_B)
        return (None, kpts_A_to_B), (None, kpts_B_to_A)

    def affine_from_homog(self, batch, kpts_A, kpts_B):
        affine_A_to_B, _, kpts_A_to_B, mask_A_to_B = local_affine_from_homography(
            kpts_A,
            batch["Homog_A_to_B"],
        )
        kpts_B_to_A = homog_transform(batch["Homog_A_to_B"].inverse(), kpts_B)
        return (mask_A_to_B, kpts_A_to_B, affine_A_to_B), (None, kpts_B_to_A)

    def supervised_loss(self, outputs, batch):
        kpts_A, kpts_B = self.detector.detect(
            batch, num_keypoints = self.num_keypoints)['keypoints'].clone().chunk(2)

        desc_grid_A, desc_grid_B = outputs["description_grid"].chunk(2)

        desc_A = F.grid_sample(
            desc_grid_A.float(), kpts_A[:,None], mode = "bilinear", align_corners = False)[:,:,0].mT
        desc_B = F.grid_sample(
            desc_grid_B.float(), kpts_B[:,None], mode = "bilinear", align_corners = False)[:,:,0].mT

        if "im_A_depth" in batch:
            if self.steer_kptwise or self.kptwise_affine_loss or self.steer_closest_prototype:
                (mask_A_to_B, kpts_A_to_B, affine_A_to_B), (mask_B_to_A, kpts_B_to_A) = (
                    self.affine_from_depth(batch, kpts_A, kpts_B)
                )
            else:
                (mask_A_to_B, kpts_A_to_B), (mask_B_to_A, kpts_B_to_A) = (
                    self.warp_from_depth(batch, kpts_A, kpts_B)
                )
        elif "Homog_A_to_B" in batch:
            if self.steer_kptwise or self.kptwise_affine_loss or self.steer_closest_prototype:
                (mask_A_to_B, kpts_A_to_B, affine_A_to_B), (mask_B_to_A, kpts_B_to_A) = (
                    self.affine_from_homog(batch, kpts_A, kpts_B)
                )
            else:
                (mask_A_to_B, kpts_A_to_B), (mask_B_to_A, kpts_B_to_A) = (
                    self.warp_from_homog(batch, kpts_A, kpts_B)
                )

        with torch.no_grad():
            D_B = torch.cdist(kpts_A_to_B, kpts_B)
            D_A = torch.cdist(kpts_A, kpts_B_to_A)
            mask = (
                (D_B == D_B.min(dim=-1, keepdim = True).values) 
                * (D_A == D_A.min(dim=-2, keepdim = True).values)
                * (D_B < 0.01)
                * (D_A < 0.01)
            )

            if self.filter_inliers_with_mask:
                mask *= mask_A_to_B[:, :, None]

            if self.artificial_incorrect_scale:
                p = torch.ones_like(kpts_A[0, :, 0])
                incorrect_ids = p.multinomial(num_samples=self.num_keypoints//2)
                angle1 = math.pi * torch.rand(self.num_keypoints//2, device=kpts_A.device)
                rot = torch.stack([
                    torch.stack([torch.cos(angle1), torch.sin(angle1)], dim=-1),
                    torch.stack([-torch.sin(angle1), torch.cos(angle1)], dim=-1),
                ], dim=-1)
                diag = 0.5 + torch.rand([self.num_keypoints//2, 2], device=kpts_A.device)
                affine_A_to_B[:, incorrect_ids] = (
                    rot.mT @ torch.diag_embed(diag) @ rot @ affine_A_to_B[:, incorrect_ids]
                )
                mask[:, incorrect_ids, :] = 0

            inds = torch.nonzero(mask)

        affine_loss = 0.
        if self.kptwise_affine_loss and not self.filter_kptwise_affine_loss:
            est_affine_A_to_ref = self.steerer.estimate_affine(
                desc_A[None, inds[:,0], inds[:,1]], to_ref=True,
            )[0]
            affine_loss = torch.norm(
                # est_affine_A_to_B - affine_A_to_B[inds[:,0], inds[:,1]],
                est_affine_A_to_ref @ torch.linalg.inv(
                    self.steerer.estimate_affine(
                        desc_B[None, inds[:, 0], inds[:, 2]], to_ref=True,
                    )[0].clone().detach()
                    @ affine_A_to_B[inds[:,0], inds[:,1]]
                )
                - torch.eye(2, device=desc_A.device, dtype=desc_A.dtype),
                dim=(-2, -1),
            ).mean()
            self.tracked_metrics["affine_loss"] = (
                0.99 * self.tracked_metrics.get(
                    "affine_loss", affine_loss.detach().item()
                ) + 0.01 * affine_loss.detach().item()
            )

        if self.steer_with_estimated_affine:
            est_affine_A_to_ref = self.steerer.estimate_affine(
                desc_A, to_ref=True,
            )
            est_affine_B_to_ref = self.steerer.estimate_affine(
                desc_B, to_ref=True,
            )

            if self.ordinary_steer_and_estimated:
                desc_A_ref = self.steerer(
                    desc_A,
                    est_affine_A_to_ref,
                )
                desc_B_ref = self.steerer(
                    desc_B,
                    est_affine_B_to_ref,
                )
            else:
                desc_A = self.steerer(
                    desc_A,
                    est_affine_A_to_ref,
                )
                desc_B = self.steerer(
                    desc_B,
                    est_affine_B_to_ref,
                )

            with torch.no_grad():
                est_A_inl = est_affine_A_to_ref[inds[:, 0], inds[:, 1]]
                est_B_inl = est_affine_B_to_ref[inds[:, 0], inds[:, 2]]
                sv_A = torch.linalg.svdvals(est_A_inl)
                sv_B = torch.linalg.svdvals(est_B_inl)
                good_idx = torch.nonzero(
                    torch.isfinite(sv_A.sum(dim=-1))
                    * torch.isfinite(sv_B.sum(dim=-1))
                    * (sv_A.max(dim=-1).values < self.sing_value_cutoff)
                    * (sv_B.max(dim=-1).values < self.sing_value_cutoff)
                    * (sv_A.min(dim=-1).values > 1. / self.sing_value_cutoff)
                    * (sv_B.min(dim=-1).values > 1. / self.sing_value_cutoff)
                    * (  # Same sign determinant as GT
                        torch.linalg.det(est_A_inl)
                        * torch.linalg.det(est_B_inl)
                        * torch.linalg.det(batch["affine_AtoB"][inds[:, 0]])
                        > 0
                    ) 
                )
                inds = inds[good_idx[:, 0]]
                if self.kptwise_affine_loss and self.filter_kptwise_affine_loss:
                    affine_loss = torch.norm(
                        (
                            fast_inv_2x2(
                                est_affine_A_to_ref[inds[:, 0], inds[:, 1]]
                            )
                            @ est_affine_B_to_ref[inds[:, 0], inds[:, 2]]
                            @ affine_A_to_B[inds[:, 0], inds[:, 1]]
                        )
                        - torch.eye(2, device=desc_A.device, dtype=desc_A.dtype),
                        dim=(-2, -1),
                    ).mean()
                    self.tracked_metrics["affine_loss"] = (
                        0.99 * self.tracked_metrics.get(
                            "affine_loss", affine_loss.detach().item()
                        ) + 0.01 * affine_loss.detach().item()
                    )
        elif self.steer_kptwise:
            desc_A = self.steerer(desc_A, affine_A_to_B)
        elif self.steerer is not None and "affine_AtoB" in batch and self.steer_base_affine:
            # same steerer for all kpts
            affine_AtoB = batch["affine_AtoB"][:, None]
            desc_A = self.steerer(desc_A, affine_AtoB)

        equiv_loss = 0.
        if self.use_equiv_loss:
            equiv_loss = torch.linalg.norm(
                desc_A[inds[:, 0], inds[:, 1]] - desc_B[inds[:, 0], inds[:, 2]],
                dim=-1,
            ).mean()
            self.tracked_metrics["equiv_loss"] = (
                0.99 * self.tracked_metrics.get(
                    "equiv_loss", equiv_loss.detach().item()
                ) + 0.01 * equiv_loss.detach().item()
            )

        if self.steerer is not None and self.steerer.use_prototype_affines:
            corr = self.matcher.similarity(desc_A, desc_B)
            if self.steerer.prototype_affines.shape[0] > self.nbr_random_prototypes:
                for _ in range(self.nbr_random_prototypes):
                    corr = torch.maximum(
                        corr,
                        self.matcher.similarity(
                            self.steerer.steer_with_random_prototypes(desc_A),
                            desc_B,
                        )
                    )
            else:
                for i in range(self.steerer.prototype_affines.shape[0]):
                    corr = torch.maximum(
                        corr,
                        self.matcher.similarity(
                            self.steerer(
                                desc_A,
                                self.steerer.prototype_affines[i, None, None],
                            ),
                            desc_B,
                        )
                    )
            if self.steer_closest_prototype:
                with torch.no_grad():
                    prototype_inds = torch.cdist(
                        affine_A_to_B[inds[:, 0], inds[:, 1]].flatten(start_dim=-2),
                        self.steerer.prototype_affines.flatten(start_dim=-2),
                    ).min(dim=-1).indices
                corr[inds[:, 0], inds[:, 1], inds[:, 2]] = torch.maximum(
                    -torch.linalg.norm(  # Fallback option of no steer
                        desc_A[inds[:, 0], inds[:, 1]]
                        - desc_B[inds[:, 0], inds[:, 2]],
                        dim=-1,
                    ),
                    -torch.linalg.norm(  # Optimal steer acc to annotations
                        self.steerer(
                            desc_A[None, inds[:, 0], inds[:, 1]],
                            self.steerer.prototype_affines[None, prototype_inds],
                        )[0] - desc_B[inds[:, 0], inds[:, 2]],
                        dim=-1,
                    )
                )
            corr = self.matcher.inv_temp * corr
            logP_A_B = corr.log_softmax(dim=-2) + corr.log_softmax(dim=-1)
        elif self.ordinary_steer_and_estimated:
            logP_A_B = 0.5 * (
                self.matcher.log_softmax(
                    self.steerer(desc_A, affine_A_to_B),
                    desc_B,
                )
                + self.matcher.log_softmax(desc_A_ref, desc_B_ref)
            )
        else:
            logP_A_B = self.matcher.log_softmax(desc_A, desc_B)
        neg_log_likelihood = -logP_A_B[inds[:,0], inds[:,1], inds[:,2]].mean()

        self.tracked_metrics["neg_log_likelihood"] = (
            0.99 * self.tracked_metrics.get(
                "neg_log_likelihood", neg_log_likelihood.detach().item()
            ) + 0.01 * neg_log_likelihood.detach().item()
        )
        if np.random.rand() > 0.99:
            print(f'nll: {self.tracked_metrics["neg_log_likelihood"]}')
            if self.steerer is not None and self.steerer.use_prototype_affines:
                print(f'prototypes: {self.steerer.prototype_affines[:8].clone().detach().cpu().numpy()}')
            if self.use_equiv_loss:
                print(f'equiv_loss: {self.tracked_metrics["equiv_loss"]}')
            if self.kptwise_affine_loss:
                print(f'aff_loss: {self.tracked_metrics["affine_loss"]}')
            if self.steerer is not None and self.steerer.learnable_reference_direction:
                print(f'steerer_ref_dir: {self.steerer.reference_direction.clone().detach().cpu().numpy()}')
            if self.steerer is not None and self.steerer.learnable_determinant_scaling:
                print(f'steerer_det_scalings: {[(n, x.data.max().item(), x.data.median().item(), x.data.min().item()) for n, x in enumerate(self.steerer.determinant_scalings)]}')

        loss = (
            self.nll_loss_factor * neg_log_likelihood 
            + self.affine_loss_factor * affine_loss
            + self.equiv_loss_factor * equiv_loss
        )

        return loss

    def forward(self,
                outputs,
                batch):
        losses = self.supervised_loss(outputs, batch)
        return losses
