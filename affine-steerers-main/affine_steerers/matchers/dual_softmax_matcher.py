import torch
from PIL import Image
import torch.nn as nn
import torchvision.models as tvm
import torch.nn.functional as F
import numpy as np
from affine_steerers.utils import to_pixel_coords, to_normalized_coords
from affine_steerers.utils import dual_log_softmax_matcher, dual_softmax_matcher
from affine_steerers.utils import scalar_product_similarity, negative_distance_similarity
from affine_steerers.utils import affine_dual_softmax_matcher
from affine_steerers.utils import fast_inv_2x2, lstsq_affine
from affine_steerers.utils import build_affine

class DualSoftMaxMatcher(nn.Module):
    def __init__(self, normalize=False,
                 inv_temp=1., threshold=0.,
                 topk=-1,
                 similarity=scalar_product_similarity):
        super().__init__()
        self.normalize = normalize
        self.inv_temp = inv_temp
        self.threshold = threshold
        self.topk = topk
        self.similarity = similarity

    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B, P_A = None, P_B = None):
        if isinstance(descriptions_A, list):
            matches = [self.match(k_A[None], d_A[None], k_B[None], d_B[None]) 
                    for k_A,d_A,k_B,d_B in
                    zip(keypoints_A, descriptions_A, keypoints_B, descriptions_B)]
            matches_A = torch.cat([m[0] for m in matches])
            matches_B = torch.cat([m[1] for m in matches])
            inds = torch.cat([m[2] + b for b, m in enumerate(matches)])
            return matches_A, matches_B, inds
        
        P = dual_softmax_matcher(
            descriptions_A, descriptions_B, 
            normalize=self.normalize,
            inv_temperature=self.inv_temp,
            similarity=self.similarity,
        )
        inds = torch.nonzero(
            (P == P.max(dim=-1, keepdim = True).values) 
            * (P == P.max(dim=-2, keepdim = True).values)
            * (P > self.threshold)
        )
        if self.topk > -1 and inds.shape[0] > self.topk:
            dists = P[inds[:, 0], inds[:, 1], inds[:, 2]]
            inds = inds[
                torch.topk(dists, k=self.topk, sorted=False).indices
            ]

        batch_inds = inds[:,0]
        matches_A = keypoints_A[batch_inds, inds[:,1]]
        matches_B = keypoints_B[batch_inds, inds[:,2]]
        return matches_A, matches_B, inds

    def log_softmax(self, descriptions_A, descriptions_B):
        return dual_log_softmax_matcher(
            descriptions_A,
            descriptions_B,
            normalize=self.normalize,
            inv_temperature=self.inv_temp,
            similarity=self.similarity,
        )

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)
    
    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)


class PostHocAffineDualSoftMaxMatcher(nn.Module):
    def __init__(
        self, steerer, normalize=False,
        inv_temp=1., threshold=0.,
        similarity=scalar_product_similarity,
        remove_negative_determinants=False,
        sing_value_cutoff=3.,
    ):
        super().__init__()
        self.steerer = steerer
        self.normalize = normalize
        self.inv_temp = inv_temp
        self.threshold = threshold
        self.similarity = similarity
        self.remove_negative_determinants = remove_negative_determinants
        self.sing_value_cutoff = sing_value_cutoff

    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B, P_A = None, P_B = None):
        if isinstance(descriptions_A, list):
            matches = [self.match(k_A[None], d_A[None], k_B[None], d_B[None]) 
                    for k_A,d_A,k_B,d_B in
                    zip(keypoints_A, descriptions_A, keypoints_B, descriptions_B)]
            matches_A = torch.cat([m[0] for m in matches])
            matches_B = torch.cat([m[1] for m in matches])
            inds = torch.cat([m[2] + b for b, m in enumerate(matches)])
            return matches_A, matches_B, inds
        
        P = dual_softmax_matcher(
            descriptions_A, descriptions_B, 
            normalize=self.normalize,
            inv_temperature=self.inv_temp,
            similarity=self.similarity,
        )

        inds = torch.nonzero(
            (P == P.max(dim=-1, keepdim = True).values) 
            * (P == P.max(dim=-2, keepdim = True).values)
            * (P > self.threshold)
        )

        desc0_A, desc1_A = self.steerer.split_into_orders(
            descriptions_A[inds[:, 0], inds[:, 1]]
        )[:2]
        desc0_B, desc1_B = self.steerer.split_into_orders(
            descriptions_B[inds[:, 0], inds[:, 2]]
        )[:2]
        lstsq_weights = torch.sqrt(
            self.steerer.feat0_to_lstsq_weights(
                desc0_A,
            )
            * self.steerer.feat0_to_lstsq_weights(
                desc0_B,
            )
        )
        est_A_to_B = lstsq_affine(
            desc1_A.reshape(desc1_A.shape[0], -1, 2),
            desc1_B.reshape(desc1_B.shape[0], -1, 2),
            lstsq_weights,
        )
        sv = torch.linalg.svdvals(est_A_to_B)
        good_est = (
            torch.isfinite(sv.sum(dim=-1))
            * (sv.max(dim=-1).values < self.sing_value_cutoff)
            * (sv.min(dim=-1).values > 1. / self.sing_value_cutoff)
        )
        if self.remove_negative_determinants:
            good_est *= (torch.linalg.det(est_A_to_B) > 0) 

        inds = inds[good_est]

        matches_A = keypoints_A[inds[:, 0], inds[:, 1]]
        matches_B = keypoints_B[inds[:, 0], inds[:, 2]]
        return matches_A, matches_B, inds

    def log_softmax(self, descriptions_A, descriptions_B):
        return dual_log_softmax_matcher(
            descriptions_A,
            descriptions_B,
            normalize=self.normalize,
            inv_temperature=self.inv_temp,
            similarity=self.similarity,
        )

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)
    
    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)

class MeanTTADualSoftMaxMatcher(torch.nn.Module):
    def __init__(
        self, steerer, 
        affine_augs,
        normalize=False,
        inv_temp=1., threshold=0.,
        similarity=scalar_product_similarity,
    ):
        super().__init__()
        self.steerer = steerer
        self.normalize = normalize
        self.inv_temp = inv_temp
        self.threshold = threshold
        self.similarity = similarity
        self.affine_augs = affine_augs

    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B,
              P_A = None, P_B = None):
        if isinstance(descriptions_A, list):
            matches = [self.match(k_A[None], d_A[None], k_B[None], d_B[None], normalize = normalize,
                                  inv_temp = inv_temp, threshold = threshold) 
                       for k_A,d_A,k_B,d_B in
                       zip(keypoints_A, descriptions_A, keypoints_B, descriptions_B)]
            matches_A = torch.cat([m[0] for m in matches])
            matches_B = torch.cat([m[1] for m in matches])
            inds = torch.cat([m[2] + b for b, m in enumerate(matches)])
            return matches_A, matches_B, inds

        descriptions_A_av = descriptions_A.clone()
        descriptions_B_av = descriptions_B.clone()
        for aff in self.affine_augs:
            descriptions_A_av += self.steerer(descriptions_A, aff[None, None])
            descriptions_B_av += self.steerer(descriptions_B, aff[None, None])
        descriptions_A_av /= len(self.affine_augs) + 1
        descriptions_B_av /= len(self.affine_augs) + 1

        P = dual_softmax_matcher(
            descriptions_A_av, descriptions_B_av, 
            normalize=self.normalize, inv_temperature=self.inv_temp,
            similarity=self.similarity,
        )

        inds = torch.nonzero(
            (P == P.max(dim=-1, keepdim = True).values) 
            * (P == P.max(dim=-2, keepdim = True).values) 
            * (P > self.threshold)
        )

        matches_A = keypoints_A[inds[:, 0], inds[:, 1]]
        matches_B = keypoints_B[inds[:, 0], inds[:, 2]]
        return matches_A, matches_B, inds

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)

    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)

class MaxSimilarityMatcher(torch.nn.Module):
    def __init__(
        self, steerer, 
        normalize=False,
        inv_temp=1., threshold=0.,
        similarity=negative_distance_similarity,
        randomized=False,
        nbr_randomized=2,
        low_gpu_memory=False,
    ):
        super().__init__()
        self.steerer = steerer
        self.normalize = normalize
        self.inv_temp = inv_temp
        self.threshold = threshold
        self.similarity = similarity
        self.randomized = randomized
        self.nbr_randomized = nbr_randomized
        self.low_gpu_memory = low_gpu_memory

    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B,
              P_A = None, P_B = None,
              H_A=None, W_A=None, H_B=None, W_B=None,
             ):
        if isinstance(descriptions_A, list):
            matches = [self.match(k_A[None], d_A[None], k_B[None], d_B[None], normalize = normalize,
                                  inv_temp = inv_temp, threshold = threshold) 
                       for k_A,d_A,k_B,d_B in
                       zip(keypoints_A, descriptions_A, keypoints_B, descriptions_B)]
            matches_A = torch.cat([m[0] for m in matches])
            matches_B = torch.cat([m[1] for m in matches])
            inds = torch.cat([m[2] + b for b, m in enumerate(matches)])
            return matches_A, matches_B, inds

        B, N_A, _ = descriptions_A.shape
        _, N_B, _ = descriptions_B.shape
        corr = self.similarity(descriptions_A, descriptions_B)
        if self.randomized:
            for _ in range(self.nbr_randomized):
                if self.low_gpu_memory:
                    prototype_ids = np.random.choice(
                        len(self.affine_augs),
                        size=descriptions_A.shape[1],
                        replace=True,
                    )
                    descriptions_A_aug = descriptions_A.clone()
                    for i in range(len(self.affine_augs)):
                        descriptions_A_aug[:, prototype_ids==i] = self.steerer(
                            descriptions_A[:, prototype_ids==i],
                            steerer.prototype_affines[None, None, i],
                        )
                else:
                    descriptions_A_aug = self.steerer.steer_with_random_prototypes(
                        descriptions_A,
                    )

                corr = torch.maximum(
                    corr,
                    self.similarity(
                        descriptions_A_aug,
                        descriptions_B,
                    ),
                )

        else:
            for i in range(self.steerer.prototype_affines.shape[0]):
                descriptions_A_aug = self.steerer(
                    descriptions_A,
                    self.steerer.prototype_affines[None, None, i],
                )
                corr = torch.maximum(
                    corr,
                    self.similarity(
                        descriptions_A_aug,
                        descriptions_B,
                    )
                )

        corr = self.inv_temp * corr
        P = corr.softmax(dim = -2) * corr.softmax(dim= -1)

        inds = torch.nonzero(
            (P == P.max(dim=-1, keepdim = True).values) 
            * (P == P.max(dim=-2, keepdim = True).values) 
            * (P > self.threshold)
        )

        matches_A = keypoints_A[inds[:, 0], inds[:, 1]]
        matches_B = keypoints_B[inds[:, 0], inds[:, 2]]
        return matches_A, matches_B, inds

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)

    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)

class MaxSimilarityTTADualSoftMaxMatcher(torch.nn.Module):
    def __init__(
        self, steerer, 
        affine_augs,
        normalize=False,
        inv_temp=1., threshold=0.,
        similarity=scalar_product_similarity,
        compensate_aspect_ratio=False,
        randomized=False,
        nbr_randomized=2,
    ):
        super().__init__()
        self.steerer = steerer
        self.normalize = normalize
        self.inv_temp = inv_temp
        self.threshold = threshold
        self.similarity = similarity
        self.affine_augs = affine_augs
        self.compensate_aspect_ratio = compensate_aspect_ratio
        self.randomized = randomized
        self.nbr_randomized = nbr_randomized

    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B,
              P_A = None, P_B = None,
              H_A=None, W_A=None, H_B=None, W_B=None,
             ):
        if isinstance(descriptions_A, list):
            matches = [self.match(k_A[None], d_A[None], k_B[None], d_B[None], normalize = normalize,
                                  inv_temp = inv_temp, threshold = threshold) 
                       for k_A,d_A,k_B,d_B in
                       zip(keypoints_A, descriptions_A, keypoints_B, descriptions_B)]
            matches_A = torch.cat([m[0] for m in matches])
            matches_B = torch.cat([m[1] for m in matches])
            inds = torch.cat([m[2] + b for b, m in enumerate(matches)])
            return matches_A, matches_B, inds

        if self.compensate_aspect_ratio:
            if H_A < W_A:
                descriptions_A = self.steerer(
                    descriptions_A,
                    build_affine(
                        angle_1=0.,
                        dilation_1=W_A/H_A,
                        dilation_2=1.,
                        angle_2=0.,
                    ).to(descriptions_A.device)
                )
            else:
                descriptions_A = self.steerer(
                    descriptions_A,
                    build_affine(
                        angle_1=0.,
                        dilation_1=1.,
                        dilation_2=H_A/W_A,
                        angle_2=0.,
                    ).to(descriptions_A.device)
                )
            if H_B < W_B:
                descriptions_B = self.steerer(
                    descriptions_B,
                    build_affine(
                        angle_1=0.,
                        dilation_1=W_B/H_B,
                        dilation_2=1.,
                        angle_2=0.,
                    ).to(descriptions_B.device)
                )
            else:
                descriptions_B = self.steerer(
                    descriptions_B,
                    build_affine(
                        angle_1=0.,
                        dilation_1=1.,
                        dilation_2=H_B/W_B,
                        angle_2=0.,
                    ).to(descriptions_B.device)
                )

        B, N_A, _ = descriptions_A.shape
        _, N_B, _ = descriptions_B.shape
        corr = self.similarity(descriptions_A, descriptions_B)
        if self.randomized:
            for _ in range(self.nbr_randomized):
                if self.steerer.use_prototype_affines:
                    descriptions_A_aug = self.steerer.steer_with_random_prototypes(
                        descriptions_A)
                else:
                    prototype_ids = np.random.choice(
                        len(self.affine_augs),
                        size=descriptions_A.shape[1],
                        replace=True,
                    )
                    descriptions_A_aug = self.steerer(
                        descriptions_A,
                        torch.stack(
                            [self.affine_augs[i] for i in prototype_ids],
                            dim=0,
                        )[None],
                    )

                corr = torch.maximum(
                    corr,
                    self.similarity(
                        descriptions_A_aug,
                        descriptions_B,
                    ),
                )
        else:
            for aff in self.affine_augs:
                descriptions_A_aug = self.steerer(descriptions_A, aff[None, None])
                corr = torch.maximum(corr, self.similarity(descriptions_A_aug,
                                                           descriptions_B))

                if not torch.all(torch.isclose(aff, torch.eye(2, device=aff.device))):
                    descriptions_B_aug = self.steerer(descriptions_B, aff[None, None])

                    corr = torch.maximum(corr, self.similarity(descriptions_A,
                                                               descriptions_B_aug))

        corr = self.inv_temp * corr
        P = corr.softmax(dim = -2) * corr.softmax(dim= -1)

        inds = torch.nonzero(
            (P == P.max(dim=-1, keepdim = True).values) 
            * (P == P.max(dim=-2, keepdim = True).values) 
            * (P > self.threshold)
        )

        matches_A = keypoints_A[inds[:, 0], inds[:, 1]]
        matches_B = keypoints_B[inds[:, 0], inds[:, 2]]
        return matches_A, matches_B, inds

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)

    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)

class SubsetMaxMatchesTTADualSoftMaxMatcher(torch.nn.Module):
    def __init__(
        self, steerer, 
        affine_augs,
        normalize=False,
        inv_temp=1., threshold=0.,
        similarity=scalar_product_similarity,
        compensate_aspect_ratio=False,
        subset_size=5000,
    ):
        super().__init__()
        self.steerer = steerer
        self.normalize = normalize
        self.inv_temp = inv_temp
        self.threshold = threshold
        self.similarity = similarity
        self.affine_augs = affine_augs
        self.compensate_aspect_ratio = compensate_aspect_ratio
        self.subset_size = subset_size

    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B,
              P_A = None, P_B = None,
              H_A=None, W_A=None, H_B=None, W_B=None,
             ):
        if isinstance(descriptions_A, list):
            matches = [self.match(k_A[None], d_A[None], k_B[None], d_B[None], normalize = normalize,
                                  inv_temp = inv_temp, threshold = threshold) 
                       for k_A,d_A,k_B,d_B in
                       zip(keypoints_A, descriptions_A, keypoints_B, descriptions_B)]
            matches_A = torch.cat([m[0] for m in matches])
            matches_B = torch.cat([m[1] for m in matches])
            inds = torch.cat([m[2] + b for b, m in enumerate(matches)])
            return matches_A, matches_B, inds

        if self.compensate_aspect_ratio:
            if H_A < W_A:
                descriptions_A = self.steerer(
                    descriptions_A,
                    build_affine(
                        angle_1=0.,
                        dilation_1=W_A/H_A,
                        dilation_2=1.,
                        angle_2=0.,
                    ).to(descriptions_A.device)
                )
            else:
                descriptions_A = self.steerer(
                    descriptions_A,
                    build_affine(
                        angle_1=0.,
                        dilation_1=1.,
                        dilation_2=H_A/W_A,
                        angle_2=0.,
                    ).to(descriptions_A.device)
                )
            if H_B < W_B:
                descriptions_B = self.steerer(
                    descriptions_B,
                    build_affine(
                        angle_1=0.,
                        dilation_1=W_B/H_B,
                        dilation_2=1.,
                        angle_2=0.,
                    ).to(descriptions_B.device)
                )
            else:
                descriptions_B = self.steerer(
                    descriptions_B,
                    build_affine(
                        angle_1=0.,
                        dilation_1=1.,
                        dilation_2=H_B/W_B,
                        angle_2=0.,
                    ).to(descriptions_B.device)
                )

        descriptions_A_subset = descriptions_A[:, ::descriptions_A.shape[1]//self.subset_size]
        descriptions_B_subset = descriptions_B[:, ::descriptions_B.shape[1]//self.subset_size]

        best_aff = torch.eye(2, device=descriptions_A.device)
        best_nbr_matches = 0
        print("==== NEW PAIR ====")
        for aff in self.affine_augs:
            descriptions_A_aug = self.steerer(descriptions_A_subset, aff[None, None])

            P = dual_softmax_matcher(
                descriptions_A_aug, descriptions_B_subset, 
                normalize=self.normalize, inv_temperature=self.inv_temp,
                similarity=self.similarity,
            )

            inds = torch.nonzero(
                (P == P.max(dim=-1, keepdim = True).values) 
                * (P == P.max(dim=-2, keepdim = True).values) 
                * (P > self.threshold)
            )

            if inds.shape[0] > best_nbr_matches:
                if best_nbr_matches == 0:
                    print("unsteered")
                    print(f"    nbr matches: {inds.shape[0]}")
                else:
                    print("new best (from steering A)")
                    print(f"    M = {aff.cpu().numpy()}")
                    print(f"    nbr matches: {inds.shape[0]}")
                best_aff = aff
                best_nbr_matches = inds.shape[0]


        descriptions_A = self.steerer(descriptions_A, best_aff[None, None])

        P = dual_softmax_matcher(
            descriptions_A, descriptions_B, 
            normalize=self.normalize, inv_temperature=self.inv_temp,
            similarity=self.similarity,
        )
        best_inds = torch.nonzero(
            (P == P.max(dim=-1, keepdim = True).values) 
            * (P == P.max(dim=-2, keepdim = True).values) 
            * (P > self.threshold)
        )

        matches_A = keypoints_A[best_inds[:, 0], best_inds[:, 1]]
        matches_B = keypoints_B[best_inds[:, 0], best_inds[:, 2]]
        return matches_A, matches_B, best_inds

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)

    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)
class MaxMatchesTTADualSoftMaxMatcher(torch.nn.Module):
    def __init__(
        self, steerer, 
        affine_augs,
        normalize=False,
        inv_temp=1., threshold=0.,
        similarity=scalar_product_similarity,
        compensate_aspect_ratio=False,
    ):
        super().__init__()
        self.steerer = steerer
        self.normalize = normalize
        self.inv_temp = inv_temp
        self.threshold = threshold
        self.similarity = similarity
        self.affine_augs = affine_augs
        self.compensate_aspect_ratio = compensate_aspect_ratio

    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B,
              P_A = None, P_B = None,
              H_A=None, W_A=None, H_B=None, W_B=None,
             ):
        if isinstance(descriptions_A, list):
            matches = [self.match(k_A[None], d_A[None], k_B[None], d_B[None], normalize = normalize,
                                  inv_temp = inv_temp, threshold = threshold) 
                       for k_A,d_A,k_B,d_B in
                       zip(keypoints_A, descriptions_A, keypoints_B, descriptions_B)]
            matches_A = torch.cat([m[0] for m in matches])
            matches_B = torch.cat([m[1] for m in matches])
            inds = torch.cat([m[2] + b for b, m in enumerate(matches)])
            return matches_A, matches_B, inds

        if self.compensate_aspect_ratio:
            if H_A < W_A:
                descriptions_A = self.steerer(
                    descriptions_A,
                    build_affine(
                        angle_1=0.,
                        dilation_1=W_A/H_A,
                        dilation_2=1.,
                        angle_2=0.,
                    ).to(descriptions_A.device)
                )
            else:
                descriptions_A = self.steerer(
                    descriptions_A,
                    build_affine(
                        angle_1=0.,
                        dilation_1=1.,
                        dilation_2=H_A/W_A,
                        angle_2=0.,
                    ).to(descriptions_A.device)
                )
            if H_B < W_B:
                descriptions_B = self.steerer(
                    descriptions_B,
                    build_affine(
                        angle_1=0.,
                        dilation_1=W_B/H_B,
                        dilation_2=1.,
                        angle_2=0.,
                    ).to(descriptions_B.device)
                )
            else:
                descriptions_B = self.steerer(
                    descriptions_B,
                    build_affine(
                        angle_1=0.,
                        dilation_1=1.,
                        dilation_2=H_B/W_B,
                        angle_2=0.,
                    ).to(descriptions_B.device)
                )

        best_inds = torch.empty([0, 3],
                                device=descriptions_A.device,
                                dtype=torch.long)
        print("==== NEW PAIR ====")
        for aff in self.affine_augs:
            descriptions_A_aug = self.steerer(descriptions_A, aff[None, None])

            P = dual_softmax_matcher(
                descriptions_A_aug, descriptions_B, 
                normalize=self.normalize, inv_temperature=self.inv_temp,
                similarity=self.similarity,
            )

            inds = torch.nonzero(
                (P == P.max(dim=-1, keepdim = True).values) 
                * (P == P.max(dim=-2, keepdim = True).values) 
                * (P > self.threshold)
            )

            if inds.shape[0] > best_inds.shape[0]:
                if best_inds.shape[0] == 0:
                    print("unsteered")
                    print(f"    nbr matches: {inds.shape[0]}")
                else:
                    print("new best (from steering A)")
                    print(f"    M = {aff.cpu().numpy()}")
                    print(f"    nbr matches: {inds.shape[0]}")
                best_inds = inds

            if not torch.all(torch.isclose(aff, torch.eye(2, device=aff.device))):
                descriptions_B_aug = self.steerer(descriptions_B, aff[None, None])

                P = dual_softmax_matcher(
                    descriptions_A, descriptions_B_aug, 
                    normalize=self.normalize, inv_temperature=self.inv_temp,
                    similarity=self.similarity,
                )

                inds = torch.nonzero(
                    (P == P.max(dim=-1, keepdim = True).values) 
                    * (P == P.max(dim=-2, keepdim = True).values) 
                    * (P > self.threshold)
                )

                if inds.shape[0] > best_inds.shape[0]:
                    print("new best (from steering B)")
                    print(f"    M = {aff.cpu().numpy()}")
                    print(f"    nbr matches: {inds.shape[0]}")
                    best_inds = inds

        matches_A = keypoints_A[best_inds[:, 0], best_inds[:, 1]]
        matches_B = keypoints_B[best_inds[:, 0], best_inds[:, 2]]
        return matches_A, matches_B, best_inds

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)

    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)

class AffineDualSoftMaxMatcher(torch.nn.Module):
    def __init__(
        self, steerer, normalize=False,
        inv_temp=1., threshold=0.,
        similarity=scalar_product_similarity,
        remove_negative_determinants=False,
        use_reference_direction=False,
        sing_value_cutoff=3.,
    ):
        super().__init__()
        self.steerer = steerer
        self.normalize = normalize
        self.inv_temp = inv_temp
        self.threshold = threshold
        self.similarity = similarity
        self.remove_negative_determinants = remove_negative_determinants
        self.use_reference_direction = use_reference_direction
        self.sing_value_cutoff = sing_value_cutoff

    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B,
              P_A = None, P_B = None):
        if isinstance(descriptions_A, list):
            matches = [self.match(k_A[None], d_A[None], k_B[None], d_B[None], normalize = normalize,
                                  inv_temp = inv_temp, threshold = threshold) 
                       for k_A,d_A,k_B,d_B in
                       zip(keypoints_A, descriptions_A, keypoints_B, descriptions_B)]
            matches_A = torch.cat([m[0] for m in matches])
            matches_B = torch.cat([m[1] for m in matches])
            inds = torch.cat([m[2] + b for b, m in enumerate(matches)])
            return matches_A, matches_B, inds

        if self.use_reference_direction:
            est_A_to_id = self.steerer.estimate_affine(descriptions_A, to_ref=True)
            descriptions_A = self.steerer(descriptions_A, est_A_to_id)
            est_B_to_id = self.steerer.estimate_affine(descriptions_B, to_ref=True)
            descriptions_B = self.steerer(descriptions_B, est_B_to_id)
            P = dual_softmax_matcher(
                descriptions_A, descriptions_B, 
                normalize=self.normalize, inv_temperature=self.inv_temp,
                similarity=self.similarity,
            )
        else:
            P = affine_dual_softmax_matcher(
                descriptions_A, descriptions_B, 
                steerer=self.steerer,
                normalize=self.normalize, inv_temperature=self.inv_temp,
                similarity=self.similarity,
                remove_negative_det=self.remove_negative_determinants,
            )

        inds = torch.nonzero(
            (P == P.max(dim=-1, keepdim = True).values) 
            * (P == P.max(dim=-2, keepdim = True).values) 
            * (P > self.threshold)
        )

        if self.use_reference_direction:
            est_A_to_B_matches = (
                fast_inv_2x2(est_B_to_id[inds[:, 0], inds[:, 2]])
                @ est_A_to_id[inds[:, 0], inds[:, 1]]
            )
            sv = torch.linalg.svdvals(est_A_to_B_matches)
            good_est = (
                torch.isfinite(sv.sum(dim=-1))
                * (sv.max(dim=-1).values < self.sing_value_cutoff)
                * (sv.min(dim=-1).values > 1. / self.sing_value_cutoff)
            )
            if self.remove_negative_determinants:
                good_est *= (torch.linalg.det(est_A_to_B_matches) > 0)
            inds = inds[good_est]

        matches_A = keypoints_A[inds[:, 0], inds[:, 1]]
        matches_B = keypoints_B[inds[:, 0], inds[:, 2]]
        return matches_A, matches_B, inds

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)

    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)
