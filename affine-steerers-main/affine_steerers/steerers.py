import torch
import torch.nn as nn
import math
import numpy as np
from affine_steerers.utils import get_reference_desc, lstsq_affine, affine_from_five_params


class Steerer1(nn.Module):
    """ Steerer with only affine order 1 blocks (so steerer is block-diagonal with 2x2 affine matrices). """

    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize

    def forward(self, features, affine):
        """
        Input:
            features: batch, kpts, feat_dim
            affine: batch, kpts, 2, 2
        Output:
            features: batch, kpts, feat_dim
        """
        if self.normalize:
            affine = affine / affine.det().abs().sqrt().clamp(min=1e-8)[..., None, None]

        b, k, f = features.shape
        features = features.reshape(b, k, f//2, 2)
        features = features @ affine.mT
        features = features.reshape(b, k, f)
        return features


class SteererSpread(nn.Module):
    """ Steerer with equal amount n=[0, 1, ..., N] dimensions, n is the order of each affine representation. """

    def __init__(
        self,
        feat_dim=256,
        max_order=4,
        block_diag_rot=True,
        block_diag_optimal_scalings=False,
        learnable_basis=False,
        contragredient_rep1=False,
        normalize=False,
        normalize_only_higher=False,
        learnable_determinant_scaling=False,
        fix_order_1_scalings=True,
        max_determinant_scaling=None,
        learnable_reference_direction=False,
        scale_reference_direction=1.,
        separate_affine_estimator=False,
        separate_affine_feat=False,
        learnable_lstsq_weights=False,
        lstsq_weights_from_feat0=True,
        use_prototype_affines=False,
        prototype_affines_init=None,
        learnable_prototype_affines=True,
    ):
        super().__init__()
        if max_order > 4:
            raise NotImplementedError("Currently only supporting affine steerer order up to 4")
        if normalize_only_higher and normalize:
            raise ValueError()
        if block_diag_optimal_scalings and not block_diag_rot:
            raise ValueError()
        if learnable_basis and block_diag_rot:
            raise ValueError()
        if separate_affine_estimator and separate_affine_feat:
            raise ValueError()
        if use_prototype_affines and prototype_affines_init is None:
            raise ValueError()
        self.feat_dim = feat_dim
        self.max_order = max_order
        self.block_diag_rot = block_diag_rot
        self.block_diag_optimal_scalings = block_diag_optimal_scalings
        self.learnable_basis = learnable_basis
        self.contragredient_rep1 = contragredient_rep1
        self.normalize = normalize
        self.normalize_only_higher = normalize_only_higher
        self.learnable_determinant_scaling = learnable_determinant_scaling
        self.fix_order_1_scalings = fix_order_1_scalings
        self.max_determinant_scaling = max_determinant_scaling
        self.learnable_reference_direction = learnable_reference_direction
        self.scale_reference_direction = scale_reference_direction
        self.separate_affine_estimator = separate_affine_estimator
        self.separate_affine_feat = separate_affine_feat
        self.learnable_lstsq_weights = learnable_lstsq_weights
        self.lstsq_weights_from_feat0 = lstsq_weights_from_feat0
        self.use_prototype_affines = use_prototype_affines
        self.prototype_affines_init = prototype_affines_init
        self.learnable_prototype_affines = learnable_prototype_affines

        self.feat_dim_per_n = [
            (n+1) * (self.feat_dim // ((n+1) * (self.max_order+1)))
            for n in range(1, self.max_order+1)]
        self.feat_dim_per_n = (
            [self.feat_dim - sum(self.feat_dim_per_n)] 
            + self.feat_dim_per_n
        )
        self.reps_per_n = [self.feat_dim_per_n[n] // (n+1) for n in range(self.max_order+1)]

        self.affine_reps = [
            None, # 0 order affine rep is identity map
            affine_rep1_contragredient if self.contragredient_rep1 else affine_rep1,
        ]
        if self.block_diag_rot:
            if self.block_diag_optimal_scalings:
                self.affine_reps += [
                    affine_rep2_bd,
                    affine_rep3_bd_opt,
                    affine_rep4_bd_opt,
                ]
            else:
                self.affine_reps += [
                    affine_rep2_bd,
                    affine_rep3_bd,
                    affine_rep4_bd,
                ]
        else:
            self.affine_reps += [
                affine_rep2,
                affine_rep3,
                affine_rep4,
            ]
        self.affine_reps = self.affine_reps[:self.max_order+1]

        if self.learnable_basis:
            self.basis_transpose = torch.nn.Parameter(
                torch.eye(feat_dim),
                requires_grad=True,
            )

        if self.learnable_determinant_scaling:
            self.determinant_scalings = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.zeros(self.reps_per_n[n]),
                        requires_grad=(
                            not (self.fix_order_1_scalings and n == 1)
                        )
                    )
                    for n in range(self.max_order+1)
                ]
            )

        if self.separate_affine_estimator:
            self.proj_to_est_affine = torch.nn.Sequential(
                torch.nn.Linear(self.feat_dim, self.feat_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(self.feat_dim, self.feat_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(self.feat_dim, 5),
            )
        else:
            if self.separate_affine_feat:
                self.proj_to_est_affine = torch.nn.Linear(self.feat_dim, 20, bias=False)
                self.reference_direction = torch.nn.Parameter(
                    self.scale_reference_direction * get_reference_desc(10),
                    requires_grad=self.learnable_reference_direction,
                )
            else:
                self.reference_direction = torch.nn.Parameter(
                    self.scale_reference_direction * get_reference_desc(self.reps_per_n[1]),
                    requires_grad=self.learnable_reference_direction,
                )

        if self.learnable_lstsq_weights:
            if lstsq_weights_from_feat0:
                self.feat_to_lstsq_weights = torch.nn.Sequential(
                    torch.nn.Linear(
                        self.reps_per_n[0],
                        self.reference_direction.shape[0]),
                    torch.nn.Sigmoid(),
                )
            else:
                self.feat_to_lstsq_weights = torch.nn.Sequential(
                    torch.nn.Linear(
                        self.feat_dim,
                        self.feat_dim,
                    ),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(
                        self.feat_dim,
                        self.feat_dim,
                    ),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(
                        self.feat_dim,
                        self.reference_direction.shape[0],
                    ),
                    torch.nn.Sigmoid(),
                )

        if self.use_prototype_affines:
            # [N, 2, 2]
            self.prototype_affines = torch.nn.Parameter(
                torch.stack(
                    self.prototype_affines_init,
                    dim=0,
                ),
                requires_grad=self.learnable_prototype_affines,
            )


    def steer_with_random_prototypes(self, features):
        # features: [B, K, C]
        prototype_ids = np.random.choice(self.prototype_affines.shape[0],
                                         size=features.shape[1],
                                         replace=True)
        return self(features, self.prototype_affines[None, prototype_ids])


    def estimate_affine(self, features, to_ref=False):
        if self.separate_affine_estimator:
            return affine_from_five_params(
                self.proj_to_est_affine(features),
                to_ref = to_ref,
            )
        if self.separate_affine_feat:
            feat0 = self.split_into_orders(features)[0]
            feat1 = self.proj_to_est_affine(features)
        else:
            feat0, feat1 = self.split_into_orders(features)[:2]

        lstsq_weights = None
        if self.lstsq_weights_from_feat0 and self.learnable_lstsq_weights:
            lstsq_weights = self.feat_to_lstsq_weights(feat0)
        elif self.learnable_lstsq_weights:
            lstsq_weights = self.feat_to_lstsq_weights(features)

        shape = feat1.shape

        if len(shape) == 3:
            feat1 = feat1.reshape(shape[0], shape[1], -1, 2)
            reference_direction = self.reference_direction[None, None]
        elif len(shape) == 2:
            feat1 = feat1.reshape(shape[0], -1, 2)
            reference_direction = self.reference_direction[None]
        elif len(shape) == 1:
            feat1 = feat1.reshape(-1, 2)

        if self.contragredient_rep1:
            if to_ref:
                return lstsq_affine(
                    reference_direction,
                    feat1,
                    weights=lstsq_weights,
                ).mT
            return lstsq_affine(
                feat1,
                reference_direction,
                weights=lstsq_weights,
            ).mT

        if to_ref:
            return lstsq_affine(
                feat1,
                reference_direction,
                weights=lstsq_weights,
            )
        return lstsq_affine(
            reference_direction,
            feat1,
            weights=lstsq_weights,
        )

    def determinant_steer(self, features, order, affine, sqrt_det=None):
        """ WARNING: returns reshaped features. """
        shape = features.shape
        if len(shape) == 3:
            b, k, f = shape
            features = features.reshape(b, k, self.reps_per_n[order], order+1)
        elif len(shape) == 4:
            b, k, k2, f = shape
            features = features.reshape(b, k, k2, self.reps_per_n[order], order+1)
        else:
            raise ValueError()

        if not self.learnable_determinant_scaling:
            return features

        if sqrt_det is None:
            sqrt_det = affine.det().abs().sqrt().clamp(min=1e-8)[..., None, None]
        if self.max_determinant_scaling is not None:
            features = features* (
                sqrt_det ** (
                    2 * self.max_determinant_scaling
                    * torch.tanh(
                        self.determinant_scalings[order][None, None, :, None]
                        if len(shape) == 3
                        else self.determinant_scalings[order][None, None, None, :, None]
                    )
                )
            )
        else:
            features = features * (
                sqrt_det ** (
                    2 * self.determinant_scalings[order][None, None, :, None]
                    if len(shape) == 3
                    else 2 * self.determinant_scalings[order][None, None, None, :, None]
                )
            )
        return features

    def steer(self, features, order, affine, sqrt_det=None):
        # order=0 features are kept constant (except potential determinant scaling),
        # remaining are steered

        if sqrt_det is None and (
            self.normalize
            or self.normalize_only_higher
            or self.learnable_determinant_scaling
        ):
            sqrt_det = affine.det().abs().sqrt().clamp(min=1e-8)[..., None, None]

        b, k, _ = features.shape

        features = self.determinant_steer(features,
                                          order,
                                          affine,
                                          sqrt_det)

        if order == 0:
            return features.reshape(b, k, -1)

        if self.normalize or (self.normalize_only_higher and order > 1):
            features = features @ self.affine_reps[order](
                affine/sqrt_det, transpose=True)
        else:
            features = features @ self.affine_reps[order](
                affine, transpose=True)
        return features.reshape(b, k, -1)

    def split_into_orders(self, features):
        if self.learnable_basis:
            features = features @ self.basis_transpose
        return list(
            features.tensor_split(
                [sum(self.feat_dim_per_n[:n+1]) for n in range(self.max_order)],
                dim=-1,
            )
        )
        
    def forward(self, features, affine):
        """
        Input:
            features: batch, kpts, feat_dim
            affine: batch, kpts, 2, 2
        Output:
            features: batch, kpts, feat_dim
        """
        feature_list = self.split_into_orders(features)

        sqrt_det = None
        if (
            self.normalize
            or self.normalize_only_higher
            or self.learnable_determinant_scaling
        ):
            sqrt_det = affine.det().abs().sqrt().clamp(min=1e-8)[..., None, None]

        for n in range(self.max_order + 1):
            feature_list[n] = self.steer(feature_list[n], n, affine, sqrt_det)

        features = torch.cat(feature_list, dim=2)
        if self.learnable_basis:
            features = features @ torch.linalg.inv(self.basis_transpose)
        return features


def affine_rep1(affine, transpose=False):
    if transpose:
        return affine.mT
    return affine


def affine_rep1_contragredient(affine, transpose=False):
    a, b, c, d = (
        affine[..., 0, 0], affine[..., 0, 1],
        affine[..., 1, 0], affine[..., 1, 1],
    )
    det = a*d - b*c
    det_sign = det.sign()
    one_over_det = det.sign() / det.abs().clamp(min=1e-6)
    return one_over_det[..., None, None] * torch.stack([
        torch.stack([
            d, -c,
        ], dim=-1),
        torch.stack([
            -b, a,
        ], dim=-1),
    ], dim=-1 if transpose else -2)


def affine_rep2(affine, transpose=False):
    """ Gives 3 dim rep with basis as in Olver, i.e. not orthogonal for rotations. """
    a, b, c, d = (
        affine[..., 0, 0], affine[..., 0, 1],
        affine[..., 1, 0], affine[..., 1, 1],
    )
    return torch.stack([
        torch.stack([
            d**2, d*c, c**2,
        ], dim=-1),
        torch.stack([
            2*b*d, a*d + b*c, 2*a*c,
        ], dim=-1),
        torch.stack([
            b**2, a*b, a**2,
        ], dim=-1),
    ], dim=-1 if transpose else -2)


def affine_rep3(affine, transpose=False):
    """ Gives 4 dim rep with basis as in Olver, i.e. not orthogonal for rotations. """
    a, b, c, d = (
        affine[..., 0, 0], affine[..., 0, 1],
        affine[..., 1, 0], affine[..., 1, 1],
    )
    return torch.stack([
        torch.stack([
            d**3, c*d**2, c**2*d, c**3,
        ], dim=-1),
        torch.stack([
            3*b*d**2, d*(a*d + 2*b*c), c*(2*a*d + b*c), 3*a*c**2,
        ], dim=-1),
        torch.stack([
            3*b**2*d, b*(2*a*d + b*c), a*(a*d + 2*b*c), 3*a**2*c,
        ], dim=-1),
        torch.stack([
            b**3, a*b**2, a**2*b, a**3,
        ], dim=-1),
    ], dim=-1 if transpose else -2)


def affine_rep4(affine, transpose=False):
    """ Gives 5 dim rep with basis as in Olver, i.e. not orthogonal for rotations. """
    a, b, c, d = (
        affine[..., 0, 0], affine[..., 0, 1],
        affine[..., 1, 0], affine[..., 1, 1],
    )
    return torch.stack([
        torch.stack([
            d**4, d**3*c, d**2*c**2, d*c**3, c**4,
        ], dim=-1),
        torch.stack([
            4*b*d**3, d**2*(a*d + 3*b*c), 2*d*c*(a*d + b*c), c**2*(3*a*d + b*c), 4*a*c**3,
        ], dim=-1),
        torch.stack([
            6*b**2*d**2, 3*b*d*(a*d + b*c), a**2*d**2 + 4*a*b*d*c + b**2*c**2, 3*a*c*(a*d + b*c), 6*a**2*c**2,
        ], dim=-1),
        torch.stack([
            4*b**3*d, b**2*(3*a*d + b*c), 2*a*b*(a*d + b*c), a**2*(a*d + 3*b*c), 4*a**3*c,
        ], dim=-1),
        torch.stack([
            b**4, a*b**3, a**2*b**2, a**3*b, a**4,
        ], dim=-1),
    ], dim=-1 if transpose else -2)


def affine_rep2_bd(affine, transpose=False, scale0=1.):
    """ Gives 3 dim rep with basis such that rotations are block-diagonal. """
    a, b, c, d = (
        affine[..., 0, 0], affine[..., 0, 1],
        affine[..., 1, 0], affine[..., 1, 1],
    )
    a2, b2, c2, d2 = a**2, b**2, c**2, d**2
    ab, ac, ad, bc, bd, cd = a*b, a*c, a*d, b*c, b*d, c*d
    return 0.5 * torch.stack([
        torch.stack([
            a2 + b2 + c2 + d2, scale0*2*(ab + cd), scale0*(-a2 + b2 - c2 + d2),
        ], dim=-1),
        torch.stack([
            2*(ac + bd)/scale0, 2*(ad + bc), 2*(-ac + bd),
        ], dim=-1),
        torch.stack([
            (-a2 - b2 + c2 + d2)/scale0, 2*(-ab + cd), a2 - b2 - c2 + d2,
        ], dim=-1),
    ], dim=-1 if transpose else -2)


def affine_rep3_bd(affine, transpose=False, scale1=1.):
    """ Gives 4 dim rep with basis such that rotations are block-diagonal. """
    a, b, c, d = (
        affine[..., 0, 0], affine[..., 0, 1],
        affine[..., 1, 0], affine[..., 1, 1],
    )
    return torch.stack([
        torch.stack([
            3*a**3/4 + 3*a*b**2/4 + 3*a*c**2/4 + d*(a*d + 2*b*c)/4,
            3*a**2*b/4 + 3*b**3/4 + 3*b*d**2/4 + c*(2*a*d + b*c)/4,
            scale1*(3*a**3/4 - 9*a*b**2/4 + 3*a*c**2/4 - 3*d*(a*d + 2*b*c)/4),
            scale1*(9*a**2*b/4 - 3*b**3/4 - 3*b*d**2/4 + 3*c*(2*a*d + b*c)/4),
        ], dim=-1),
        torch.stack([
            3*a**2*c/4 + b*(2*a*d + b*c)/4 + 3*d**2*c/4 + 3*c**3/4,
            a*(a*d + 2*b*c)/4 + 3*b**2*d/4 + 3*d**3/4 + 3*d*c**2/4,
            scale1*(3*a**2*c/4 - 3*b*(2*a*d + b*c)/4 - 9*d**2*c/4 + 3*c**3/4),
            scale1*(3*a*(a*d + 2*b*c)/4 - 3*b**2*d/4 - 3*d**3/4 + 9*d*c**2/4),
        ], dim=-1),
        torch.stack([
            (a**3/4 + a*b**2/4 - 3*a*c**2/4 - d*(a*d + 2*b*c)/4)/scale1,
            (a**2*b/4 + b**3/4 - 3*b*d**2/4 - c*(2*a*d + b*c)/4)/scale1,
            a**3/4 - 3*a*b**2/4 - 3*a*c**2/4 + 3*d*(a*d + 2*b*c)/4,
            3*a**2*b/4 - b**3/4 + 3*b*d**2/4 - 3*c*(2*a*d + b*c)/4,
        ], dim=-1),
        torch.stack([
            (3*a**2*c/4 + b*(2*a*d + b*c)/4 - d**2*c/4 - c**3/4)/scale1,
            (a*(a*d + 2*b*c)/4 + 3*b**2*d/4 - d**3/4 - d*c**2/4)/scale1,
            3*a**2*c/4 - 3*b*(2*a*d + b*c)/4 + 3*d**2*c/4 - c**3/4,
            3*a*(a*d + 2*b*c)/4 - 3*b**2*d/4 + d**3/4 - 3*d*c**2/4,
        ], dim=-1),
    ], dim=-1 if transpose else -2)


def affine_rep4_bd(affine, transpose=False, scale0=1., scale2=1.):
    """ Gives 5 dim rep with basis such that rotations are block-diagonal. """
    a, b, c, d = (
        affine[..., 0, 0], affine[..., 0, 1],
        affine[..., 1, 0], affine[..., 1, 1],
    )
    return torch.stack([
        torch.stack([
            (3*a**4/8 + 3*a**2*b**2/4 + a**2*d**2/4 + 3*a**2*c**2/4
            + a*b*d*c + 3*b**4/8 + 3*b**2*d**2/4 + b**2*c**2/4
            + 3*d**4/8 + 3*d**2*c**2/4 + 3*c**4/8),
            (3*a**4/8 + 3*a**2*c**2/4 - 3*b**4/8
            - 3*b**2*d**2/4 - 3*d**4/8 + 3*c**4/8)*(scale0/scale2),
            (3*a**3*b/4 + 3*a*b**3/4 + 3*a*c*(a*d + b*c)/4
            + 3*b*d*(a*d + b*c)/4 + 3*d**3*c/4 + 3*d*c**3/4)*(scale0/scale2),
            (3*a**4/8 - 9*a**2*b**2/4 - 3*a**2*d**2/4 + 3*a**2*c**2/4
            - 3*a*b*d*c + 3*b**4/8 + 3*b**2*d**2/4 - 3*b**2*c**2/4
            + 3*d**4/8 - 9*d**2*c**2/4 + 3*c**4/8)*scale0,
            (3*a**3*b/2 - 3*a*b**3/2 + 3*a*c*(a*d + b*c)/2
            - 3*b*d*(a*d + b*c)/2 - 3*d**3*c/2 + 3*d*c**3/2)*scale0,
        ], dim=-1),
        torch.stack([
            (a**4/2 + a**2*b**2 + b**4/2 - d**4/2 - d**2*c**2 - c**4/2)*(scale2/scale0),
            a**4/2 - b**4/2 + d**4/2 - c**4/2,
            a**3*b + a*b**3 - d**3*c - d*c**3,
            (a**4/2 - 3*a**2*b**2 + b**4/2 - d**4/2 + 3*d**2*c**2 - c**4/2)*scale2,
            (2*a**3*b - 2*a*b**3 + 2*d**3*c - 2*d*c**3)*scale2,
        ], dim=-1),
        torch.stack([
            (a**3*c + a*b*(a*d + b*c) + a*c**3 + b**3*d + b*d**3
            + d*c*(a*d + b*c))*(scale2/scale0),
            a**3*c + a*c**3 - b**3*d - b*d**3,
            (a**2*(a*d + 3*b*c)/2 + b**2*(3*a*d + b*c)/2
            + d**2*(a*d + 3*b*c)/2 + c**2*(3*a*d + b*c)/2),
            (a**3*c - 3*a*b*(a*d + b*c) + a*c**3 + b**3*d
            + b*d**3 - 3*d*c*(a*d + b*c))*scale2,
            (a**2*(a*d + 3*b*c) - b**2*(3*a*d + b*c)
            - d**2*(a*d + 3*b*c) + c**2*(3*a*d + b*c))*scale2,
        ], dim=-1),
        torch.stack([
            (a**4/8 + a**2*b**2/4 - a**2*d**2/4 - 3*a**2*c**2/4
            - a*b*d*c + b**4/8 - 3*b**2*d**2/4 - b**2*c**2/4
            + d**4/8 + d**2*c**2/4 + c**4/8)/scale0,
            (a**4/8 - 3*a**2*c**2/4 - b**4/8 + 3*b**2*d**2/4
            - d**4/8 + c**4/8)/scale2,
            (a**3*b/4 + a*b**3/4 - 3*a*c*(a*d + b*c)/4
            - 3*b*d*(a*d + b*c)/4 + d**3*c/4 + d*c**3/4)/scale2,
            (a**4/8 - 3*a**2*b**2/4 + 3*a**2*d**2/4 - 3*a**2*c**2/4
            + 3*a*b*d*c + b**4/8 - 3*b**2*d**2/4 + 3*b**2*c**2/4
            + d**4/8 - 3*d**2*c**2/4 + c**4/8),
            (a**3*b/2 - a*b**3/2 - 3*a*c*(a*d + b*c)/2
            + 3*b*d*(a*d + b*c)/2 - d**3*c/2 + d*c**3/2),
        ], dim=-1),
        torch.stack([
            (a**3*c/2 + a*b*(a*d + b*c)/2 - a*c**3/2 + b**3*d/2
            - b*d**3/2 - d*c*(a*d + b*c)/2)/scale0,
            (a**3*c/2 - a*c**3/2 - b**3*d/2 + b*d**3/2)/scale2,
            (a**2*(a*d + 3*b*c)/4 + b**2*(3*a*d + b*c)/4
            - d**2*(a*d + 3*b*c)/4 - c**2*(3*a*d + b*c)/4)/scale2,
            (a**3*c/2 - 3*a*b*(a*d + b*c)/2 - a*c**3/2 + b**3*d/2
            - b*d**3/2 + 3*d*c*(a*d + b*c)/2),
            (a**2*(a*d + 3*b*c)/2 - b**2*(3*a*d + b*c)/2 +
            d**2*(a*d + 3*b*c)/2 - c**2*(3*a*d + b*c)/2),
        ], dim=-1),
    ], dim=-1 if transpose else -2)


def affine_rep3_bd_opt(affine, transpose=False):
    """ Scales blocks 'optimally' in some sense... """
    return affine_rep3_bd(affine,
                          transpose=transpose,
                          scale1=1./math.sqrt(3))


def affine_rep4_bd_opt(affine, transpose=False):
    """ Scales blocks 'optimally' in some sense... """
    return affine_rep4_bd(affine,
                          transpose=transpose,
                          scale0=1./math.sqrt(3),
                          scale2=0.5)


def test_steerers():
    b = 3; k = 125; f = 256;
    device = 'cuda'

    def rotation_matrix(t):
        return torch.tensor([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    def sample_affine():
        angle_1 = np.random.uniform(high=2*np.pi)
        angle_2 = np.random.uniform(high=2*np.pi)
        dilation_1 = np.exp(np.random.uniform(low=np.log(0.75), high=np.log(2.)))
        dilation_2 = np.exp(np.random.uniform(low=np.log(0.75), high=np.log(2.)))
        if np.random.uniform() < .5:
            dilation_1 *= -1
        return (
            rotation_matrix(angle_2 - angle_1) 
            @ torch.diag(torch.tensor([dilation_1, dilation_2]))
            @ rotation_matrix(angle_1)
        )

    # affine_1 = torch.randn([b, k, 2, 2], device=device)
    # affine_2 = torch.randn([b, k, 2, 2], device=device)

    affine_1 = torch.zeros([b, k, 2, 2], device=device)
    affine_2 = torch.zeros([b, k, 2, 2], device=device)
    for i in range(b):
        for j in range(k):
            affine_1[i, j] = sample_affine().float().to(device)
            affine_2[i, j] = sample_affine().float().to(device)

    # affine_1 = torch.tensor([[math.cos(math.pi/4), -math.sin(math.pi/4)],
    #                          [math.sin(math.pi/4), math.cos(math.pi/4)]],
    #                         device=device)[None, None]
    # affine_2 = torch.tensor([[math.cos(math.pi/8), -math.sin(math.pi/8)],
    #                          [math.sin(math.pi/8), math.cos(math.pi/8)]],
    #                         device=device)[None, None]

    affine_2_1 = affine_2 @ affine_1
    features = torch.randn([b, k, f], device=device)

    for steerer, name in [
        (Steerer1(), "only 1"),
        (SteererSpread(max_order=1, block_diag_rot=False, normalize=False),
        "order 1"),
        (SteererSpread(max_order=2, block_diag_rot=False, normalize=False),
        "order 2"),
        (SteererSpread(max_order=3, block_diag_rot=False, normalize=False),
        "order 3"),
        (SteererSpread(max_order=4, block_diag_rot=False, normalize=False),
        "order 4"),
        (SteererSpread(max_order=1, block_diag_rot=True, normalize=False),
        "order 1, bd"),
        (SteererSpread(max_order=2, block_diag_rot=True, normalize=False),
        "order 2, bd"),
        (SteererSpread(max_order=3, block_diag_rot=True, normalize=False),
        "order 3, bd"),
        (SteererSpread(max_order=4, block_diag_rot=True, normalize=False),
        "order 4, bd"),
        (Steerer1(normalize=True), "only 1, norm"),
        (SteererSpread(max_order=1, block_diag_rot=False, normalize=True),
        "order 1, norm"),
        (SteererSpread(max_order=2, block_diag_rot=False, normalize=True),
        "order 2, norm"),
        (SteererSpread(max_order=3, block_diag_rot=False, normalize=True),
        "order 3, norm"),
        (SteererSpread(max_order=4, block_diag_rot=False, normalize=True),
        "order 4, norm"),
        (SteererSpread(max_order=4, block_diag_rot=False, normalize=False, normalize_only_higher=True),
        "order 4, norm high"),
        (SteererSpread(max_order=1, block_diag_rot=True, normalize=True),
        "order 1, bd, norm"),
        (SteererSpread(max_order=2, block_diag_rot=True, normalize=True),
        "order 2, bd, norm"),
        (SteererSpread(max_order=3, block_diag_rot=True, normalize=True),
        "order 3, bd, norm"),
        (SteererSpread(max_order=4, block_diag_rot=True, normalize=True),
        "order 4, bd, norm"),
        (SteererSpread(max_order=4, block_diag_rot=True, normalize=False, normalize_only_higher=True),
        "order 4, bd, norm high"),
        (SteererSpread(max_order=4,
                       block_diag_rot=True,
                       block_diag_optimal_scalings=True,
                       normalize=False),
        "order 4, bd opt"),
        (SteererSpread(max_order=4,
                       block_diag_rot=True,
                       block_diag_optimal_scalings=True,
                       normalize=True),
        "order 4, bd opt, norm"),
        (SteererSpread(max_order=4,
                       block_diag_rot=True,
                       block_diag_optimal_scalings=True,
                       normalize=False,
                       normalize_only_higher=True),
        "order 4, bd opt, norm high"),
        (SteererSpread(max_order=4,
                       block_diag_rot=True,
                       block_diag_optimal_scalings=True,
                       contragredient_rep1=True,
                       normalize=False,
                       normalize_only_higher=True),
        "order 4, bd opt, norm high, contagred"),
        (SteererSpread(max_order=4,
                       block_diag_rot=False,
                       block_diag_optimal_scalings=False,
                       contragredient_rep1=False,
                       learnable_basis=True,
                       learnable_determinant_scaling=True,
                       normalize=True,
                       normalize_only_higher=False),
        "order 4, learn base, norm high, contagred"),
    ]:
        if "learn base" in name:
            steerer.to(device)
            steerer.basis_transpose.data = torch.randn(steerer.basis_transpose.data.shape, device=device)
            for k in range(len(steerer.determinant_scalings)):
                steerer.determinant_scalings[k].data = torch.randn(steerer.determinant_scalings[k].data.shape, device=device)
        features_out_1 = steerer(features, affine_1)
        features_out_2 = steerer(features_out_1, affine_2)

        features_out = steerer(features, affine_2_1)

        feat_abs = features_out.abs()
        print(name)
        print("------------------------------")
        print(f"max feat: {feat_abs.max().item()}, median feat: {feat_abs.median().item()}")
        errs = (features_out_2 - features_out).abs()
        print(f"max err: {errs.max().item()}, median err: {errs.median().item()}")
        assert torch.allclose(features_out_2, features_out, atol=1e-2), breakpoint()
        print("==============================")
    print("Test passed!")


if __name__ == "__main__":
    test_steerers()
