import os
import math
import warnings
from argparse import ArgumentParser

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset
import torch.nn as nn

from affine_steerers.train import train_k_steps
from affine_steerers.datasets.megadepth import MegadepthBuilder
from affine_steerers.descriptors.descriptor_loss import DescriptorLoss
from affine_steerers.checkpoint import CheckPoint
from affine_steerers.descriptors.dedode_descriptor import affine_steerersDescriptor
from affine_steerers.encoder import VGG
from affine_steerers.decoder import ConvRefiner, Decoder
from affine_steerers import dedode_detector_L
from affine_steerers.benchmarks import MegadepthNLLBenchmark
from affine_steerers.steerers import Steerer1, SteererSpread
from affine_steerers.utils import (
    inverse_distance_similarity, gauss_similarity, scalar_product_similarity,
    negative_distance_similarity, get_reference_desc, build_affine
)
from affine_steerers.matchers.dual_softmax_matcher import DualSoftMaxMatcher


def train(pretrained_path, pretrained_name, detector_weights_path = "dedode_detector_L.pth"):
    warnings.filterwarnings("ignore", category=UserWarning)
    NUM_PROTOTYPES = 256 # == descriptor size
    residual = True
    hidden_blocks = 5
    amp_dtype = torch.float16#torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp = True
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,
            ),
            "4": ConvRefiner(
                256+256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,

            ),
            "2": ConvRefiner(
                128+128,
                64,
                32 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,

            ),
            "1": ConvRefiner(
                64 + 32,
                32,
                1 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,
            ),
        }
    )
    import os
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    encoder = VGG(size = "19", pretrained = True, amp = amp, amp_dtype = amp_dtype)
    decoder = Decoder(conv_refiner, num_prototypes=NUM_PROTOTYPES)
    model = affine_steerersDescriptor(encoder = encoder, decoder = decoder).cuda()

    steerer = SteererSpread(
        max_order=4,
        normalize=True,
        normalize_only_higher=False,
        fix_order_1_scalings=False,
        max_determinant_scaling=None,
        block_diag_rot=False,
        block_diag_optimal_scalings=False,
        learnable_determinant_scaling=True,
        learnable_basis=True,
        learnable_reference_direction=False,
        learnable_lstsq_weights=False,
        use_prototype_affines=True,
        prototype_affines_init=[
            build_affine(
                angle_1=0.,
                dilation_1=1.,
                dilation_2=1.,
                angle_2=-math.pi/8,
            ).to('cuda'),
            build_affine(
                angle_1=0.,
                dilation_1=1.,
                dilation_2=1.,
                angle_2=math.pi/8,
            ).to('cuda'),
        ]
    ).cuda()
    params = [
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": model.decoder.parameters(), "lr": 2e-4},
        {"params": steerer.parameters(), "lr": 1e-2},
    ]
    optim = AdamW(params, weight_decay = 1e-5)
    n0, N, k = 0, 50_000, 1000
    lr_scheduler = CosineAnnealingLR(optim, T_max = N)
    checkpointer = CheckPoint("workspace/", name = experiment_name)

    detector = dedode_detector_L(weights=torch.load(detector_weights_path), remove_borders=True)

    pretrained = CheckPoint(
        pretrained_path,
        name=pretrained_name,
    )
    model, _, _, _ = pretrained.load(
        model, None, None, None, steerer=steerer,
    )

    matcher = DualSoftMaxMatcher(
        normalize=False,
        inv_temp=5,
        threshold=0.01,
        similarity=negative_distance_similarity,
    )

    loss = DescriptorLoss(
        detector=detector,
        steerer=steerer,
        steer_kptwise=False,
        filter_inliers_with_mask=False,
        matcher=matcher,
    )

    H, W = 512, 512
    mega = MegadepthBuilder(data_root="data/megadepth",
                            loftr_ignore=True,
                            imc21_ignore=True)
    use_horizontal_flip_aug = False
    use_affine_aug = False

    megadepth_train1 = mega.build_scenes(
        split="train_loftr",
        min_overlap=0.01,
        ht=H,
        wt=W,
        shake_t=32,
        use_horizontal_flip_aug=use_horizontal_flip_aug,
        use_affine_aug=use_affine_aug,
    )
    megadepth_train2 = mega.build_scenes(
        split="train_loftr",
        min_overlap=0.35,
        ht=H,
        wt=W,
        shake_t=32,
        use_horizontal_flip_aug=use_horizontal_flip_aug,
        use_affine_aug=use_affine_aug,
    )

    megadepth_train = ConcatDataset(megadepth_train1 + megadepth_train2)
    mega_ws = mega.weight_scenes(megadepth_train, alpha=0.75)
    
    megadepth_test = mega.build_scenes(
        split="test_loftr", min_overlap=0.01, ht=H, wt=W, shake_t=0,
    )
    mega_test = MegadepthNLLBenchmark(ConcatDataset(megadepth_test),
                                      matcher_model=matcher)
    grad_scaler = torch.cuda.amp.GradScaler()
    
    for n in range(n0, N, k):
        mega_sampler = torch.utils.data.WeightedRandomSampler(
            mega_ws, num_samples = 8 * k, replacement=False
        )
        mega_dataloader = iter(
            torch.utils.data.DataLoader(
                megadepth_train,
                batch_size = 8,
                sampler = mega_sampler,
                num_workers = 8,
            )
        )
        train_k_steps(
            n, k, mega_dataloader, model, loss, optim, lr_scheduler, grad_scaler = grad_scaler,
        )
        checkpointer.save(model, optim, lr_scheduler, n, steerer=steerer)
        mega_test.benchmark(detector = detector, descriptor = model)


if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1" # For BF16 computations
    os.environ["OMP_NUM_THREADS"] = "16"
    parser = ArgumentParser()
    parser.add_argument("--pretrained_path")
    parser.add_argument("--pretrained_name")
    parser.add_argument("--detector_weights_name", default='dedode_detector_SO2.pth')
    parser.add_argument("--detector_weights_dir", default='./')
    args, _ = parser.parse_known_args()
    train(args.pretrained_path, args.pretrained_name, args.detector_weights_dir + args.detector_weights_name)
    
    
