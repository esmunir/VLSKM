import os
import warnings
from argparse import ArgumentParser

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset
import torch.nn as nn

import kornia.augmentation as K

from affine_steerers.train import train_k_steps
from affine_steerers.datasets.megadepth import MegadepthBuilder
from affine_steerers.datasets.homog import generate_megadepth_homography_data
from affine_steerers.datasets.homog import GeometricSequential, MyHorizontalFlip
from affine_steerers.descriptors.descriptor_loss import DescriptorLoss
from affine_steerers.checkpoint import CheckPoint
from affine_steerers.descriptors.dedode_descriptor import affine_steerersDescriptor
from affine_steerers.encoder import VGG_DINOv2
from affine_steerers.decoder import ConvRefiner, Decoder
from affine_steerers import dedode_detector_L
from affine_steerers.benchmarks import MegadepthNLLBenchmark
from affine_steerers.steerers import Steerer1, SteererSpread
from affine_steerers.utils import inverse_distance_similarity, gauss_similarity, scalar_product_similarity, negative_distance_similarity, get_reference_desc
from affine_steerers.matchers.dual_softmax_matcher import DualSoftMaxMatcher


def train(detector_weights_path = "dedode_detector_L.pth"):
    NUM_PROTOTYPES = 256 # == descriptor size
    residual = True
    hidden_blocks = 5
    amp_dtype = torch.float16#torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp = True
    conv_refiner = nn.ModuleDict(
        {
            "14": ConvRefiner(
                1024,
                768,
                512 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,
            ),
            "8": ConvRefiner(
                512 + 512,
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
    vgg_kwargs = dict(size = "19", pretrained = True, amp = amp, amp_dtype = amp_dtype)
    dinov2_kwargs = dict(amp = amp, amp_dtype = amp_dtype)
    encoder = VGG_DINOv2(vgg_kwargs = vgg_kwargs, dinov2_kwargs = dinov2_kwargs)
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

    matcher = DualSoftMaxMatcher(
        normalize=False,
        inv_temp=5,
        threshold=0.01,
        similarity=negative_distance_similarity,
    )

    loss = DescriptorLoss(
        detector=detector,
        steerer=steerer,
        steer_kptwise=True,
        filter_inliers_with_mask=True,
        matcher=matcher,
    )

    H, W = 560, 560

    H_generator = GeometricSequential(
        K.RandomPerspective(
            0.6,
            sampling_method='area_preserving',
            p=1,
            align_corners=True,
        ),
        K.RandomAffine(
            degrees=360,
            translate=0,
            scale=[1.5, 2.],
            p=1,
            align_corners=True,
        ),
        MyHorizontalFlip(p=.5),
    )

    megadepth_train = generate_megadepth_homography_data(
        data_root="data/megadepth",
        H_generator=H_generator,
        H=H, W=W,
    )

    mega = MegadepthBuilder(data_root="data/megadepth",
                            loftr_ignore=True,
                            imc21_ignore=True)

    megadepth_test = mega.build_scenes(
        split="test_loftr", min_overlap=0.01, ht=H, wt=W, shake_t=0,
    )
    mega_test = MegadepthNLLBenchmark(ConcatDataset(megadepth_test),
                                      matcher_model=matcher)
    grad_scaler = torch.cuda.amp.GradScaler()
    
    for n in range(n0, N, k):
        sampler = torch.utils.data.RandomSampler(
           megadepth_train, num_samples = 8 * k, replacement=False
        )
        mega_dataloader = iter(
            torch.utils.data.DataLoader(
                megadepth_train,
                batch_size = 8,
                sampler = sampler,
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
    parser.add_argument("--detector_weights_name", default='dedode_detector_SO2.pth')
    parser.add_argument("--detector_weights_dir", default='./')
    args, _ = parser.parse_known_args()
    train(args.detector_weights_dir + args.detector_weights_name)
    
    
