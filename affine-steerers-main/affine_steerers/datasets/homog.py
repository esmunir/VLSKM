from glob import glob
from PIL import Image
import numpy as np
import torch
import albumentations as A
import kornia.augmentation as K
from torchvision import transforms
from kornia.geometry.transform import warp_perspective

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class MyHorizontalFlip():
    """ Hacky way to generate horizontal flips. """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.flags = None

    def compute_transformation(self, x, params=None, flags=None):
        flip = torch.eye(3, device=x.device, dtype=x.dtype)
        flip[0, 0] = -1.
        flip[0, 2] = x.shape[-1] - 1.
        return flip

    def generate_parameters(self, b_size):
        return None


class GeometricSequential(K.AugmentationBase2D):
    def __init__(self, *transforms: list, align_corners=True) -> None:
        self.transforms = transforms
        self.align_corners = align_corners

    def __call__(self, x, mode="bilinear"):
        b, c, h, w = x.shape
        M = torch.eye(3, device=x.device)[None].expand(b, 3, 3)
        for t in self.transforms:
            if np.random.rand() < t.p:
                
                M = M.matmul(
                    t.compute_transformation(x, t.generate_parameters((b, c, h, w)), t.flags)
                )
        return (
            warp_perspective(
                x, M, dsize=(h, w), mode=mode, align_corners=self.align_corners
            ),
            M,
        )

    def apply_transform(self, x, M, mode="bilinear"):
        b, c, h, w = x.shape
        return warp_perspective(
            x, M, dsize=(h, w), align_corners=self.align_corners, mode=mode
        )

def make_normalize_transform(
    mean = IMAGENET_DEFAULT_MEAN,
    std = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)

def make_unnormalize_transform(
    mean = IMAGENET_DEFAULT_MEAN,
    std = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=[-mu/sigma for mu, sigma in zip(mean, std)], std=[1/sigma for sigma in std])

def get_lg_augmentation() -> A.Compose:
    transforms = A.Compose([
            A.RandomGamma(p=0.1, gamma_limit=(15, 65)),
            A.HueSaturationValue(p=0.1, val_shift_limit=(-100, -40)),
            A.OneOf(
                [
                    A.Blur(blur_limit=(3, 9)),
                    A.MotionBlur(blur_limit=(3, 25)),
                    A.ISONoise(),
                    A.ImageCompression(),
                ],
                p=0.1,
            ),
            A.Blur(p=0.1, blur_limit=(3, 9)),
            A.MotionBlur(p=0.1, blur_limit=(3, 25)),
            A.RandomBrightnessContrast(
                p=0.5, brightness_limit=(-0.4, 0.0), contrast_limit=(-0.3, 0.0)
            ),
            A.CLAHE(p=0.2),
        ], p = 0.95)
    return transforms


class HomographyDataset:
    def __init__(
        self,
        ims: list[str],
        H_generator : K.RandomPerspective = None,
        photometric_distortion : A.Compose = None,
        H = 512,
        W = 512,
    ) -> None:
        self.ims = np.array(ims)
        self.H_generator = H_generator
        self.H = H
        self.W = W
        self.photometric_distortion = photometric_distortion
        self.normalizer = make_normalize_transform()
        self.unnormalizer = make_unnormalize_transform()

    def __len__(self):
        return len(self.ims)

    def normalize_H(self, Homog, H, W, device = "cpu"):
        T = torch.tensor([[[2 / W, 0, -1], [0, 2 / H, -1], [0, 0, 1]]], device=device)
        H_n = T @ Homog @ torch.linalg.inv(T)
        return H_n

    def warp_from_H(self, im_A, H):
        b, c, h, w = im_A.shape
        device = im_A.device
        im_A_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            )
        )
        im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0], torch.ones_like(im_A_coords[0])), dim=-1)[
            None
        ].expand(b, h, w, 3)
        map_A_to_B = torch.einsum("bhwc,bdc -> bhwd", im_A_coords, H)
        map_A_to_B = map_A_to_B[..., :2] / (map_A_to_B[..., 2:] + 1e-8)
        warp_A_to_B = torch.cat((im_A_coords, map_A_to_B), dim = -1)
        return warp_A_to_B

    def covisible_mask(self, im_A, H_source_to_A, H_source_to_B):
        ones = torch.ones_like(im_A)
        mask = self.H_generator.apply_transform(ones, H_source_to_A)
        H = H_source_to_A @ torch.linalg.inv(H_source_to_B)
        mask *= self.H_generator.apply_transform(ones, H)
        return mask

    def __getitem__(self, idx):
        with torch.no_grad():
            try:
                source = Image.open(self.ims[idx]).convert('RGB').resize((self.W, self.H))
                source = np.array(source)
                im_A = self.normalizer(torch.tensor(self.photometric_distortion(image=source)["image"]).float().div(255).permute(2, 0, 1)[None])
                im_B = self.normalizer(torch.tensor(self.photometric_distortion(image=source)["image"]).float().div(255).permute(2, 0, 1)[None])
                im_A, H_source_to_A = self.H_generator(im_A)
                im_B, H_source_to_B = self.H_generator(im_B)
                H_A_to_B = H_source_to_B @ torch.linalg.inv(H_source_to_A)
                H_B_to_A = np.linalg.inv(H_A_to_B)
                
                mask_A = self.covisible_mask(im_A, H_source_to_A, H_source_to_B)
                mask_B = self.covisible_mask(im_B, H_source_to_B, H_source_to_A)
                
                device = H_A_to_B.device
                H_A_to_B = self.normalize_H(H_A_to_B, self.H, self.W, device = device)
                H_B_to_A = self.normalize_H(H_B_to_A, self.H, self.W, device = device)

                #warp_A_to_B = self.warp_from_H(im_A, H_A_to_B)
                #warp_B_to_A = self.warp_from_H(im_B, H_B_to_A)
                
                data_dict = {
                    "im_A": im_A[0],
                    "im_B": im_B[0],
                    "mask_A": mask_A[0, 0],
                    #"warp_A_to_B": warp_A_to_B[0],
                    "mask_B": mask_B[0, 0],
                    #"warp_B_to_A": warp_B_to_A[0],
                    "Homog_A_to_B": H_A_to_B[0],
                    "H_source_to_A": H_source_to_A[0],
                    "H_source_to_B": H_source_to_B[0],
                }
                return data_dict
            except Exception as e:
                print(e)
                print(f"Failed to load image {self.ims[idx]}")
                print("Loading a random image instead.")
                rand_ind = np.random.choice(range(len(self)))
                return self[rand_ind]


def generate_megadepth_homography_data(
    data_root,
    H_generator = None,
    H=512, W=512,
):
    from glob import glob
    ignore_scenes = [
        "0017", "0004", "0048", "0013", "0015", "0022",  # test
        "0121", "0133", "0168", "0178", "0229", "0349",  # loftr ignore
        "0412", "0430", "0443", "1001", "5014", "5015", "5016",
        "0008", "0019", "0021", "0024", "0025", "0032", "0063", "1589", # imc21
    ]
    ims = []
    for image in glob(f"{data_root}/Undistorted_SfM/*/images/*.jpg"):
        flag = True
        for name in ignore_scenes:
            if name in image:
                flag = False
                continue
        if flag:
            ims.append(image)
    print(f"Created dataset. Nbr images: {len(ims)}.")
    photometric = get_lg_augmentation()
    if H_generator is None:
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
    dataset = HomographyDataset(
        ims,
        photometric_distortion=photometric,
        H_generator=H_generator,
        H=H, W=W,
    )
    return dataset

def generate_revisitop1m_homography_data(
    data_root,
    H_generator = None,
    H=512, W=512,
):
    from glob import glob
    ims = [image for image in glob(f"{data_root}/jpg/*/*.jpg")]
    print(f"Created dataset. Nbr images: {len(ims)}.")
    photometric = get_lg_augmentation()
    if H_generator is None:
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
    dataset = HomographyDataset(
        ims,
        photometric_distortion=photometric,
        H_generator=H_generator,
        H=H, W=W,
    )
    return dataset

def generate_mapillary_planet_scale(folders):
    from glob import glob
    #print(f"{folders=}")
    ims = [image for folder in folders for image in glob(f"{folder}/*.jpg") ]
    #print(f"{ims=}")
    H_generator = GeometricSequential(K.RandomPerspective(0.6, sampling_method='area_preserving', p=1, align_corners = True), K.RandomAffine(degrees=0, translate = 0, scale = [1.5, 2.], p = 1, align_corners = True))
    photometric = get_lg_augmentation()
    dataset = HomographyDataset(ims, photometric_distortion = photometric, H_generator = H_generator)
    return dataset
    

if __name__ == "__main__":
    dataset = generate_mapillary_planet_scale()
    data_dict = dataset[0]
    Image.fromarray(data_dict['im_A'].mul(255).permute(1,2,0).numpy().astype(np.uint8)).save("im_A.jpg")
    Image.fromarray(data_dict['im_B'].mul(255).permute(1,2,0).numpy().astype(np.uint8)).save("im_B.jpg")
    Image.fromarray(data_dict['mask'].mul(255).numpy().astype(np.uint8)).save("mask.jpg")
    
