# Affine steerers for structured keypoint description
Bökman, Edstedt, Felsberg, Kahl; ECCV 2024

[arxiv link](https://arxiv.org/abs/2408.14186), [pdf](https://arxiv.org/pdf/2408.14186)

[colab demo](https://colab.research.google.com/drive/11hREdmmmrZRJ0USiBS6eQpupWjm4DXKF?usp=sharing)

The code is not cleaned up, I might do that in the future if there is interest.
I'll put up the code now in any case, as it's been too long since the conference for waiting further (I blame the delay on PhD graduation).
The API/training etc works approximately as in [DeDoDe](https://github.com/parskatt/dedode) and [rotation_steerers](https://github.com/georg-bn/rotation-steerers).

Model weights are found in the releases, more detector (and other descriptor) weights are in the DeDoDe and rotation_steerers repos.

A note on the naming conventions: "AffEqui" are the descriptors that are approximately equivariant/steerable with respect to affine transformations. "AffSteer" are descriptors, where the steering elements are trained without supervising them to correspond to interpretable transformations during the fine-tuning stage, i.e. they are not equivariant/steerable with respect to affine transformations.

## Matching example

```python
from affine_steerers.utils import build_affine, load_default_steerer
from affine_steerers.matchers.dual_softmax_matcher import DualSoftMaxMatcher, MaxSimilarityMatcher
from affine_steerers import dedode_detector_L, dedode_descriptor_B, dedode_descriptor_G

model_name = "aff_equi_G"  # ordinary training
# model_name = "aff_steer_G"  # pretraining + prototype training

detector = dedode_detector_L(
    weights=torch.load("dedode_detector_C4.pth")  # Get these weights from the releases of this repo
)
descriptor_path = f"descriptor_{model_name}.pth"  # Get these weights from the releases of this repo
steerer_path = f"steerer_{model_name}.pth"
if "G" in descriptor_path:
    descriptor = dedode_descriptor_G(
        weights=torch.load(descriptor_path)
    )
else:
    descriptor = dedode_descriptor_B(
        weights=torch.load(descriptor_path)
    )
steerer = load_default_steerer(
    steerer_path,
).cuda().eval()

steerer.use_prototype_affines = True
# don't use below for AffSteer
steerer.prototype_affines = torch.stack(
    [
    build_affine(
        angle_1=0.,
        dilation_1=1.,
        dilation_2=1.,
        angle_2=r * 2 * math.pi / 8
        )
    for r in range(8)
    ],  # + ... more affines
    dim=0,
).cuda()

matcher = MaxSimilarityMatcher(
    steerer=steerer,
    normalize=False,
    inv_temp=5,
    threshold=0.01,
)

im_A_path = "im_A.jpg"
im_B_path = "im_B.jpg"
im_A = Image.open(im_A_path)
im_B = Image.open(im_B_path)
w_A, h_A = im_A.size
w_B, h_B = im_B.size

# Detection of keypoints
detections_A = detector.detect_from_path(im_A_path, num_keypoints = 10_000)
keypoints_A, P_A = detections_A["keypoints"], detections_A["confidence"]
detections_B = detector.detect_from_path(im_B_path, num_keypoints = 10_000)
keypoints_B, P_B = detections_B["keypoints"], detections_B["confidence"]

# Describe keypoints and match descriptions (API as in DeDoDe)
descriptions_A = descriptor.describe_keypoints_from_path(im_A_path, keypoints_A)["descriptions"]
descriptions_B = descriptor.describe_keypoints_from_path(im_B_path, keypoints_B)["descriptions"]
matches_A, matches_B, batch_ids = matcher.match(
    keypoints_A, descriptions_A,
    keypoints_B, descriptions_B,
)
matches_A, matches_B = matcher.to_pixel_coords(
    matches_A, matches_B, 
    h_A, w_A, h_B, w_B,
)
```

## Citation (Bibtex)
```
@inproceedings{bokman2024affine,
    title={Affine steerers for structured keypoint description},
    author={B{\"o}kman, Georg and Edstedt, Johan and Felsberg, Michael and Kahl, Fredrik},
    booktitle={European Conference on Computer Vision},
    pages={449--468},
    year={2024}
}
```
