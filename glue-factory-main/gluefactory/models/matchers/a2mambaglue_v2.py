import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
import math

# ⚡ THE TRUE A2SSM KERNEL
import selective_scan_cuda_oflex_rh

try:
    from flash_attn.modules.mha import FlashCrossAttention
except ModuleNotFoundError:
    FlashCrossAttention = None

if FlashCrossAttention or hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = True
else:
    FLASH_AVAILABLE = False

torch.backends.cudnn.deterministic = True

# ======================================================================
# ⚡ CUSTOM CUDA KERNEL WRAPPER (From A2Mamba)
# ======================================================================
class SelectiveScanStateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
        if u.stride(-1) != 1: u = u.contiguous()
        if delta.stride(-1) != 1: delta = delta.contiguous()
        if D is not None: D = D.contiguous()
        if B.stride(-1) != 1: B = B.contiguous()
        if z is not None and z.stride(-1) != 1: z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True

        out, x, *rest = selective_scan_cuda_oflex_rh.fwd(u, delta, A, B, D, delta_bias, delta_softplus, 1, True)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, D, delta_bias, x = ctx.saved_tensors
            z, out = None, None
        else:
            u, delta, A, B, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1: dout = dout.contiguous()
        
        du, ddelta, dA, dB, dD, ddelta_bias, *rest = selective_scan_cuda_oflex_rh.bwd(
            u, delta, A, B, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        return (du, ddelta, dA, dB, dD if D is not None else None, dz, ddelta_bias if delta_bias is not None else None, None, None, None)

def selective_scan_state_fn(u, delta, A, B, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    return SelectiveScanStateFn.apply(u, delta, A, B, D, z, delta_bias, delta_softplus, return_last_state)


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
def normalize_keypoints(kpts: torch.Tensor, size: Optional[torch.Tensor] = None) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts

def pad_to_length(x: torch.Tensor, length: int) -> Tuple[torch.Tensor]:
    if length <= x.shape[-2]:
        return x, torch.ones_like(x[..., :1], dtype=torch.bool)
    pad = torch.ones(*x.shape[:-2], length - x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)
    y = torch.cat([x, pad], dim=-2)
    mask = torch.zeros(*y.shape[:-1], 1, dtype=torch.bool, device=x.device)
    mask[..., : x.shape[-2], :] = True
    return y, mask

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])

class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)

class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 4),
            nn.ReLU(), nn.Linear(dim // 4, 1), nn.Sigmoid(),
        )

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        return (self.token(desc0.detach()).squeeze(-1), self.token(desc1.detach()).squeeze(-1))

class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn("FlashAttention is not available.", stacklevel=2)
        self.enable_flash = allow_flash and FLASH_AVAILABLE
        self.has_sdp = hasattr(F, "scaled_dot_product_attention")
        if allow_flash and FlashCrossAttention:
            self.flash_ = FlashCrossAttention()
        if self.has_sdp:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if q.shape[-2] == 0 or k.shape[-2] == 0:
            return q.new_zeros((*q.shape[:-1], v.shape[-1]))
        if self.enable_flash and q.device.type == "cuda":
            if self.has_sdp:
                args = [x.half().contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)
                return v if mask is None else v.nan_to_num()
            else:
                assert mask is None
                q, k, v = [x.transpose(-2, -3).contiguous() for x in [q, k, v]]
                m = self.flash_(q.half(), torch.stack([k, v], 2).half())
                return m.transpose(-2, -3).to(q.dtype).clone()
        elif self.has_sdp:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            attn = F.softmax(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)

# ======================================================================
# 🏆 TRUE A2SSM MIXER (Mathematically Identical to A2Mamba Paper)
# ======================================================================
class TrueA2MambaAttentionMixer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True, d_state=16, d_conv=5, expand=2) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        
        # --- ATTENTION INIT ---
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj_attn = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # --- MAMBA INIT ---
        self.d_state = d_state
        self.d_inner = int(expand * embed_dim)
        self.dt_rank = math.ceil(embed_dim / 16)
        
        self.in_proj_mamba = nn.Linear(embed_dim, self.d_inner * 2, bias=bias)
        self.conv1d_x = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv//2)
        self.conv1d_z = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv//2)
        
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj_mamba = nn.Linear(self.d_inner, embed_dim, bias=bias)

        # --- COMBINATION INIT ---
        self.ffn = nn.Sequential(
            nn.Linear(3 * embed_dim, 2 * embed_dim), 
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, encoding: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        # ==========================================
        # 1. ATTENTION GRAPH EXTRACTION
        # ==========================================
        qkv = self.Wqkv(x).unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        
        q, k = apply_cached_rotary_emb(encoding, q), apply_cached_rotary_emb(encoding, k)
        
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj_attn(context.transpose(1, 2).flatten(start_dim=-2))

        # Explicit A_matrix 
        scale = q.shape[-1] ** -0.5
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
        if mask is not None: sim.masked_fill(~mask, -float("inf"))
        A_matrix = F.softmax(sim, dim=-1)

        # ==========================================
        # 2. MAMBA SEQUENCE PREPARATION
        # ==========================================
        xz = rearrange(self.in_proj_mamba(x), "b l d -> b d l")
        x_m, z_m = xz.chunk(2, dim=1)
        
        x_m, z_m = F.silu(self.conv1d_x(x_m)), F.silu(self.conv1d_z(z_m))
        
        x_dbl = self.x_proj(rearrange(x_m, "b d l -> (b l) d"))
        dt, B_m, C_m = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=L)
        B_m = rearrange(B_m, "(b l) s -> b s l", l=L).contiguous()
        C_m = rearrange(C_m, "(b l) s -> b s l", l=L).contiguous()
        A_m = -torch.exp(self.A_log.float())

        # ==========================================
        # 3. ⚡ TRUE A2SSM: RAW STATE EXTRACTION
        # ==========================================
        state = selective_scan_state_fn(x_m, dt, A_m, B_m, D=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True)

        # ==========================================
        # 4. ⚡ TRUE A2SSM: THE GEOMETRIC BRIDGE
        # ==========================================
        state = rearrange(state, "b (h d_head) s l -> b h l (d_head s)", h=self.num_heads)
        state_routed = torch.matmul(A_matrix, state)
        state_routed = rearrange(state_routed, "b h l (d_head s) -> b (h d_head) s l", h=self.num_heads, s=self.d_state)

        # ==========================================
        # 5. ⚡ TRUE A2SSM: MANUAL FINISH
        # ==========================================
        state_routed = state_routed.to(torch.float32)
        C_m = C_m.to(torch.float32).unsqueeze(1) 
        
        y = torch.sum(state_routed * C_m, dim=2) 
        
        y = y + x_m.float() * self.D.view(-1, 1).float()
        y = y * F.silu(z_m.float())
        
        mamba_out = self.out_proj_mamba(rearrange(y, "b d l -> b l d"))

        # ==========================================
        # 6. FINAL CONCATENATION
        # ==========================================
        return x + self.ffn(torch.cat([x, message, mamba_out], -1))

class CrossBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2), (qk0, qk1, v0, v1))
        
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None)
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerMambaLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # ⚡ USE TRUE A2SSM MIXER
        self.mamba_selfattn_mixer = TrueA2MambaAttentionMixer(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(self, desc0, desc1, encoding0, encoding1, mask0: Optional[torch.Tensor] = None, mask1: Optional[torch.Tensor] = None):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, mask0, mask1)
        else:
            desc0 = self.mamba_selfattn_mixer(desc0, encoding0)
            desc1 = self.mamba_selfattn_mixer(desc1, encoding1)
            return self.cross_attn(desc0, desc1)

    def masked_forward(self, desc0, desc1, encoding0, encoding1, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.mamba_selfattn_mixer(desc0, encoding0, mask0)
        desc1 = self.mamba_selfattn_mixer(desc1, encoding1, mask1)
        return self.cross_attn(desc0, desc1, mask)


def sigmoid_log_double_softmax(sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores

class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)

def filter_matches(scores: torch.Tensor, th: float):
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)

    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1

class A2MambaGlue(nn.Module):
    default_conf = {
        "name": "a2mambaglue", 
        "input_dim": 256,  
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  
        "mp": False,  
        "depth_confidence": -1,  
        "width_confidence": -1,  
        "filter_threshold": 0.01,  
        "weights": None,
    }

    pruning_keypoint_thresholds = {"cpu": -1, "mps": -1, "cuda": 1024, "flash": 1536}
    required_data_keys = ["image0", "image1"]

    features = {
        "superpoint": {"weights": "superpoint_mambaglue", "input_dim": 256},
        "disk": {"weights": "disk_mambaglue", "input_dim": 128},
        "aliked": {"weights": "aliked_mambaglue", "input_dim": 128},
        "sift": {"weights": "sift_mambaglue", "input_dim": 128, "add_scale_ori": True},
        "doghardnet": {"weights": "doghardnet_mambaglue", "input_dim": 128, "add_scale_ori": True},
    }

    def __init__(self, features="superpoint", **conf) -> None:
        super().__init__()
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        if features is not None:
            for k, v in self.features[features].items():
                setattr(conf, k, v)

        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(2 + 2 * self.conf.add_scale_ori, head_dim, head_dim)

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim
        self.transformermambas = nn.ModuleList([TransformerMambaLayer(d, h, conf.flash) for _ in range(n)])
        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList([TokenConfidence(d) for _ in range(n - 1)])
        self.register_buffer("confidence_thresholds", torch.Tensor([self.confidence_threshold(i) for i in range(self.conf.n_layers)]))

        state_dict = None
        if features is not None:
            local_path = Path("checkpoint_best.tar") 
            if local_path.exists():
                checkpoint = torch.load(str(local_path), map_location="cpu")
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                self.load_state_dict(state_dict, strict=False)

        if state_dict:
            for i in range(self.conf.n_layers):
                state_dict = {k.replace("matcher.", ""): v for k, v in state_dict.items()}
                state_dict = {k.replace("extractor", ""): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)

        self.static_lengths = None

    def compile(self, mode="reduce-overhead", static_lengths=[256, 512, 768, 1024, 1280, 1536]):
        torch._inductor.cudagraph_mark_step_begin()
        for i in range(self.conf.n_layers):
            self.transformermambas[i].masked_forward = torch.compile(self.transformermambas[i].masked_forward, mode=mode, fullgraph=True)
        self.static_lengths = static_lengths

    def forward(self, data: dict) -> dict:
        with torch.autocast(enabled=self.conf.mp, device_type="cuda"):
            return self._forward(data)

    def _forward(self, data: dict) -> dict:
        data0, data1 = data["image0"], data["image1"]
        kpts0, kpts1 = data0["keypoints"], data1["keypoints"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        size0, size1 = data0.get("image_size"), data1.get("image_size")
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        desc0 = data0["descriptors"].detach().contiguous()
        desc1 = data1["descriptors"].detach().contiguous()

        if torch.is_autocast_enabled():
            desc0, desc1 = desc0.half(), desc1.half()

        mask0, mask1 = None, None
        c = max(m, n)
        do_compile = self.static_lengths and c <= max(self.static_lengths)
        if do_compile:
            kn = min([k for k in self.static_lengths if k >= c])
            desc0, mask0 = pad_to_length(desc0, kn)
            desc1, mask1 = pad_to_length(desc1, kn)
            kpts0, _ = pad_to_length(kpts0, kn)
            kpts1, _ = pad_to_length(kpts1, kn)
            
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        do_early_stop = self.conf.depth_confidence > 0
        do_point_pruning = self.conf.width_confidence > 0 and not do_compile
        pruning_th = self.pruning_min_kpts(device)

        all_desc0, all_desc1 = [], []

        if do_point_pruning:
            ind0 = torch.arange(0, m, device=device)[None]
            ind1 = torch.arange(0, n, device=device)[None]
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
            
        token0, token1 = None, None
        
        for i in range(self.conf.n_layers):
            if desc0.shape[1] == 0 or desc1.shape[1] == 0: break
                
            desc0, desc1 = self.transformermambas[i](desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1)
            
            desc0 = desc0.nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)
            desc1 = desc1.nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)
            
            all_desc0.append(desc0)
            all_desc1.append(desc1)

            if i == self.conf.n_layers - 1: continue 

            if do_early_stop:
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0[..., :m], token1[..., :n], i, m + n): break
                    
            if do_point_pruning and desc0.shape[-2] > pruning_th:
                scores0 = self.log_assignment[i].get_matchability(desc0)
                prunemask0 = self.get_pruning_mask(token0, scores0, i)
                keep0 = torch.where(prunemask0)[1]
                ind0 = ind0.index_select(1, keep0)
                desc0 = desc0.index_select(1, keep0)
                encoding0 = encoding0.index_select(-2, keep0)
                prune0[:, ind0] += 1
                
            if do_point_pruning and desc1.shape[-2] > pruning_th:
                scores1 = self.log_assignment[i].get_matchability(desc1)
                prunemask1 = self.get_pruning_mask(token1, scores1, i)
                keep1 = torch.where(prunemask1)[1]
                ind1 = ind1.index_select(1, keep1)
                desc1 = desc1.index_select(1, keep1)
                encoding1 = encoding1.index_select(-2, keep1)
                prune1[:, ind1] += 1

        if desc0.shape[1] == 0 or desc1.shape[1] == 0:
            m0 = desc0.new_full((b, m), -1, dtype=torch.long)
            m1 = desc1.new_full((b, n), -1, dtype=torch.long)
            mscores0, mscores1 = desc0.new_zeros((b, m)), desc1.new_zeros((b, n))
            matches, mscores = desc0.new_empty((b, 0, 2), dtype=torch.long), desc0.new_empty((b, 0))
            if not do_point_pruning:
                prune0 = torch.ones_like(mscores0) * self.conf.n_layers
                prune1 = torch.ones_like(mscores1) * self.conf.n_layers
            return {
                "matches0": m0, "matches1": m1, "matching_scores0": mscores0, "matching_scores1": mscores1,
                "stop": i + 1, "matches": matches, "scores": mscores, "prune0": prune0, "prune1": prune1,
                "ref_descriptors0": torch.stack(all_desc0, 1) if all_desc0 else None,
                "ref_descriptors1": torch.stack(all_desc1, 1) if all_desc1 else None, "log_assignment": None,
            }

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)
        matches, mscores = [], []
        
        for k in range(b):
            valid = m0[k] > -1
            m_indices_0, m_indices_1 = torch.where(valid)[0], m0[k][valid]
            if do_point_pruning:
                m_indices_0, m_indices_1 = ind0[k, m_indices_0], ind1[k, m_indices_1]
            matches.append(torch.stack([m_indices_0, m_indices_1], -1))
            mscores.append(mscores0[k][valid])

        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = torch.where(m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            m1_[:, ind1] = torch.where(m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            mscores0_ = torch.zeros((b, m), device=mscores0.device)
            mscores1_ = torch.zeros((b, n), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones_like(mscores0) * self.conf.n_layers
            prune1 = torch.ones_like(mscores1) * self.conf.n_layers

        return {
            "matches0": m0, "matches1": m1, "matching_scores0": mscores0, "matching_scores1": mscores1,
            "stop": i + 1, "matches": matches, "scores": mscores, "prune0": prune0, "prune1": prune1,
            "ref_descriptors0": torch.stack(all_desc0, 1) if all_desc0 else None,
            "ref_descriptors1": torch.stack(all_desc1, 1) if all_desc1 else None, "log_assignment": scores,
        }

    def confidence_threshold(self, layer_index: int) -> float:
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.conf.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int) -> torch.Tensor:
        keep = scores > (1 - self.conf.width_confidence)
        if confidences is not None: 
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(self, confidences0: torch.Tensor, confidences1: torch.Tensor, layer_index: int, num_points: int) -> torch.Tensor:
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.conf.depth_confidence

    def pruning_min_kpts(self, device: torch.device):
        if self.conf.flash and FLASH_AVAILABLE and device.type == "cuda":
            return self.pruning_keypoint_thresholds["flash"]
        else:
            return self.pruning_keypoint_thresholds[device.type]