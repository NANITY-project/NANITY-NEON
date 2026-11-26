"""
nanity_fast.py -- training-speed forward for the NANITY architecture.

Subclasses NanityForCausalLM, so parameter names/shapes are IDENTICAL to
modeling_nanity.py (checkpoints load both ways, export_gguf/export_nctr work
unchanged). Only the forward is different:

  * fp32 MASTER weights, bf16 compute: every weight is cast to the activation
    dtype inside the (compiled) block functions. The old trainer did
    model.to(bfloat16) and ran AdamW directly on bf16 params, which silently
    drops small updates (bf16 has ~3 digits) -- worst late in the cosine decay.
  * block = [compiled pre-attn] -> [eager attention] -> [compiled post-attn].
    The pre/post functions take the weights as ARGUMENTS, so torch.compile
    traces each ONCE and reuses it for all layers (no 36x compile, no
    dynamic=True). Attention stays eager so flash_attn (opaque kernel) never
    blocks fullgraph compile.
  * attention backends:
      flash_varlen : flash_attn_varlen_func (native GQA, document masking via
                     cu_seqlens). Best. Needs the flash-attn package
                     (ROCm: Triton backend via FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE,
                     or the CK build that ships in AMD's rocm/pytorch-training images).
      sdpa         : torch SDPA flash/efficient kernel, causal over the whole row
                     (documents in a packed row can see each other). K/V are
                     expanded to n_head by hand instead of enable_gqa=True --
                     some ROCm builds have no fused GQA path and silently fall
                     to the O(T^2)-memory math kernel, which at T=8192 is both
                     slow and an OOM source.
      masked       : dense block-diagonal mask via SDPA. Correct but slow and
                     memory hungry -- CPU / tests only.
  * forward_hidden() returns the final-norm hidden states; the vocab
    projection + loss live in fused_ce.py (never materialises full logits).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modeling_nanity import NanityConfig, NanityForCausalLM

try:  # optional
    from flash_attn import flash_attn_varlen_func as _flash_varlen
except Exception:  # ImportError or a broken build
    _flash_varlen = None

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _SDPA_GPU = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
except Exception:
    sdpa_kernel = None
    _SDPA_GPU = None


# --------------------------------------------------------------------------- #
# pure functions (compiled once, weights passed in)
# --------------------------------------------------------------------------- #

def _rms(x, w, eps):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * w.float()).to(x.dtype)


def _rope(x, cos, sin, rot_dim: int):
    """Adjacent-pair RoPE (spec section 6). x [B,T,H,D]; cos/sin [B,T,rot/2] fp32."""
    B, T, H, D = x.shape
    xr = x[..., :rot_dim].float().reshape(B, T, H, rot_dim // 2, 2)
    x1, x2 = xr[..., 0], xr[..., 1]
    c, s = cos[:, :, None, :], sin[:, :, None, :]
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).reshape(B, T, H, rot_dim).to(x.dtype)
    if rot_dim == D:
        return out
    return torch.cat([out, x[..., rot_dim:]], dim=-1)


def _pre_attn(x, cos, sin, norm_w, wq, wk, wv,
              eps: float, n_head: int, n_kv: int, head_dim: int, rot_dim: int):
    B, T, _ = x.shape
    cd = x.dtype
    h = _rms(x, norm_w, eps)
    q = F.linear(h, wq.to(cd)).view(B, T, n_head, head_dim)
    k = F.linear(h, wk.to(cd)).view(B, T, n_kv, head_dim)
    v = F.linear(h, wv.to(cd)).view(B, T, n_kv, head_dim)
    return _rope(q, cos, sin, rot_dim), _rope(k, cos, sin, rot_dim), v


def _post_attn(x, a, wo, norm_w, wg, wu, wd, eps: float):
    cd = x.dtype
    x = x + F.linear(a, wo.to(cd))
    h = _rms(x, norm_w, eps)
    g = F.linear(h, wg.to(cd))
    u = F.linear(h, wu.to(cd))
    return x + F.linear(F.silu(g) * u, wd.to(cd))


def _attention(q, k, v, backend: str, seg, cu, max_seqlen: int):
    B, T, H, D = q.shape
    Hkv = k.shape[2]
    if backend == "flash_varlen":
        o = _flash_varlen(q.reshape(B * T, H, D), k.reshape(B * T, Hkv, D),
                          v.reshape(B * T, Hkv, D), cu, cu, max_seqlen, max_seqlen,
                          causal=True)
        return o.reshape(B, T, H * D)
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if Hkv != H:
        g = H // Hkv
        kt = kt.repeat_interleave(g, dim=1)   # head h <- kv head h // g (same as enable_gqa)
        vt = vt.repeat_interleave(g, dim=1)
    if backend == "masked":
        causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        mask = ((seg[:, :, None] == seg[:, None, :]) & causal)[:, None]
        o = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask)
    else:  # sdpa
        if q.is_cuda and sdpa_kernel is not None:
            with sdpa_kernel(_SDPA_GPU):   # no MATH fallback: fail loudly, not O(T^2)
                o = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
        else:
            o = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
    return o.transpose(1, 2).reshape(B, T, H * D)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

class FastNanity(NanityForCausalLM):
    def __init__(self, cfg: NanityConfig, attn: str = "auto",
                 compute_dtype: torch.dtype = torch.bfloat16, ckpt_layers: int = 0):
        super().__init__(cfg)
        self.compute_dtype = compute_dtype
        self.ckpt_layers = ckpt_layers
        self.attn_backend = self._pick_backend(attn)
        self._pre, self._post = _pre_attn, _post_attn

    @staticmethod
    def _pick_backend(attn: str) -> str:
        if attn == "auto":
            return "flash_varlen" if (_flash_varlen is not None and torch.cuda.is_available()) \
                else ("sdpa" if torch.cuda.is_available() else "masked")
        if attn == "flash_varlen" and _flash_varlen is None:
            raise RuntimeError("--attn flash_varlen but `import flash_attn` failed")
        return attn

    @property
    def doc_aware(self) -> bool:
        """True if the backend isolates documents inside a packed row."""
        return self.attn_backend in ("flash_varlen", "masked")

    def enable_compile(self, mode: Optional[str] = None):
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        kw = dict(fullgraph=True, dynamic=False)
        if mode and mode != "default":
            kw["mode"] = mode
        self._pre = torch.compile(_pre_attn, **kw)
        self._post = torch.compile(_post_attn, **kw)

    def _block(self, blk, x, cos, sin, seg, cu, max_seqlen):
        c = self.cfg
        a = blk.attn
        q, k, v = self._pre(x, cos, sin, blk.attn_norm.weight,
                            a.attn_q.weight, a.attn_k.weight, a.attn_v.weight,
                            c.rms_norm_eps, c.n_head, c.n_kv_head, c.head_dim, c.rope_dimension_count)
        o = _attention(q, k, v, self.attn_backend, seg, cu, max_seqlen)
        f = blk.ffn
        return self._post(x, o, a.attn_output.weight, blk.ffn_norm.weight,
                          f.ffn_gate.weight, f.ffn_up.weight, f.ffn_down.weight, c.rms_norm_eps)

    def forward_hidden(self, input_ids, pos=None, seg=None, cu=None, max_seqlen: int = 0):
        """input_ids [B,T] long; pos [B,T] long (per-document positions, or None
        for 0..T-1); seg [B,T] int (only for backend 'masked'); cu/max_seqlen only
        for 'flash_varlen'. Returns final-norm hidden [B,T,d] in compute dtype."""
        B, T = input_ids.shape
        assert T <= self.cfg.context_length, f"seq len {T} > context_length {self.cfg.context_length}"
        x = F.embedding(input_ids, self.token_embd.weight).to(self.compute_dtype)
        if pos is None:
            cos = self.rope_cos[:T][None].expand(B, -1, -1)
            sin = self.rope_sin[:T][None].expand(B, -1, -1)
        else:
            cos, sin = self.rope_cos[pos], self.rope_sin[pos]
        for i, blk in enumerate(self.blk):
            if self.training and i < self.ckpt_layers:
                x = checkpoint(self._block, blk, x, cos, sin, seg, cu, max_seqlen, use_reentrant=False)
            else:
                x = self._block(blk, x, cos, sin, seg, cu, max_seqlen)
        return _rms(x, self.output_norm.weight, self.cfg.rms_norm_eps)


# --------------------------------------------------------------------------- #
# MFU helpers
# --------------------------------------------------------------------------- #

_PEAK_TFLOPS_BF16 = {"MI300": 1307.4, "MI325": 1307.4, "H100": 989.4, "H200": 989.4,
                     "A100": 312.0, "L40": 181.0, "4090": 165.2}


def peak_tflops_for(device_name: str) -> float:
    for k, v in _PEAK_TFLOPS_BF16.items():
        if k in device_name:
            return v
    return 0.0


def flops_per_token(cfg: NanityConfig, seq_len: int, causal_discount: bool = False) -> float:
    """fwd+bwd FLOPs/token, PaLM convention: 6*N + 12*L*q_dim*T, where N counts
    the matmul params (blocks + the vocab projection; the embedding LOOKUP is
    free). causal_discount halves the attention term (flash only does half)."""
    per_layer = (cfg.n_embd * cfg.q_dim * 2 + cfg.n_embd * cfg.kv_dim * 2
                 + 3 * cfg.n_embd * cfg.n_ff)
    n = cfg.n_layer * per_layer + cfg.vocab_size * cfg.n_embd
    attn = 12 * cfg.n_layer * cfg.q_dim * seq_len
    return 6 * n + (attn / 2 if causal_discount else attn)
