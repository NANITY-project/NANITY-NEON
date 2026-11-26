"""
fused_ce.py -- fused (chunked) output-projection + cross-entropy, pure PyTorch
(no Triton / Liger / CUDA-only deps, so it runs on ROCm and CPU).

Why
---
With V=200064, a naive loss keeps bf16 logits + fp32 copy + softmax buffer + grads
alive for every loss token at once: ~16*N*V bytes (N=32k tokens -> ~100 GB). That
is the "constant OOM".

How (same trick as Liger's FusedLinearCrossEntropy)
---------------------------------------------------
Walk the tokens in chunks. For each chunk, in the FORWARD pass:
    logits = h_c @ W^T            (1 matmul)
    loss_c, dlogits = CE(logits)  (elementwise, in place)
    grad_h_c  = dlogits @ W       (1 matmul)
    grad_W   += dlogits^T @ h_c   (1 matmul)
and keep only grad_h (N x d) and grad_W (V x d). Backward just scales them by
the incoming scalar grad. Total = 3 matmuls, the same as a normal fwd+bwd, so
there is NO recompute (the checkpoint-based version needed a 4th matmul), and
peak extra memory is ~ chunk*V*(2+4+2) bytes regardless of N.

Interface
---------
    loss, per_tok = fused_linear_ce(h, W, labels, scale, chunk=4096,
                                    label_smoothing=0.0)

  h       [N, d]  hidden states (compute dtype, e.g. bf16), requires grad
  W       [V, d]  output / tied-embedding weight (fp32 master is fine; it is
                  cast to h.dtype for the matmuls, grad comes back fp32)
  labels  [N]     target ids, -100 = ignore
  scale   [N]     fp32 per-token weight INCLUDING the normalisation, i.e.
                  loss = sum_i scale_i * ce_i  (e.g. w_i / sum(w))
  returns loss (scalar, differentiable) and per_tok (fp32 [N], detached CE per
  token, for think/answer logging).
"""
from __future__ import annotations

import torch

IGNORE = -100


def _chunk_forward(hc, w, lc, ls, V):
    """Returns (logits_f32 [c,V], lse [c], ce [c], valid [c], lc_safe [c])."""
    logits = (hc @ w.t()).float()
    valid = lc != IGNORE
    lc_safe = lc.clamp(min=0)
    lse = torch.logsumexp(logits, dim=-1)
    tgt = logits.gather(1, lc_safe[:, None]).squeeze(1)
    ce = lse - tgt
    if ls > 0.0:
        smooth = lse - logits.mean(dim=-1)
        ce = (1.0 - ls) * ce + ls * smooth
    ce = torch.where(valid, ce, torch.zeros_like(ce))
    return logits, lse, ce, valid, lc_safe


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, weight, labels, scale, chunk, label_smoothing):
        n, d = h.shape
        V = weight.shape[0]
        cdt = h.dtype
        need_h, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        w = weight if weight.dtype == cdt else weight.to(cdt)

        per_tok = torch.zeros(n, dtype=torch.float32, device=h.device)
        loss = torch.zeros((), dtype=torch.float32, device=h.device)
        grad_h = torch.empty_like(h) if need_h else None
        grad_w = torch.zeros(weight.shape, dtype=torch.float32, device=h.device) if need_w else None

        for i in range(0, n, chunk):
            hc, lc, sc = h[i:i + chunk], labels[i:i + chunk], scale[i:i + chunk]
            logits, lse, ce, valid, lc_safe = _chunk_forward(hc, w, lc, label_smoothing, V)
            per_tok[i:i + chunk] = ce
            loss += (ce * sc).sum()
            if need_h or need_w:
                # d ce / d logits = softmax - target_dist, built in place.
                logits.sub_(lse[:, None]).exp_()                 # softmax
                if label_smoothing > 0.0:
                    logits.sub_(label_smoothing / V)             # uniform part
                rows = torch.arange(logits.shape[0], device=logits.device)
                logits[rows, lc_safe] -= (1.0 - label_smoothing)  # one-hot part
                logits.mul_((sc * valid)[:, None])
                gl = logits.to(cdt)
                del logits
                if need_h:
                    grad_h[i:i + chunk] = gl @ w
                if need_w:
                    grad_w += (gl.t() @ hc)                      # bf16 GEMM, fp32 accumulate
                del gl

        ctx.mark_non_differentiable(per_tok)
        ctx.save_for_backward(grad_h, grad_w)
        return loss, per_tok

    @staticmethod
    def backward(ctx, grad_loss, _grad_per_tok):
        grad_h, grad_w = ctx.saved_tensors
        g = grad_loss
        return (
            grad_h * g.to(grad_h.dtype) if grad_h is not None else None,
            grad_w * g if grad_w is not None else None,
            None, None, None, None,
        )


@torch.no_grad()
def linear_ce_per_token(h, weight, labels, chunk=4096, label_smoothing=0.0):
    """Eval path: per-token CE (fp32 [N]) without any gradient work."""
    n = h.shape[0]
    V = weight.shape[0]
    w = weight if weight.dtype == h.dtype else weight.to(h.dtype)
    out = torch.zeros(n, dtype=torch.float32, device=h.device)
    for i in range(0, n, chunk):
        _, _, ce, _, _ = _chunk_forward(h[i:i + chunk], w, labels[i:i + chunk], label_smoothing, V)
        out[i:i + chunk] = ce
    return out


def fused_linear_ce(h, weight, labels, scale, chunk=4096, label_smoothing=0.0):
    if h.shape[0] == 0:  # batch with no loss tokens
        z = (h.sum() + weight.sum()) * 0.0
        return z, torch.zeros(0, dtype=torch.float32, device=h.device)
    if not torch.is_grad_enabled():
        pt = linear_ce_per_token(h, weight, labels, chunk, label_smoothing)
        return (pt * scale).sum(), pt
    return _FusedLinearCE.apply(h, weight, labels, scale, chunk, label_smoothing)
