"""
fused_ce.py -- chunked linear + cross-entropy, pure PyTorch (no Triton / Liger / CUDA-only deps,
so it works on ROCm and on CPU for debugging).

Problem it solves
-----------------
modeling_nanity.py computes `logits = F.linear(h, W)` for every loss-bearing position, and the
trainer then does `.float()` and `F.cross_entropy(...)`. For N loss tokens and vocab V that keeps
alive, at the same time:  bf16 logits (2NV bytes) + fp32 copy (4NV) + CE softmax buffer (4NV)
+ fp32 grad (4NV) + bf16 grad (2NV)  ~= 16NV bytes.
With V=200064 and N=32768 (batch 4 x 8192) that is ~100 GB -- the constant OOM. In pretraining every
position is loss-bearing, so `select_positions` gives no savings there.

This computes the loss in row-chunks under torch.utils.checkpoint: each chunk's logits exist only
inside that chunk (forward) and are recomputed (backward), so peak memory is ~16 * chunk * V bytes
(chunk=4096, V=200k: ~13 GB; V=50k: ~3 GB) regardless of N. Extra cost: one more output-projection
matmul in backward (a few % of a step for a 1.5B model).

It returns the PER-TOKEN loss vector (fp32), so the trainer's answer/think weighting and logging
keep working unchanged.

How to use (3 small edits)
--------------------------
1) modeling_nanity.py, NanityForCausalLM.forward: add a parameter `skip_logits: bool = False`
   and, right after `h = self.output_norm(h)`:

        if skip_logits:
            return h, presents

2) train_nanity_fixed.py (training loop AND evaluate()), replace

        logits_valid, _ = model(input_ids, select_positions=valid_idx)
        logits_valid = logits_valid.float()
        ... per_tok = F.cross_entropy(logits_valid, labels_valid, label_smoothing=..., reduction="none")

   with

        from fused_ce import chunked_linear_ce
        h, _ = model(input_ids, skip_logits=True)
        h_valid = h.reshape(-1, h.shape[-1]).index_select(0, valid_idx)
        base = model._orig_mod if hasattr(model, "_orig_mod") else model
        per_tok = chunked_linear_ce(h_valid, base.output_weight(), labels_valid,
                                    chunk=4096, label_smoothing=args.label_smoothing)

   (The online-distillation branch needs full student logits for the KL term; leave that branch as is.)

3) Pretraining (all positions loss-bearing): valid_idx is simply every position; pass the labels
   already aligned to positions, exactly as the existing collate does.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_loss(h, weight, labels, label_smoothing: float, ignore_index: int):
    logits = F.linear(h, weight).float()
    return F.cross_entropy(
        logits, labels, reduction="none",
        label_smoothing=label_smoothing, ignore_index=ignore_index,
    )


def chunked_linear_ce(
    h: torch.Tensor,            # [N, d]  final hidden states (after output_norm)
    weight: torch.Tensor,       # [V, d]  output / tied embedding matrix
    labels: torch.Tensor,       # [N]     target ids (ignore_index rows give 0 loss)
    chunk: int = 4096,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Per-token cross-entropy of softmax(h @ weight.T) vs labels, fp32, shape [N]."""
    n = h.shape[0]
    if n <= chunk:
        return _chunk_loss(h, weight, labels, label_smoothing, ignore_index)
    out = []
    for i in range(0, n, chunk):
        hc, lc = h[i:i + chunk], labels[i:i + chunk]
        if torch.is_grad_enabled() and (h.requires_grad or weight.requires_grad):
            out.append(checkpoint(_chunk_loss, hc, weight, lc, label_smoothing, ignore_index,
                                  use_reentrant=False))
        else:
            out.append(_chunk_loss(hc, weight, lc, label_smoothing, ignore_index))
    return torch.cat(out)
