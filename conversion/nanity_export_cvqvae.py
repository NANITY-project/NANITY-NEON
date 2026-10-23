#!/usr/bin/env python3
"""
nanity_export_cvqvae.py — convert a NERVA conditional VQ-VAE checkpoint
(tools/train_cvqvae.py in the NERVA repo, e.g. runs/cvqvae/best.pt) into a
NEON-loadable GGUF (general.architecture = "nanity-cvqvae", read by
runtime/motion/rawllm_cvqvae.hpp).

Checkpoint layout this script understands (dict saved by train_cvqvae.py):
    model   state_dict   (act_emb, cond_mlp, enc.*, vq.embed, dec.*, in_scale, in_shift)
    cfg     dict         (n_actions, window, hidden, emb, code_dim, num_codes,
                          downs, n_res, cond_enc, ...)
    norm    dict         (mean, std, dstd — per-channel, 30 channels)
    action_names, dt     (optional metadata)
Anything else in the state_dict is refused by name rather than silently dropped.

Layout decisions (these make the runtime fast):
  * Every Conv1d weight (out, in, k) is exported as ONE 2-D matrix
    [out, k*in] with taps outermost: column index = tap*in + channel. The
    runtime gathers k time-shifted input rows per output step (im2col) and
    runs one batched GEMM over all time steps. GGUF shape is [k*in, out].
  * Training-only buffers (vq.cluster_size, vq.embed_avg, vq.initted) are not
    exported. The codebook (vq.embed) is always F32.
  * Biases, GroupNorm affine, the normalisation stats and the codebook stay F32.
    --dtype f16 only re-encodes the large conv/linear matrices and is lossy,
    so it is opt-in. (Q8_0 is not offered: hidden=528 gives K that is not a
    multiple of 32.)

Usage:
    python3 nanity_export_cvqvae.py best.pt cvqvae.gguf [--dtype f32|f16]
    pip install torch gguf numpy

Security: torch.load() is called with weights_only=True (no arbitrary pickle
execution). If your checkpoint needs full unpickling, pass --trust-pickle,
and only do that for files you made yourself.
"""
import argparse, json, sys
import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("Missing deps: pip install gguf numpy torch")

IGNORED = {"vq.cluster_size", "vq.embed_avg", "vq.initted"}   # EMA training state
REQUIRED_CFG = ("n_actions", "window", "hidden", "emb", "code_dim", "num_codes", "downs", "n_res")
GROUPS = 8                                                      # nn.GroupNorm(8, hidden) in train_cvqvae.py


def load_ckpt(path, trust):
    try:
        import torch
    except ImportError:
        sys.exit("Missing dep: pip install torch   (only needed to read the .pt file)")
    try:
        ck = torch.load(path, map_location="cpu", weights_only=not trust)
    except Exception as e:
        sys.exit(f"torch.load failed ({type(e).__name__}: {e}).\n"
                 f"If you trust this file (you made it), retry with --trust-pickle.")
    if not isinstance(ck, dict) or "model" not in ck or "cfg" not in ck or "norm" not in ck:
        sys.exit("Not a train_cvqvae.py checkpoint: expected a dict with 'model', 'cfg' and 'norm'.")
    return ck


def to_np(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=np.float32)


def export_ckpt(ck, dst, dtype="f32"):
    """ck: dict with 'model' (name -> tensor/ndarray), 'cfg', 'norm' (+ optional 'dt', 'action_names')."""
    cfg, sd = ck["cfg"], ck["model"]
    miss = [k for k in REQUIRED_CFG if k not in cfg]
    if miss:
        sys.exit(f"cfg is missing keys: {miss}")
    h, emb, D, K = int(cfg["hidden"]), int(cfg["emb"]), int(cfg["code_dim"]), int(cfg["num_codes"])
    downs, n_res, n_act = int(cfg["downs"]), int(cfg["n_res"]), int(cfg["n_actions"])
    if h % GROUPS:
        sys.exit(f"hidden={h} is not divisible by {GROUPS} (GroupNorm)")
    T = {k: to_np(v) for k, v in sd.items() if hasattr(v, "shape") and len(getattr(v, "shape", ())) > 0}

    nch = int(T["in_scale"].shape[0])
    wdt = np.float32 if dtype == "f32" else np.float16
    out = {}
    used = set()

    def take(name):
        if name not in T:
            sys.exit(f"checkpoint is missing tensor '{name}' (cfg says hidden={h} downs={downs} n_res={n_res})")
        used.add(name); return T[name]

    def vec(dst, src=None, expect=None):
        v = take(src or dst).reshape(-1)
        if expect is not None and v.shape[0] != expect:
            sys.exit(f"{src or dst}: size {v.shape[0]} != expected {expect}")
        out[dst] = np.ascontiguousarray(v, dtype=np.float32)

    def lin(dst, n, k, src=None, f32=False):             # nn.Linear weight (out,in) -> [N,K]
        W = take((src or dst) + ".weight")
        if W.shape != (n, k): sys.exit(f"{src or dst}.weight shape {W.shape} != {(n, k)}")
        out[dst + ".weight"] = np.ascontiguousarray(W, dtype=np.float32 if f32 else wdt)
        vec(dst + ".bias", (src or dst) + ".bias", n)

    def conv(dst, n, cin, ksz):                          # Conv1d weight (out,in,k) -> [out, k*in], tap-major
        W = take(dst + ".weight")
        if W.shape != (n, cin, ksz): sys.exit(f"{dst}.weight shape {W.shape} != {(n, cin, ksz)}")
        out[dst + ".weight"] = np.ascontiguousarray(W.transpose(0, 2, 1).reshape(n, ksz * cin), dtype=wdt)
        vec(dst + ".bias", None, n)

    def film_res(p):
        vec(p + ".norm.weight", None, h); vec(p + ".norm.bias", None, h)
        lin(p + ".film", 2 * h, emb)
        conv(p + ".conv1", h, h, 3)
        conv(p + ".conv2", h, h, 1)

    # --- conditioning ---
    out["act_emb.weight"] = np.ascontiguousarray(take("act_emb.weight"), dtype=np.float32)
    if out["act_emb.weight"].shape != (n_act + 1, emb):
        sys.exit(f"act_emb.weight {out['act_emb.weight'].shape} != {(n_act + 1, emb)}")
    lin("cond_mlp.0", emb, emb); lin("cond_mlp.2", emb, emb)
    vec("in_scale", None, nch); vec("in_shift", None, nch)

    # --- encoder ---
    conv("enc.inp", h, nch, 3)
    for s in range(downs):
        for b in range(n_res): film_res(f"enc.stages.{s}.0.{b}")
        conv(f"enc.stages.{s}.1", h, h, 4)
    vec("enc.norm.weight", None, h); vec("enc.norm.bias", None, h)
    conv("enc.out", D, h, 3)

    # --- codebook (always F32) ---
    E = take("vq.embed")
    if E.shape != (K, D): sys.exit(f"vq.embed {E.shape} != {(K, D)}")
    out["vq.embed"] = np.ascontiguousarray(E, dtype=np.float32)

    # --- decoder ---
    conv("dec.inp", h, D, 3)
    for s in range(downs):
        for b in range(n_res): film_res(f"dec.stages.{s}.0.{b}")
        conv(f"dec.stages.{s}.1", h, h, 3)
    vec("dec.norm.weight", None, h); vec("dec.norm.bias", None, h)
    conv("dec.out", nch, h, 3)

    # --- normalisation statistics ---
    nm = ck["norm"]
    for key in ("mean", "std", "dstd"):
        if key not in nm: sys.exit(f"norm['{key}'] missing")
        v = np.asarray(nm[key], dtype=np.float32).reshape(-1)
        if v.shape[0] != nch: sys.exit(f"norm['{key}'] has {v.shape[0]} values, expected {nch}")
        out["norm." + key] = v

    leftovers = sorted(set(T) - used - IGNORED)
    if leftovers:
        sys.exit("Refusing to export: unrecognised tensors in the checkpoint (this exporter does not know "
                 f"their meaning):\n  " + "\n  ".join(leftovers[:20]))
    for name in out:
        if len(name) > 63: sys.exit(f"tensor name too long for GGUF: {name}")

    w = gguf.GGUFWriter(dst, "nanity-cvqvae")
    for k, v in (("window", cfg["window"]), ("hidden", h), ("emb", emb), ("code_dim", D), ("num_codes", K),
                 ("downs", downs), ("n_res", n_res), ("n_actions", n_act), ("channels", nch), ("groups", GROUPS),
                 ("cond_enc", int(bool(cfg.get("cond_enc", True))))):
        w.add_uint32(f"nanity.cvqvae.{k}", int(v))
    if "dt" in ck: w.add_float32("nanity.cvqvae.dt", float(ck["dt"]))
    if "action_names" in ck: w.add_string("nanity.cvqvae.action_names", json.dumps(list(ck["action_names"])))
    for name, arr in out.items():
        w.add_tensor(name, arr)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    nparam = sum(int(np.prod(v.shape)) for v in out.values())
    print(f"wrote {dst}: tensors={len(out)} params={nparam/1e6:.2f}M dtype={dtype} "
          f"window={cfg['window']} codes={K}x{D} actions={n_act} (+null)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32",
                    help="storage type for conv/linear matrices (lossy if f16; opt-in)")
    ap.add_argument("--trust-pickle", action="store_true", help="torch.load(weights_only=False); only for files you made")
    a = ap.parse_args()
    export_ckpt(load_ckpt(a.src, a.trust_pickle), a.dst, a.dtype)


if __name__ == "__main__":
    main()
