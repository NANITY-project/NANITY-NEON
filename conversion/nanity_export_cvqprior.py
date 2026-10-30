#!/usr/bin/env python3
"""
nanity_export_cvqprior.py — convert a NERVA cVQ-VAE *prior* checkpoint
(tools/train_cvqvae.py in the NERVA repo, e.g. runs/cvqvae/prior_best.pt) into a
NEON-loadable GGUF (general.architecture = "nanity-cvqprior", read by
runtime/motion/rawllm_cvqprior.hpp).

The prior is the small causal transformer (CodePrior) that samples the VQ-VAE's
discrete codes:   [ACTION] c1..c16 [SEP] c1..c16 [SEP] ...
The VAE (nanity_export_cvqvae.py) decodes those codes to motion; the prior is
what makes "generate new motion" possible without an input clip.

Checkpoint layout this script understands (dict saved by train_cvqvae.py):
    model   state_dict  (tok, pos, blocks.N.{n1,qkv,proj,n2,ff.0,ff.2}, ln, head)
    cfg     dict        (K, n_actions, n, windows, d, layers, heads, dropout[, goal_dim, n_styles])
    names   list[str]   action names (optional but strongly recommended)
    stride  int         frame stride between consecutive windows (optional; the
                        runtime/UI need it to stitch windows into a clip)
    dt      float       seconds per frame (optional)
    val_nll float       validation NLL of the checkpoint (optional, informational)
    goal_stats  dict    (priors trained with goal conditioning) mean/std (normalisation of the 4 goal numbers), turn_max,
                        speed_cruise, speed_max: what a planner needs to ask for goals the data actually contains
    goal_names  list    the goal numbers' names (dyaw, dx, dz, speed)
    style_names list    (priors trained with style ids) index -> style name; the last embedding row is the NULL style
Goal / style conditioning is optional: a prior trained without it (cfg has no goal_dim / n_styles, or both 0) exports exactly as before.
With it the extra tensors are goal_mlp.0 (Linear goal_dim->d), goal_mlp.2 (Linear d->d), goal_null (d) and style_emb.weight (n_styles+1, d).
Anything else in the state_dict is refused by name rather than silently dropped.

Layout decisions:
  * nn.Linear weights (out, in) are written as-is: GGUF shape [in, out], which is
    exactly what runtime/nn/rawllm_nn.hpp's Linear expects (row = one output).
  * Token / position embeddings, LayerNorm affine and all biases stay F32.
  * --dtype f16 re-encodes only the big matrices (qkv, proj, ff, head). It is lossy
    (the sampled distribution shifts slightly), so it is opt-in.
  * Dropout is inference-off and is not stored. Token ids are exactly the trainer's:
    codes 0..K-1 | SEP = K | actions K+1..K+n_actions | NULL = K+1+n_actions | PAD.

Usage:
    python3 nanity_export_cvqprior.py prior_best.pt cvqprior.gguf [--dtype f32|f16]
    pip install torch gguf numpy

Security: torch.load() is called with weights_only=True first (no arbitrary pickle
execution). If your checkpoint needs full unpickling (numpy scalars in 'stride' are a
common reason), pass --trust-pickle, and only do that for files you made yourself.
"""
import argparse, json, sys
import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("Missing deps: pip install gguf numpy torch")

REQUIRED_CFG = ("K", "n_actions", "n", "windows", "d", "layers", "heads")


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
    if not isinstance(ck, dict) or "model" not in ck or "cfg" not in ck:
        sys.exit("Not a train_cvqvae.py prior checkpoint: expected a dict with 'model' and 'cfg'.")
    if "K" not in ck["cfg"] or "windows" not in ck["cfg"]:
        sys.exit("This looks like a VAE checkpoint (cfg has no 'K'/'windows'); use nanity_export_cvqvae.py for it.")
    return ck


def to_np(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=np.float32)


def export_ckpt(ck, dst, dtype="f32"):
    """ck: dict with 'model' (name -> tensor/ndarray), 'cfg' (+ optional 'names', 'stride', 'dt', 'val_nll')."""
    cfg, sd = ck["cfg"], ck["model"]
    miss = [k for k in REQUIRED_CFG if k not in cfg]
    if miss:
        sys.exit(f"cfg is missing keys: {miss}")
    K, n_act, n, Lw = int(cfg["K"]), int(cfg["n_actions"]), int(cfg["n"]), int(cfg["windows"])
    d, nl, H = int(cfg["d"]), int(cfg["layers"]), int(cfg["heads"])
    if d % H:
        sys.exit(f"d={d} is not divisible by heads={H}")
    seq_len = Lw * (n + 1)
    V = K + 3 + n_act
    T = {k: to_np(v) for k, v in sd.items() if hasattr(v, "shape") and len(getattr(v, "shape", ())) > 0}

    wdt = np.float32 if dtype == "f32" else np.float16
    out, used = {}, set()

    def take(name):
        if name not in T:
            sys.exit(f"checkpoint is missing tensor '{name}' (cfg says layers={nl} d={d})")
        used.add(name); return T[name]

    def vec(name, expect):
        v = take(name).reshape(-1)
        if v.shape[0] != expect:
            sys.exit(f"{name}: size {v.shape[0]} != expected {expect}")
        out[name] = np.ascontiguousarray(v, dtype=np.float32)

    def lin(stem, nout, kin, bias=True, f32=False):      # nn.Linear weight (out,in) -> [N,K]
        W = take(stem + ".weight")
        if W.shape != (nout, kin):
            sys.exit(f"{stem}.weight shape {W.shape} != {(nout, kin)}")
        out[stem + ".weight"] = np.ascontiguousarray(W, dtype=np.float32 if f32 else wdt)
        if bias:
            vec(stem + ".bias", nout)

    def emb(name, rows):
        E = take(name)
        if E.shape != (rows, d):
            sys.exit(f"{name} shape {E.shape} != {(rows, d)}")
        out[name] = np.ascontiguousarray(E, dtype=np.float32)

    emb("tok.weight", V)
    emb("pos.weight", seq_len - 1)
    for i in range(nl):
        p = f"blocks.{i}"
        vec(p + ".n1.weight", d); vec(p + ".n1.bias", d)
        lin(p + ".qkv", 3 * d, d); lin(p + ".proj", d, d)
        vec(p + ".n2.weight", d); vec(p + ".n2.bias", d)
        lin(p + ".ff.0", 4 * d, d); lin(p + ".ff.2", d, 4 * d)
    vec("ln.weight", d); vec("ln.bias", d)
    lin("head", K + 1, d, bias=False)

    G, NS = int(cfg.get("goal_dim", 0) or 0), int(cfg.get("n_styles", 0) or 0)
    if G:
        lin("goal_mlp.0", d, G, f32=True); lin("goal_mlp.2", d, d, f32=True); vec("goal_null", d)     # tiny and it carries the conditioning: always F32
        gs = ck.get("goal_stats")
        if not gs or len(gs.get("mean", [])) != G or len(gs.get("std", [])) != G:
            sys.exit(f"cfg says goal_dim={G} but the checkpoint has no goal_stats (mean/std of {G} values): the goal numbers cannot be normalised without them")
    if NS:
        emb("style_emb.weight", NS + 1)
        sn = [str(x) for x in (ck.get("style_names") or [])]
        if sn and len(sn) != NS:
            sys.exit(f"'style_names' has {len(sn)} entries but cfg.n_styles = {NS}")

    leftovers = sorted(set(T) - used)
    if leftovers:
        sys.exit("Refusing to export: unrecognised tensors in the checkpoint (this exporter does not know "
                 "their meaning):\n  " + "\n  ".join(leftovers[:20]))
    for name in out:
        if len(name) > 63:
            sys.exit(f"tensor name too long for GGUF: {name}")

    names = [str(s) for s in ck.get("names", [])]
    if names and len(names) != n_act:
        sys.exit(f"'names' has {len(names)} entries but cfg.n_actions = {n_act}")

    w = gguf.GGUFWriter(dst, "nanity-cvqprior")
    for k, v in (("K", K), ("n_actions", n_act), ("n", n), ("windows", Lw), ("d", d),
                 ("layers", nl), ("heads", H), ("stride", int(ck.get("stride", 0) or 0))):
        w.add_uint32(f"nanity.cvqprior.{k}", int(v))
    if "dt" in ck:
        w.add_float32("nanity.cvqprior.dt", float(ck["dt"]))
    if "val_nll" in ck:
        w.add_float32("nanity.cvqprior.val_nll", float(ck["val_nll"]))
    if names:
        w.add_string("nanity.cvqprior.action_names", json.dumps(names))
    w.add_uint32("nanity.cvqprior.goal_dim", G); w.add_uint32("nanity.cvqprior.n_styles", NS)
    if G:
        gs = ck["goal_stats"]
        w.add_string("nanity.cvqprior.goal_names", json.dumps([str(x) for x in (ck.get("goal_names") or [f"g{i}" for i in range(G)])]))
        for i in range(G):                                                  # scalar keys: no array metadata needed in the loader
            w.add_float32(f"nanity.cvqprior.goal_mean.{i}", float(gs["mean"][i])); w.add_float32(f"nanity.cvqprior.goal_std.{i}", float(gs["std"][i]))
        for k in ("turn_max", "speed_cruise", "speed_max"):
            if k in gs:
                w.add_float32(f"nanity.cvqprior.goal_{k}", float(gs[k]))
    if NS and ck.get("style_names"):
        w.add_string("nanity.cvqprior.style_names", json.dumps([str(x) for x in ck["style_names"]]))
    for name, arr in out.items():
        w.add_tensor(name, arr)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    nparam = sum(int(np.prod(v.shape)) for v in out.values())
    print(f"wrote {dst}: tensors={len(out)} params={nparam/1e6:.2f}M dtype={dtype} "
          f"codes={K} per-window={n} context={Lw} windows layers={nl} d={d} heads={H} actions={n_act} (+null)"
          + (f" goal={G} numbers" if G else " no goal") + (f" styles={NS} (+null)" if NS else " no style")
          + ("" if ck.get("stride") else "   [no stride stored: the UI/stitcher will need it from the VAE dataset]"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32",
                    help="storage type for the big matrices (lossy if f16; opt-in)")
    ap.add_argument("--trust-pickle", action="store_true", help="torch.load(weights_only=False); only for files you made")
    a = ap.parse_args()
    export_ckpt(load_ckpt(a.src, a.trust_pickle), a.dst, a.dtype)


if __name__ == "__main__":
    main()
