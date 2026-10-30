#!/usr/bin/env python3
"""
tests/verify_cvqprior.py — check the NEON cVQ-VAE prior runtime against a reference implementation.

    python3 tests/verify_cvqprior.py --ckpt runs/cvqvae/prior_best.pt --binary ./cvqprior [--nerva /path/to/NERVA] [--gguf cvqprior.gguf]
    python3 tests/verify_cvqprior.py --synthetic --binary ./cvqprior          # random weights, no torch/NERVA needed

  --binary    runtime/motion/cvqprior_cli.cpp built with:
              g++ -std=c++20 -O2 -pthread -mavx2 -mfma -mf16c runtime/motion/cvqprior_cli.cpp -o cvqprior
  --ckpt      a real prior_best.pt (read with torch.load, weights_only=True; --trust-pickle if yours needs more)
  --nerva     the NERVA checkout: its tools/train_cvqvae.py CodePrior + filter_logits become the reference (needs torch).
              Without it a float64 numpy twin of the same graph is the reference.
  --synthetic random weights of the default NERVA prior shape (or --K/--n/--windows/--d/--layers/--heads/--actions)
              add --goal (4 goal numbers + the null goal) and/or --styles N to test goal / style conditioning and classifier-free guidance

Checks, on several random token sequences (codes, SEPs, every action token incl. NULL):
  1. logits at EVERY position            2. the filtered next-token distribution (temperature / top-k / top-p)
  3. the C++ KV-cache path == its own full forward (cvqprior selftest), including the sliding-window rebuild
  4. (goal / style priors) logits with per-window goals and a style, the null goal, and the guided next-token logits of the incremental path
     (uncond + s * (cond - uncond), s = 1.7) against the reference; selftest repeats this through the sliding context
Exit code 0 = all within tolerance (float32 rounding level expected: ~1e-4 relative or better).
"""
import argparse, os, subprocess, sys, tempfile, importlib.util
import numpy as np


def run(binary, *args):
    r = subprocess.run([binary, *map(str, args)], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"{binary} failed: {' '.join(map(str, args))}\n{r.stdout}{r.stderr}")
    return r.stdout


def parse_csv(txt):
    return np.array([[float(v) for v in l.split(",")] for l in txt.strip().splitlines()])


# ----------------------------------------------------------------------------------------- numpy reference (float64)
def _ln(x, w, b):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * w + b


def _gelu(x):
    from math import erf
    return 0.5 * x * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


def np_forward(sd, cfg, tok, goal_slots=None, style=None):
    """goal_slots: [windows][goal_dim] NORMALISED goals (slot j = j-th window) or None = the null goal; style: id or None = the null style."""
    W = {k: np.asarray(v, np.float64) for k, v in sd.items()}
    d, H, nl = cfg["d"], cfg["heads"], cfg["layers"]; hd = d // H; T = len(tok)
    x = W["tok.weight"][tok] + W["pos.weight"][:T]
    G, NS, n, Lw = int(cfg.get("goal_dim", 0) or 0), int(cfg.get("n_styles", 0) or 0), cfg["n"], cfg["windows"]
    if G:
        if goal_slots is None:
            ge = np.tile(W["goal_null"], (T, 1))
        else:
            h = _gelu(np.asarray(goal_slots, np.float64) @ W["goal_mlp.0.weight"].T + W["goal_mlp.0.bias"])
            emb = h @ W["goal_mlp.2.weight"].T + W["goal_mlp.2.bias"]                            # [windows, d]
            ge = emb[np.minimum(np.arange(T) // (n + 1), Lw - 1)]
        x = x + ge
    if NS:
        x = x + W["style_emb.weight"][NS if style is None else style]
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for i in range(nl):
        p = f"blocks.{i}."
        h = _ln(x, W[p + "n1.weight"], W[p + "n1.bias"])
        qkv = (h @ W[p + "qkv.weight"].T + W[p + "qkv.bias"]).reshape(T, 3, H, hd)
        q, k, v = (qkv[:, j].transpose(1, 0, 2) for j in range(3))                   # [H,T,hd]
        a = q @ k.transpose(0, 2, 1) / np.sqrt(hd) + mask
        a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        y = (a @ v).transpose(1, 0, 2).reshape(T, d)
        x = x + y @ W[p + "proj.weight"].T + W[p + "proj.bias"]
        h = _gelu(_ln(x, W[p + "n2.weight"], W[p + "n2.bias"]) @ W[p + "ff.0.weight"].T + W[p + "ff.0.bias"])
        x = x + h @ W[p + "ff.2.weight"].T + W[p + "ff.2.bias"]
    return _ln(x, W["ln.weight"], W["ln.bias"]) @ W["head.weight"].T              # [T,K+1]


def np_filtered(lg, K, temperature, top_k, top_p):
    lg = lg[:K].astype(np.float64) / max(temperature, 1e-4)
    if top_k and top_k < K:
        kth = np.sort(lg)[::-1][top_k - 1]; lg = np.where(lg < kth, -np.inf, lg)
    if top_p < 1.0:
        si = np.argsort(-lg, kind="stable"); sl = lg[si]
        pr = np.exp(sl - sl.max()); pr /= pr.sum()
        rm = (np.cumsum(pr) - pr) > top_p
        keep = np.full(K, -np.inf); keep[si[~rm]] = sl[~rm]; lg = keep
    e = np.exp(lg - lg.max()); return e / e.sum()


def synthetic(a):
    rng = np.random.default_rng(0)
    K, n_act, n, Lw, d, nl, H = a.K, a.actions, a.n, a.windows, a.d, a.layers, a.heads
    V, S = K + 3 + n_act, Lw * (n + 1)
    sd = {"tok.weight": rng.standard_normal((V, d)) * 0.5, "pos.weight": rng.standard_normal((S - 1, d)) * 0.5,
          "ln.weight": 1 + 0.1 * rng.standard_normal(d), "ln.bias": 0.1 * rng.standard_normal(d), "head.weight": rng.standard_normal((K + 1, d)) * 0.3}
    for i in range(nl):
        p = f"blocks.{i}."
        for nm in ("n1", "n2"):
            sd[p + nm + ".weight"] = 1 + 0.1 * rng.standard_normal(d); sd[p + nm + ".bias"] = 0.1 * rng.standard_normal(d)
        for nm, (o, k) in {"qkv": (3 * d, d), "proj": (d, d), "ff.0": (4 * d, d), "ff.2": (d, 4 * d)}.items():
            sd[p + nm + ".weight"] = rng.standard_normal((o, k)) / np.sqrt(k) * 1.5; sd[p + nm + ".bias"] = 0.1 * rng.standard_normal(o)
    cfg = dict(K=K, n_actions=n_act, n=n, windows=Lw, d=d, layers=nl, heads=H, dropout=0.1)
    extra = {}
    if a.goal:
        G = 4; cfg["goal_dim"] = G
        sd["goal_mlp.0.weight"] = rng.standard_normal((d, G)) * 0.6; sd["goal_mlp.0.bias"] = 0.1 * rng.standard_normal(d)
        sd["goal_mlp.2.weight"] = rng.standard_normal((d, d)) / np.sqrt(d) * 1.5; sd["goal_mlp.2.bias"] = 0.1 * rng.standard_normal(d)
        sd["goal_null"] = rng.standard_normal(d) * 0.5
        extra.update(goal_names=["dyaw", "dx", "dz", "speed"],
                     goal_stats=dict(mean=[0.02, -0.01, 0.55, 0.9], std=[0.45, 0.3, 0.5, 0.6], turn_max=0.9, speed_cruise=1.2, speed_max=2.1))
    if a.styles:
        cfg["n_styles"] = a.styles
        sd["style_emb.weight"] = rng.standard_normal((a.styles + 1, d)) * 0.5
        extra["style_names"] = [f"style{i}" for i in range(a.styles)]
    sd = {k: v.astype(np.float32) for k, v in sd.items()}
    names = [f"act{i}" for i in range(n_act)]
    return dict(model=sd, cfg=cfg, names=names, stride=48, dt=1 / 30, val_nll=3.21, **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True); ap.add_argument("--ckpt"); ap.add_argument("--nerva"); ap.add_argument("--gguf")
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--trust-pickle", action="store_true")
    ap.add_argument("--K", type=int, default=512); ap.add_argument("--n", type=int, default=16); ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=6); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--actions", type=int, default=7); ap.add_argument("--seqs", type=int, default=6)
    ap.add_argument("--goal", action="store_true", help="synthetic: goal-conditioned prior"); ap.add_argument("--styles", type=int, default=0, help="synthetic: number of style ids")
    ap.add_argument("--tol", type=float, default=1e-4, help="max relative error (vs the logits' max magnitude)")
    a = ap.parse_args()
    if not a.ckpt and not a.synthetic:
        sys.exit("give --ckpt prior_best.pt or --synthetic")

    here = os.path.dirname(os.path.abspath(__file__))
    root = next((r for r in (os.path.join(here, ".."), os.path.join(here, "..", "..")) if os.path.exists(os.path.join(r, "conversion", "nanity_export_cvqprior.py"))), None)
    if root is None: sys.exit("cannot find conversion/nanity_export_cvqprior.py next to or above this script")
    spec = importlib.util.spec_from_file_location("nanity_export_cvqprior", os.path.join(root, "conversion", "nanity_export_cvqprior.py"))
    ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)

    ck = synthetic(a) if a.synthetic else ex.load_ckpt(a.ckpt, a.trust_pickle)
    cfg = ck["cfg"]; sd = ck["model"]; K, nact, n, Lw = cfg["K"], cfg["n_actions"], cfg["n"], cfg["windows"]
    tmp = tempfile.mkdtemp(prefix="verify_cvqprior_")
    gguf = a.gguf
    if not gguf:
        gguf = os.path.join(tmp, "p.gguf"); ex.export_ckpt(ck, gguf)
    print(run(a.binary, gguf, "info").strip())

    ref_torch = None
    if a.nerva and not a.synthetic:
        import torch
        sys.path.insert(0, os.path.join(a.nerva, "tools")); import train_cvqvae as T
        prior = T.CodePrior(**{k: v for k, v in cfg.items()}); prior.load_state_dict(sd); prior.eval()
        ref_torch = (torch, T, prior)
    sdn = {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)) for k, v in sd.items()}
    print("reference:", "PyTorch CodePrior from NERVA" if ref_torch else "numpy float64 twin")

    rng = np.random.default_rng(1)
    S1 = Lw * (n + 1) - 1
    worst_l = worst_p = worst_c = 0.0
    G, NS = int(cfg.get("goal_dim", 0) or 0), int(cfg.get("n_styles", 0) or 0)
    if G:
        gst = ck["goal_stats"]; gm = [float(np.float32(v)) for v in gst["mean"]]; gs_ = [float(np.float32(v)) for v in gst["std"]]
    for si in range(a.seqs):
        length = int(rng.integers(1, S1 + 1)) if si else S1                          # first one is a full-length context
        act_tok = K + 1 + int(rng.integers(0, nact + 1))                              # last value = NULL
        tok = np.concatenate([[act_tok], rng.integers(0, K + 1, size=length - 1)]).astype(np.int64)
        if ref_torch:
            torch, T, prior = ref_torch
            with torch.no_grad():
                lg_ref = prior(torch.tensor(tok)[None])[0].numpy().astype(np.float64)
        else:
            lg_ref = np_forward(sdn, cfg, tok)
        lg = parse_csv(run(a.binary, gguf, "logits", "--tokens", ",".join(map(str, tok))))
        el = np.abs(lg - lg_ref).max() / (np.abs(lg_ref).max() + 1e-12)
        worst_l = max(worst_l, el)
        # filtered distribution at the last position
        t_, k_, p_ = [(1.0, 0, 1.0), (0.7, 40, 1.0), (1.0, 0, 0.9), (0.9, 25, 0.85)][si % 4]
        pr = parse_csv(run(a.binary, gguf, "probs", "--tokens", ",".join(map(str, tok)), "--temperature", t_, "--top-k", k_, "--top-p", p_))[0]
        if ref_torch:
            torch, T, prior = ref_torch
            l = torch.tensor(lg_ref[-1:, :K]).float() / max(t_, 1e-4)
            pr_ref = torch.softmax(T.filter_logits(l, k_, p_), -1)[0].numpy().astype(np.float64)
        else:
            pr_ref = np_filtered(lg_ref[-1], K, t_, k_, p_)
        ep = np.abs(pr - pr_ref).max()
        worst_p = max(worst_p, ep)
        print(f"  seq {si} len {length:3d} act {act_tok - K - 1:2d}  logits err {el:.2e}   filtered-probs err {ep:.2e} (T={t_} k={k_} p={p_}, {int((pr > 0).sum())} codes kept)")
        if G or NS:
            nslots = (length - 1) // (n + 1) + 1
            raw = np.array(gm) + np.array(gs_) * rng.standard_normal((nslots, G)) if G else None
            sty = int(rng.integers(0, NS)) if NS else None
            gnorm = None
            if G:
                gnorm = np.zeros((Lw, G)); gnorm[:nslots] = (raw - np.array(gm, np.float32)) / np.array(gs_, np.float32)
            gfmt = lambda r: ";".join(",".join(f"{v:.9g}" for v in row) for row in r)
            extra_args = (["--goals", gfmt(raw)] if G else []) + (["--style", str(sty)] if NS else [])
            tk_s = ",".join(map(str, tok))
            # conditioned logits at every position (full forward with goals + style)
            lg_c = parse_csv(run(a.binary, gguf, "logits", "--tokens", tk_s, *extra_args))
            ref_c = np_forward(sdn, cfg, tok, gnorm, sty)
            e_c = np.abs(lg_c - ref_c).max() / (np.abs(ref_c).max() + 1e-12)
            # the null goal (+ style) path
            lg_n = parse_csv(run(a.binary, gguf, "logits", "--tokens", tk_s, *(["--style", str(sty)] if NS else [])))
            ref_n = np_forward(sdn, cfg, tok, None, sty)
            e_n = np.abs(lg_n - ref_n).max() / (np.abs(ref_n).max() + 1e-12)
            msg = f"           goal/style: conditioned err {e_c:.2e}   null-goal err {e_n:.2e}"
            worst_c = max(worst_c, e_c, e_n)
            if G:                                                                   # guided next-token logits through the incremental path
                s_cfg = 1.7
                lg_g = parse_csv(run(a.binary, gguf, "cfglogits", "--tokens", tk_s, *extra_args, "--cfg", s_cfg))[0]
                ref_g = ref_n[-1] + s_cfg * (ref_c[-1] - ref_n[-1])
                e_g = np.abs(lg_g - ref_g).max() / (np.abs(ref_g).max() + 1e-12)
                lg_g1 = parse_csv(run(a.binary, gguf, "cfglogits", "--tokens", tk_s, *extra_args))[0]                   # cfg 1 -> plain conditioned
                e_g1 = np.abs(lg_g1 - ref_c[-1]).max() / (np.abs(ref_c[-1]).max() + 1e-12)
                msg += f"   guided (s={s_cfg}) err {e_g:.2e}   unguided err {e_g1:.2e}"
                worst_c = max(worst_c, e_g, e_g1)
                # the goal must actually change the prediction (otherwise the test proves nothing)
                moved = np.abs(ref_c[-1] - ref_n[-1]).max() / (np.abs(ref_c[-1]).max() + 1e-12)
                msg += f"   (goal moves logits by {moved:.2f} rel)"
                if moved < 1e-3: worst_c = max(worst_c, 1.0)
            print(msg)
    st = subprocess.run([a.binary, gguf, "selftest"], capture_output=True, text=True)
    print(" ", (st.stdout + st.stderr).strip())
    ok = worst_l < a.tol and worst_p < 1e-4 and worst_c < a.tol and st.returncode == 0
    print(f"\nworst logits err {worst_l:.2e}, worst filtered-probs err {worst_p:.2e}" + (f", worst goal/style/guidance err {worst_c:.2e}" if (G or NS) else ""))
    print("RESULT:", "PASS — NEON matches the reference" if ok else "FAIL")
    sys.exit(0 if ok else 1)


main()
