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

Checks, on several random token sequences (codes, SEPs, every action token incl. NULL):
  1. logits at EVERY position            2. the filtered next-token distribution (temperature / top-k / top-p)
  3. the C++ KV-cache path == its own full forward (cvqprior selftest), including the sliding-window rebuild
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


def np_forward(sd, cfg, tok):
    W = {k: np.asarray(v, np.float64) for k, v in sd.items()}
    d, H, nl = cfg["d"], cfg["heads"], cfg["layers"]; hd = d // H; T = len(tok)
    x = W["tok.weight"][tok] + W["pos.weight"][:T]
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
    sd = {k: v.astype(np.float32) for k, v in sd.items()}
    cfg = dict(K=K, n_actions=n_act, n=n, windows=Lw, d=d, layers=nl, heads=H, dropout=0.1)
    names = [f"act{i}" for i in range(n_act)]
    return dict(model=sd, cfg=cfg, names=names, stride=48, dt=1 / 30, val_nll=3.21)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True); ap.add_argument("--ckpt"); ap.add_argument("--nerva"); ap.add_argument("--gguf")
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--trust-pickle", action="store_true")
    ap.add_argument("--K", type=int, default=512); ap.add_argument("--n", type=int, default=16); ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=6); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--actions", type=int, default=7); ap.add_argument("--seqs", type=int, default=6)
    ap.add_argument("--tol", type=float, default=1e-4, help="max relative error (vs the logits' max magnitude)")
    a = ap.parse_args()
    if not a.ckpt and not a.synthetic:
        sys.exit("give --ckpt prior_best.pt or --synthetic")

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("nanity_export_cvqprior", os.path.join(here, "..", "conversion", "nanity_export_cvqprior.py"))
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
    worst_l = worst_p = 0.0
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
    st = subprocess.run([a.binary, gguf, "selftest"], capture_output=True, text=True)
    print(" ", (st.stdout + st.stderr).strip())
    ok = worst_l < a.tol and worst_p < 1e-4 and st.returncode == 0
    print(f"\nworst logits err {worst_l:.2e}, worst filtered-probs err {worst_p:.2e}")
    print("RESULT:", "PASS — NEON matches the reference" if ok else "FAIL")
    sys.exit(0 if ok else 1)


main()
