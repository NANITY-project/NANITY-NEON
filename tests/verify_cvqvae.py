#!/usr/bin/env python3
"""
tests/verify_cvqvae.py — check the NEON cVQ-VAE runtime against the real PyTorch model, on YOUR checkpoint.

    python3 tests/verify_cvqvae.py --ckpt runs/cvqvae/best.pt --nerva /path/to/NERVA --binary ./cvqvae \
            [--gguf cvqvae.gguf] [--data data/vae/val_windows.npz] [--clips 8]

  --nerva   the NERVA repo checkout (its tools/train_cvqvae.py defines the model; used as the reference)
  --binary  runtime/motion/cvqvae_cli.cpp built with:
            g++ -std=c++20 -O2 -pthread -mavx2 -mfma -mf16c runtime/motion/cvqvae_cli.cpp -o cvqvae
  --gguf    an already exported file; if omitted the checkpoint is exported to a temp file first
  --data    optional val/train windows .npz (windows, valid, action); without it, random inputs are used

For each clip and for the real action AND the "null" label it compares, between PyTorch (eval mode) and NEON:
  1. pre-quantisation latents          2. chosen code indices          3. decoder output given the PyTorch codes
Exit code 0 = all within tolerance (float32 rounding level expected: ~1e-5 relative or better).
"""
import argparse, os, subprocess, sys, tempfile
import numpy as np


def run(binary, *args):
    r = subprocess.run([binary, *map(str, args)], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"{binary} failed: {' '.join(map(str, args))}\n{r.stderr}")
    return r.stdout


def parse_csv(txt):
    return np.array([[float(v) for v in l.split(",")] for l in txt.strip().splitlines()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--nerva", required=True); ap.add_argument("--binary", required=True)
    ap.add_argument("--gguf"); ap.add_argument("--data"); ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-3, help="max relative error (vs the output's max magnitude)")
    a = ap.parse_args()

    import torch
    sys.path.insert(0, os.path.join(a.nerva, "tools"))
    import train_cvqvae as T                                   # the reference model definition

    model, d, norm = T.load_ckpt(a.ckpt, "cpu")                # builds CVQVAE(**cfg), loads weights, vq._ready = True
    model.eval()
    cfg = d["cfg"]; nact = cfg["n_actions"]; W = cfg["window"]
    mean, std = np.asarray(norm["mean"], np.float64), np.asarray(norm["std"], np.float64)

    tmp = tempfile.mkdtemp(prefix="verify_cvqvae_")
    gguf = a.gguf
    if not gguf:
        gguf = os.path.join(tmp, "m.gguf")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "..", "conversion", "nanity_export_cvqvae.py"), a.ckpt, gguf], check=True)

    rng = np.random.default_rng(0)
    if a.data:
        z = np.load(a.data, allow_pickle=False)
        wins, acts = z["windows"].astype(np.float64), z["action"].astype(np.int64)
        sel = rng.choice(len(wins), size=min(a.clips, len(wins)), replace=False)
        clips = [((wins[i] - mean) / std, int(acts[i])) for i in sel]
        src = f"{len(clips)} real windows from {a.data}"
    else:
        clips = []
        for _ in range(a.clips):
            x = rng.standard_normal((W, 30)) * 0.6
            x[:, T.FLAGS] = (rng.random((W, len(T.FLAGS))) > 0.5).astype(float)       # flags are 0/1 in model space (mean 0, std 1)
            clips.append((x, int(rng.integers(0, nact))))
        src = f"{len(clips)} random inputs (pass --data for real windows)"
    print(f"checkpoint: {a.ckpt}  (epoch {d.get('epoch')}, val {d.get('best_val')})   inputs: {src}")

    worst = dict(z=0.0, y=0.0); code_total = code_match = 0; bad = []
    for ci, (xn, act) in enumerate(clips):
        xin = os.path.join(tmp, "x.csv"); np.savetxt(xin, xn, delimiter=",", fmt="%.9g")
        for label, ai in (("action", act), ("null", -1)):
            tact = nact if ai < 0 else ai
            with torch.no_grad():
                xt = torch.tensor(xn, dtype=torch.float32)[None]; at = torch.tensor([tact])
                z_t = model.encode(xt, at)                                           # [1,N,D]
                zq, idx_t, _, _ = model.vq(z_t)
                y_t = model.decode_codes(idx_t, at)                                  # [1,T,30] raw (flags = logits)
            z_t, idx_t, y_t = z_t[0].numpy().astype(np.float64), idx_t[0].numpy(), y_t[0].numpy().astype(np.float64)

            zf = os.path.join(tmp, "z.csv")
            codes = np.array(run(a.binary, gguf, "encode", "--in", xin, "--normalized", "--action", "null" if ai < 0 else ai, "--latents-out", zf).split(), dtype=int)
            z_n = np.loadtxt(zf, delimiter=",", ndmin=2)
            y_n = parse_csv(run(a.binary, gguf, "decode", "--normalized", "--action", "null" if ai < 0 else ai, "--codes", ",".join(map(str, idx_t))))

            ez = np.abs(z_n - z_t).max() / (np.abs(z_t).max() + 1e-12)
            ey = np.abs(y_n - y_t).max() / (np.abs(y_t).max() + 1e-12)
            worst["z"], worst["y"] = max(worst["z"], ez), max(worst["y"], ey)
            code_total += len(idx_t); code_match += int((codes == idx_t).sum())
            if (codes != idx_t).any():       # tolerable only if the two codes are float-rounding near-ties in PyTorch's own latents
                E = model.vq.embed.detach().numpy().astype(np.float64)
                for n in np.where(codes != idx_t)[0]:
                    dd = ((z_t[n] - E) ** 2).sum(1)
                    if abs(dd[codes[n]] - dd[idx_t[n]]) > 1e-3 * max(1.0, dd.min()): bad.append((ci, label, int(n)))
            print(f"  clip {ci} {label:6s} latent err {ez:.2e}   decode err {ey:.2e}   codes {int((codes == idx_t).sum())}/{len(idx_t)} equal")

    ok = worst["z"] < a.tol and worst["y"] < a.tol and not bad
    print(f"\nworst latent err {worst['z']:.2e}, worst decoder err {worst['y']:.2e}, code agreement {code_match}/{code_total}"
          + (f", NON-tie code mismatches: {bad}" if bad else ""))
    print("RESULT:", "PASS — NEON matches PyTorch" if ok else "FAIL")
    sys.exit(0 if ok else 1)


main()
