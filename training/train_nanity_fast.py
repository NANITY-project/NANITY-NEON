#!/usr/bin/env python3
"""
train_nanity_fast.py -- high-MFU trainer for NANITY (single GPU, built for MI300X).

Lives NEXT TO train_nanity_fixed.py and imports its tokenizer loader, HF sync and
GGUF export (so those keep working unchanged). Checkpoints are interchangeable
with the old trainer (same parameter names).

What is different from train_nanity_fixed.py (details in each module's docstring):
  1. next-token labels are SHIFTED (the old collate_fn trained logits[t] -> ids[t]).
  2. fixed-shape packed rows (no pad-to-longest, static shapes, no dynamic=True),
     per-document positions + cu_seqlens when flash_attn is installed.
  3. fp32 master weights + bf16 compute (old: pure-bf16 params + AdamW).
  4. compiled pre/post-attention blocks (compiled once, shared by all layers),
     eager attention (flash_attn varlen, or SDPA flash with manual GQA expand).
  5. fused chunked output-projection + CE: never materialises [N, 200064] logits.
  6. no gradient checkpointing by default (--ckpt-layers N / --grad-ckpt to opt in).
  7. --steps / --warmup-steps are OPTIMIZER steps (old: micro-batches). --tokens works too.
  8. --bench N: synthetic-data throughput/MFU/peak-memory probe, no dataset needed.

Not supported here (use train_nanity_fixed.py): LoRA, online --teacher KL.

Typical:
  # 1) probe the config (5 minutes, no data needed)
  python3 train_nanity_fast.py --preset 4b --bench 30 --batch 2 --max-seq-len 8192
  # 2) train
  python3 train_nanity_fast.py --preset 4b --phase warmup --data data/pretrain.bin \
      --out checkpoints/nanity_4b --tokens 12e9 --batch 2 --grad-accum 32 --lr 3e-4 \
      --warmup-steps 500 --save-every 500
"""
from __future__ import annotations

import os

# Must be set before the first GPU op. setdefault -> your own exports win.
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HIP_FORCE_DEV_KERNARG", "1")      # lower kernel-launch latency on MI300-class GPUs

import argparse
import atexit
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_here = Path(__file__).parent
sys.path.insert(0, str(_here))

from modeling_nanity import NanityConfig                       # noqa: E402
import train_nanity_fixed as tnf                               # noqa: E402  (tokenizer, HFSync, export_gguf)
from nanity_fast import FastNanity, flops_per_token, peak_tflops_for   # noqa: E402
from fused_ce import fused_linear_ce, linear_ce_per_token      # noqa: E402
import packed_data as pdata                                    # noqa: E402


def cosine_lr(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * (step + 1) / max(warmup, 1)
    if step >= total:
        return lr_min
    p = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * p))


# --------------------------------------------------------------------------- #
# checkpoints: small bf16 weights file (export/HF) + full fp32 resume state
# --------------------------------------------------------------------------- #

def save_all(model, opt, step, loss_val, out_dir: Path, cfg, meta, hf, save_state, upload_state):
    out_dir.mkdir(parents=True, exist_ok=True)
    wpath = out_dir / f"ckpt_{step:07d}.pt"
    sd = {k: (v.detach().to(torch.bfloat16) if v.is_floating_point() else v)
          for k, v in model.state_dict().items()}
    torch.save({"step": step, "loss": loss_val, "model": sd,
                "nanity_config": cfg.__dict__, "train_meta": meta}, wpath)
    for old in sorted(out_dir.glob("ckpt_[0-9]*.pt"))[:-3]:
        old.unlink()
    print(f"[ckpt] saved {wpath} ({wpath.stat().st_size / 1e9:.1f} GB, bf16 weights)")
    if hf is not None:
        hf.upload_async(wpath)
    spath = None
    if save_state:
        spath = out_dir / f"state_{step:07d}.pt"
        tmp = spath.with_suffix(".tmp")
        torch.save({"step": step, "loss": loss_val, "master": model.state_dict(),
                    "optimizer": opt.state_dict(), "nanity_config": cfg.__dict__,
                    "train_meta": meta}, tmp)
        tmp.replace(spath)
        for old in sorted(out_dir.glob("state_[0-9]*.pt"))[:-1]:
            old.unlink()
        print(f"[ckpt] saved {spath} ({spath.stat().st_size / 1e9:.1f} GB, fp32 master + Adam)")
        if hf is not None and upload_state:
            hf.upload_async(spath)
    return wpath


# --------------------------------------------------------------------------- #
# one micro-batch: forward + loss (shared by training, eval and bench)
# --------------------------------------------------------------------------- #

def micro_loss(model, b, args, chunk):
    h = model.forward_hidden(b["input_ids"], b.get("pos"), b.get("seg"), b.get("cu"),
                             b.get("max_seqlen", 0))
    hv = h.reshape(-1, h.shape[-1]).index_select(0, b["valid_idx"])
    lab = b["labels"].reshape(-1).index_select(0, b["valid_idx"])
    ans = b["ans_valid"]
    tw = 1.0 + (args.answer_loss_weight - 1.0) * ans.float()
    scale = tw / tw.sum().clamp(min=1e-8)
    loss, per_tok = fused_linear_ce(hv, model.output_weight(), lab, scale, chunk=chunk,
                                    label_smoothing=args.label_smoothing)
    return loss, per_tok, ans


@torch.no_grad()
def evaluate(model, rows, args, device, doc_aware, need_seg, max_batches):
    was = model.training
    model.eval()
    loader = pdata.PackedLoader(rows, args.batch, 0, doc_aware, need_seg, shuffle=False,
                                pin=(device == "cuda"), once=True)
    tot, thk, ansl = [], [], []
    for i, b in enumerate(loader):
        if i >= max_batches:
            loader.close()
            break
        b = pdata.to_device(b, device)
        h = model.forward_hidden(b["input_ids"], b.get("pos"), b.get("seg"), b.get("cu"),
                                 b.get("max_seqlen", 0))
        hv = h.reshape(-1, h.shape[-1]).index_select(0, b["valid_idx"])
        lab = b["labels"].reshape(-1).index_select(0, b["valid_idx"])
        pt = linear_ce_per_token(hv, model.output_weight(), lab, args.ce_chunk)
        ans = b["ans_valid"]
        if pt.numel():
            tot.append(pt.mean())
            if (~ans).any():
                thk.append(pt[~ans].mean())
            if ans.any():
                ansl.append(pt[ans].mean())
    if was:
        model.train()
    f = lambda xs: torch.stack(xs).mean().item() if xs else float("nan")
    return {"loss": f(tot), "think_loss": f(thk), "answer_loss": f(ansl)}


# --------------------------------------------------------------------------- #

def setup_tunableop(mode: str):
    if mode == "off" or not torch.cuda.is_available():
        return
    try:
        import torch.cuda.tunable as tunable
        tunable.enable(True)
        tunable.tuning_enable(mode == "tune")
        tunable.set_filename("nanity_tunableop.csv", insert_device_ordinal=True)
        print(f"[tunableop] mode={mode} -> nanity_tunableop*.csv "
              f"({'tuning new GEMM shapes, first steps are slow' if mode == 'tune' else 'reusing saved results'})")
    except Exception as e:
        print(f"[tunableop] unavailable on this torch build: {e!r}")


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed_all(args.seed)
        gpu = torch.cuda.get_device_name(0)
        print(f"[hw] {gpu}  torch={torch.__version__} hip={getattr(torch.version, 'hip', None)}")
        cd = torch.bfloat16
    else:
        if not args.allow_cpu:
            sys.exit("[hw] no GPU visible to torch (pass --allow-cpu for a smoke test)")
        device, gpu, cd = "cpu", "cpu", torch.float32
        print("[hw] CPU smoke-test mode (fp32)")
    setup_tunableop(args.tunableop)

    # ---- config / resume -------------------------------------------------
    ck = None
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = NanityConfig(**ck["nanity_config"])
    else:
        cfg = {"4b": NanityConfig.nanity_4b, "1_5b": NanityConfig.nanity_1_5b,
               "tiny": NanityConfig.nanity_tiny}[args.preset]()
    T = args.max_seq_len or cfg.context_length
    if T > cfg.context_length:
        sys.exit(f"--max-seq-len {T} > context_length {cfg.context_length}; raise it in the config first")
    B, accum = args.batch, args.grad_accum

    ckpt_layers = cfg.n_layer if args.grad_ckpt else args.ckpt_layers
    model = FastNanity(cfg, attn=args.attn, compute_dtype=cd, ckpt_layers=ckpt_layers).to(device)
    start_step = 0
    if ck is not None:
        sd = ck["master"] if "master" in ck else ck["model"]
        model.load_state_dict(sd)           # bf16 -> fp32 copy if it is a weights-only file
        start_step = int(ck.get("step", 0)) if args.resume_step is None else args.resume_step
        print(f"[resume] weights from {args.resume} (step {start_step}, "
              f"{'full fp32 master' if 'master' in ck else 'bf16 weights only'})")
    model.train()
    doc_aware = model.doc_aware
    need_seg = model.attn_backend == "masked"
    print(f"[attn] backend={model.attn_backend} doc_isolation={'yes' if doc_aware else 'NO (rows attend across documents)'}"
          f"  ckpt_layers={ckpt_layers}/{cfg.n_layer}")
    if device == "cuda" and not args.no_compile:
        print(f"[compile] compiling pre/post-attention blocks (mode={args.compile_mode}); first step is slow")
        model.enable_compile(args.compile_mode)

    n_params = sum(p.numel() for p in model.parameters())
    fpt = flops_per_token(cfg, T)
    peak = args.peak_tflops or peak_tflops_for(gpu)
    print(f"[model] {n_params / 1e9:.3f}B params, {fpt / 1e9:.1f} GFLOP/token (T={T}), "
          f"peak={peak or 'unknown'} TFLOPs")

    # ---- optimizer ---------------------------------------------------------
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if (p.ndim < 2 or "token_embd" in n or "output" in n or "norm" in n) else decay).append(p)
    groups = [{"params": decay, "weight_decay": args.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    try:
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, args.beta2), eps=1e-6, fused=True)
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, args.beta2), eps=1e-6)
    if ck is not None and "optimizer" in ck:
        try:
            opt.load_state_dict(ck["optimizer"])
            print("[resume] optimizer state restored")
        except Exception as e:
            print(f"[resume] optimizer state NOT restored ({e}); Adam restarts from zero")
    del ck

    # ---- benchmark mode (no data) -------------------------------------------
    if args.bench:
        return bench(args, model, opt, cfg, device, T, fpt, peak, doc_aware, need_seg)

    # ---- data ----------------------------------------------------------------
    if not args.data:
        sys.exit("--data is required")
    tok = tnf.load_tokenizer(args.tokenizer)
    pad_id = tok.pad_token_id or 0

    def open_source(path):
        if pdata.MAGIC == open(path, "rb").read(8):
            return pdata.BinDocSource(path)
        print(f"[data] {path} is JSONL -> tokenizing with the old ConversationDataset path")
        ds = tnf.ConversationDataset(path, tok, max_len=T, tokenizer_source=args.tokenizer)
        return pdata.ExampleListSource(ds.examples)

    def fp(path):
        st = os.stat(path)
        return f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}|{args.tokenizer}"

    src = open_source(args.data)
    print(f"[data] {len(src)} documents in {args.data}")
    cache_dir = args.cache_dir or str(Path(args.data).parent)
    if args.val_data:
        train_rows, _, tm = pdata.build_or_load(src, T, pad_id, args.seed, 0.0, 0, fp(args.data), cache_dir)
        vsrc = open_source(args.val_data)
        val_rows, _, _ = pdata.build_or_load(vsrc, T, pad_id, args.seed, 0.0, 0, fp(args.val_data), cache_dir)
    else:
        train_rows, val_rows, tm = pdata.build_or_load(src, T, pad_id, args.seed, args.val_split,
                                                       args.val_max_docs, fp(args.data), cache_dir)
    n_rows = len(train_rows)

    used_mask = None
    if args.freeze_unseen_vocab:
        used_mask = pdata.used_vocab_mask(train_rows, cfg.vocab_size, tok.all_special_ids).to(device)
        print(f"[vocab] {int(used_mask.sum())}/{cfg.vocab_size} vocab rows appear in the data; "
              f"gradient of the rest is zeroed")

    # ---- step budget -----------------------------------------------------------
    tokens_per_step = B * T * accum
    steps = args.steps
    if args.tokens:
        steps = math.ceil(args.tokens / tokens_per_step)
    steps = steps or 50000
    steps_per_epoch = max(1, (n_rows // B) // accum)
    epochs = steps / steps_per_epoch
    print(f"[budget] {tokens_per_step:,} tokens/optimizer-step ({B} rows x {T} x accum {accum}); "
          f"{steps} steps = {steps * tokens_per_step / 1e9:.2f}B tokens = {epochs:.2f} epochs over "
          f"{tm['train_real'] / 1e9:.2f}B real tokens ({steps_per_epoch} steps/epoch)")
    if args.max_epochs > 0 and epochs > args.max_epochs:
        steps = int(steps_per_epoch * args.max_epochs)
        print(f"[budget] clipped to --max-epochs {args.max_epochs}: {steps} steps (pass --max-epochs 0 to disable)")

    meta = {"data_path": args.data, "data_sha256": None, "tokenizer_hf": args.tokenizer,
            "phase": args.phase, "teacher_hf": None, "seed": args.seed,
            "optimizer": {"type": "AdamW", "lr": args.lr, "betas": [0.9, args.beta2], "eps": 1e-6,
                          "warmup_steps": args.warmup_steps, "weight_decay": args.weight_decay},
            "resumed_from": args.resume, "resumed_from_sha256": None,
            "framework": f"torch=={torch.__version__}", "trainer": "train_nanity_fast"}

    hf = None
    if args.hf_repo:
        hf = tnf.HFSync(args.hf_repo, token=args.hf_token, private=not args.hf_public)
        atexit.register(hf.wait)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = pdata.PackedLoader(train_rows, B, args.seed, doc_aware, need_seg,
                                start_micro=start_step * accum, pin=(device == "cuda"))
    it = iter(loader)
    z = lambda: torch.zeros((), device=device)
    s_loss, s_think, c_think, s_ans, c_ans = z(), z(), z(), z(), z()
    real_tokens, last_gn, last_loss = 0, z(), float("nan")
    best, bad, best_step = float("inf"), 0, start_step
    lr_min = args.lr * args.min_lr_ratio
    t_log, skipped = time.time(), 0
    opt.zero_grad(set_to_none=True)
    print(f"[train] phase={args.phase} steps {start_step}->{steps}  lr={args.lr:.2e}")

    for step in range(start_step, steps):
        lr = cosine_lr(step, args.warmup_steps, steps, args.lr, lr_min)
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(accum):
            b = next(it)
            real_tokens += b["n_real"]
            b = pdata.to_device(b, device)
            loss, per_tok, ans = micro_loss(model, b, args, args.ce_chunk)
            (loss / accum).backward()
            with torch.no_grad():
                s_loss += loss.detach()
                s_think += per_tok[~ans].sum(); c_think += (~ans).sum()
                s_ans += per_tok[ans].sum();    c_ans += ans.sum()
        if used_mask is not None:
            g = model.token_embd.weight.grad
            if g is not None:
                g[~used_mask] = 0
            if model.output is not None and model.output.weight.grad is not None:
                model.output.weight.grad[~used_mask] = 0
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        last_gn = gn
        if bool(torch.isfinite(gn)):
            opt.step()
        else:
            skipped += 1
            print(f"[warn] step {step + 1}: non-finite grad norm, update skipped ({skipped} so far)")
        opt.zero_grad(set_to_none=True)
        done = step + 1

        if done % args.log_every == 0 or done == steps:
            n = args.log_every if done % args.log_every == 0 else done % args.log_every
            last_loss = (s_loss / (n * accum)).item()
            th = (s_think / c_think.clamp(min=1)).item()
            an = (s_ans / c_ans.clamp(min=1)).item()
            dt = time.time() - t_log
            tps = real_tokens / dt
            mfu = f"{100 * tps * fpt / (peak * 1e12):.1f}%" if peak else "n/a"
            mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            eta_h = (steps - done) * (dt / n) / 3600
            print(f"[step {done:7d}/{steps}] loss={last_loss:.4f} think={th:.4f} answer={an:.4f} "
                  f"gnorm={last_gn.item():.2f} lr={lr:.2e} tok/s={tps:,.0f} MFU={mfu} "
                  f"mem={mem:.0f}GB tokens={done * tokens_per_step / 1e9:.3f}B eta={eta_h:.1f}h", flush=True)
            for t in (s_loss, s_think, c_think, s_ans, c_ans):
                t.zero_()
            real_tokens, t_log = 0, time.time()

        if val_rows is not None and done % args.val_every == 0:
            m = evaluate(model, val_rows, args, device, doc_aware, need_seg, args.val_batches)
            print(f"[val  {done:7d}] loss={m['loss']:.4f} think={m['think_loss']:.4f} answer={m['answer_loss']:.4f}")
            track = m["answer_loss"] if not math.isnan(m["answer_loss"]) else m["loss"]
            if track < best - args.early_stop_min_delta:
                best, bad, best_step = track, 0, done
            else:
                bad += 1
                if args.early_stop_patience > 0 and bad >= args.early_stop_patience:
                    print(f"[early-stop] no val improvement in {bad} checks (best step {best_step})")
                    save_all(model, opt, done, last_loss, out_dir, cfg, meta, hf, args.save_state, args.hf_upload_state)
                    break

        if done % args.save_every == 0 or done == steps:
            save_all(model, opt, done, last_loss, out_dir, cfg, meta, hf, args.save_state, args.hf_upload_state)
    loader.close()
    print("[train] done")

    if not args.no_auto_export:
        final = sorted(out_dir.glob("ckpt_[0-9]*.pt"))[-1]
        tnf.export_gguf(str(final), str(out_dir / "nanity_final.gguf"),
                        tokenizer_source_gguf=args.tokenizer_gguf,
                        tokenizer_hf=None if args.tokenizer_gguf else args.tokenizer,
                        quant=args.export_quant)


# --------------------------------------------------------------------------- #

def bench(args, model, opt, cfg, device, T, fpt, peak, doc_aware, need_seg):
    B, accum = args.batch, args.grad_accum
    rng = np.random.default_rng(0)
    batches = []
    for _ in range(4):  # synthetic packed rows: random tokens, docs of 512..T/2 tokens
        ids = rng.integers(0, cfg.vocab_size, size=(B, T), dtype=np.uint32)
        seg = np.zeros((B, T), np.uint16)
        for r in range(B):
            pos, d = 0, 0
            while pos < T:
                n = int(rng.integers(512, max(T // 2, 513)))
                seg[r, pos:pos + n] = d
                pos, d = pos + n, d + 1
        fl = np.full((B, T), 3, np.uint8)
        batches.append(pdata.batch_from_arrays(ids, fl, seg, doc_aware, need_seg))
    warm = max(3, args.bench // 5)
    total = args.bench + warm
    print(f"[bench] B={B} T={T} accum={accum}: {warm} warmup + {args.bench} timed micro-batches "
          f"(first steps include compile{' / TunableOp tuning' if args.tunableop == 'tune' else ''})")
    tokens = 0
    t0 = None
    for i in range(total):
        if i == warm:
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0, tokens = time.time(), 0
        b = pdata.to_device(batches[i % 4], device)
        loss, _, _ = micro_loss(model, b, args, args.ce_chunk)
        (loss / accum).backward()
        tokens += b["n_real"]
        if (i + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    tps = tokens / dt
    mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    print(f"[bench] {tps:,.0f} tok/s  {tps * fpt / 1e12:.0f} TFLOPs"
          + (f"  MFU {100 * tps * fpt / (peak * 1e12):.1f}% of {peak:.0f}" if peak else "")
          + f"  {dt / args.bench:.3f} s/micro-batch  peak mem {mem:.0f} GB  (loss {loss.item():.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = ap.add_argument
    a("--preset", choices=["4b", "1_5b", "tiny"], default="4b")
    a("--phase", choices=["warmup", "distill", "finetune"], default="warmup",
      help="label only (plain CE in all phases here)")
    a("--data"); a("--val-data", dest="val_data"); a("--out", default="checkpoints/nanity")
    a("--resume", help="state_*.pt (full) or ckpt_*.pt / old-trainer .pt (weights; optimizer best-effort)")
    a("--resume-step", dest="resume_step", type=int, default=None)
    a("--tokenizer", default="microsoft/Phi-4-mini-instruct")
    a("--tokenizer-gguf", dest="tokenizer_gguf")
    a("--cache-dir", dest="cache_dir", help="where packed row files are cached (default: next to --data)")
    # budget
    a("--steps", type=int, default=None, help="OPTIMIZER steps")
    a("--tokens", type=float, default=None, help="total token budget (alternative to --steps), e.g. 12e9")
    a("--batch", type=int, default=2, help="rows of --max-seq-len tokens per micro-batch")
    a("--grad-accum", dest="grad_accum", type=int, default=16)
    a("--max-seq-len", dest="max_seq_len", type=int, default=None)
    a("--max-epochs", dest="max_epochs", type=float, default=4.0)
    # optimisation
    a("--lr", type=float, default=3e-4); a("--min-lr-ratio", dest="min_lr_ratio", type=float, default=0.1)
    a("--warmup-steps", dest="warmup_steps", type=int, default=500, help="OPTIMIZER steps")
    a("--weight-decay", dest="weight_decay", type=float, default=0.1)
    a("--beta2", type=float, default=0.95); a("--grad-clip", dest="grad_clip", type=float, default=1.0)
    a("--label-smoothing", dest="label_smoothing", type=float, default=0.0)
    a("--answer-loss-weight", dest="answer_loss_weight", type=float, default=1.0)
    a("--freeze-unseen-vocab", dest="freeze_unseen_vocab", action="store_true")
    a("--seed", type=int, default=42)
    # speed
    a("--attn", choices=["auto", "flash_varlen", "sdpa", "masked"], default="auto")
    a("--ckpt-layers", dest="ckpt_layers", type=int, default=0, help="checkpoint the first N layers only")
    a("--grad-ckpt", dest="grad_ckpt", action="store_true", help="checkpoint ALL layers (old default)")
    a("--no-compile", dest="no_compile", action="store_true")
    a("--compile-mode", dest="compile_mode", default="default",
      help="torch.compile mode, e.g. max-autotune-no-cudagraphs (slow to compile; try it in --bench)")
    a("--ce-chunk", dest="ce_chunk", type=int, default=4096, help="tokens per fused-CE chunk")
    a("--tunableop", choices=["off", "tune", "use"], default="off",
      help="ROCm TunableOp GEMM autotuning: run once with 'tune', then train with 'use'")
    a("--peak-tflops", dest="peak_tflops", type=float, default=0.0)
    a("--bench", type=int, default=0, help="N timed micro-batches on synthetic data, then exit")
    a("--allow-cpu", dest="allow_cpu", action="store_true")
    # eval / logging / saving
    a("--val-split", dest="val_split", type=float, default=0.05)
    a("--val-max-docs", dest="val_max_docs", type=int, default=2000)
    a("--val-every", dest="val_every", type=int, default=250)
    a("--val-batches", dest="val_batches", type=int, default=30)
    a("--early-stop-patience", dest="early_stop_patience", type=int, default=0)
    a("--early-stop-min-delta", dest="early_stop_min_delta", type=float, default=0.01)
    a("--log-every", dest="log_every", type=int, default=10, help="optimizer steps")
    a("--save-every", dest="save_every", type=int, default=500, help="optimizer steps")
    a("--no-save-state", dest="save_state", action="store_false", help="skip the 55GB fp32 resume state")
    a("--hf-repo", dest="hf_repo"); a("--hf-token", dest="hf_token")
    a("--hf-public", dest="hf_public", action="store_true")
    a("--hf-upload-state", dest="hf_upload_state", action="store_true",
      help="also upload the big fp32 resume state to the Hub")
    a("--no-auto-export", dest="no_auto_export", action="store_true")
    a("--export-quant", dest="export_quant", default="F16", choices=["F32", "F16", "Q4_0"])
    args = ap.parse_args()
    if not args.bench and not (args.steps or args.tokens):
        args.steps = None  # falls back to 50000 inside train()
    train(args)


if __name__ == "__main__":
    main()
