"""
packed_data.py -- fixed-shape packed batches for the fast trainer.

* Reads the prepare_data.py .bin format STREAMING through mmap (no list of
  Example objects, no 16 forked DataLoader workers each touching refcounts of
  millions of Python objects -> no host-RAM blowup).
* Packs documents into rows of exactly T tokens (bounded best-fit, ~1-2% pad)
  and stores them on disk as three flat memmaps (ids u32 / flags u8 / seg u16),
  cached next to the data file.
* Every batch has the SAME shape [B, T] -> torch.compile sees one static shape,
  no padding to the longest row, no recompiles.
* LABELS ARE SHIFTED: logits at position t predict token t+1. The label for
  position t is ids[t+1] iff token t+1 is loss-bearing (mask) AND belongs to the
  same document as t. (The old collate_fn used labels[t] = ids[t], i.e. it asked
  position t to output the token it was just given.)
* Everything per-batch (labels, valid indices, per-document positions,
  cu_seqlens) is computed on the CPU in a background thread, so the training
  loop never syncs on the GPU to find out shapes.

flags bit0 = token is a loss target (mask), bit1 = token is final-answer.
seg    = document index inside the row; PAD_SEG marks the padded tail.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
import queue
import struct
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch

MAGIC = b"NANTYBN1"
PAD_SEG = 0xFFFF
IGNORE = -100


# --------------------------------------------------------------------------- #
# document sources
# --------------------------------------------------------------------------- #

class BinDocSource:
    """Random access to a prepare_data.py .bin file via mmap (zero-copy)."""

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        mm = self._mm
        if mm[:8] != MAGIC:
            raise ValueError(f"{path}: not a NANITY .bin file")
        (hlen,) = struct.unpack_from("<I", mm, 8)
        self.header = json.loads(mm[12:12 + hlen].decode("utf-8"))
        self._w = 2 if self.header.get("id_typecode", "I") == "H" else 4
        self._idt = np.uint16 if self._w == 2 else np.uint32
        offs, pos, size = [], 12 + hlen, len(mm)
        while pos + 4 <= size:
            (n,) = struct.unpack_from("<I", mm, pos)
            end = pos + 4 + n * (self._w + 2)
            if end > size:
                break  # truncated tail record
            offs.append(pos)
            pos = end
        self.offsets = np.asarray(offs, dtype=np.int64)

    def __len__(self):
        return len(self.offsets)

    def get(self, i: int):
        pos = int(self.offsets[i])
        (n,) = struct.unpack_from("<I", self._mm, pos)
        p = pos + 4
        ids = np.frombuffer(self._mm, dtype=self._idt, count=n, offset=p)
        p += n * self._w
        mask = np.frombuffer(self._mm, dtype=np.uint8, count=n, offset=p)
        ans = np.frombuffer(self._mm, dtype=np.uint8, count=n, offset=p + n)
        return ids, mask, ans


class ExampleListSource:
    """Adapter for a list[Example] (the JSONL / ConversationDataset path)."""

    def __init__(self, examples):
        self.examples = examples
        self.header = {}

    def __len__(self):
        return len(self.examples)

    def get(self, i: int):
        ex = self.examples[i]
        idt = np.uint16 if ex.ids.typecode == "H" else np.uint32
        return (np.frombuffer(ex.ids, dtype=idt),
                np.frombuffer(ex.mask, dtype=np.uint8),
                np.frombuffer(ex.is_answer, dtype=np.uint8))


# --------------------------------------------------------------------------- #
# packing
# --------------------------------------------------------------------------- #

class _Row:
    __slots__ = ("ids", "fl", "seg", "used", "ndoc")

    def __init__(self, T, pad_id):
        self.ids = np.full(T, pad_id, np.uint32)
        self.fl = np.zeros(T, np.uint8)
        self.seg = np.full(T, PAD_SEG, np.uint16)
        self.used = 0
        self.ndoc = 0


def _pack(source, order, T, pad_id, prefix: Path, open_rows: int = 32, log=print):
    """Bounded best-fit packing of `order` (doc indices) into rows of T tokens.
    Writes <prefix>.ids.u32 / .flags.u8 / .seg.u16. Returns (n_rows, n_real_tokens)."""
    files = [open(f"{prefix}.{ext}", "wb") for ext in ("ids.u32", "flags.u8", "seg.u16")]
    n_rows = n_real = n_trunc = 0
    rows: list[_Row] = []

    def flush(r: _Row):
        nonlocal n_rows
        r.ids.tofile(files[0]); r.fl.tofile(files[1]); r.seg.tofile(files[2])
        n_rows += 1

    for k, di in enumerate(order):
        ids, mask, ans = source.get(int(di))
        n = len(ids)
        if n == 0:
            continue
        if n > T:
            n_trunc += 1
            ids, mask, ans, n = ids[:T], mask[:T], ans[:T], T
        best, best_left = None, None
        for r in rows:
            left = T - r.used - n
            if left >= 0 and (best is None or left < best_left):
                best, best_left = r, left
        if best is None:
            if len(rows) >= open_rows:
                fullest = max(rows, key=lambda r: r.used)
                rows.remove(fullest)
                flush(fullest)
            best = _Row(T, pad_id)
            rows.append(best)
        u = best.used
        best.ids[u:u + n] = ids
        m = (mask != 0).astype(np.uint8)
        best.fl[u:u + n] = m | (((ans != 0).astype(np.uint8) & m) << 1)
        best.seg[u:u + n] = best.ndoc
        best.used += n
        best.ndoc += 1
        n_real += n
        if (k + 1) % 500_000 == 0:
            log(f"[pack] {k + 1}/{len(order)} docs -> {n_rows} rows")
    for r in rows:
        flush(r)
    for f in files:
        f.close()
    if n_trunc:
        log(f"[pack] WARNING: {n_trunc} docs longer than T={T} were truncated")
    return n_rows, n_real


class PackedRows:
    def __init__(self, prefix: Path, T: int):
        self.T = T
        self.ids = np.memmap(f"{prefix}.ids.u32", dtype=np.uint32, mode="r").reshape(-1, T)
        self.fl = np.memmap(f"{prefix}.flags.u8", dtype=np.uint8, mode="r").reshape(-1, T)
        self.seg = np.memmap(f"{prefix}.seg.u16", dtype=np.uint16, mode="r").reshape(-1, T)

    def __len__(self):
        return self.ids.shape[0]


def build_or_load(source, T: int, pad_id: int, seed: int, val_split: float, val_max_docs: int,
                  fingerprint: str, cache_dir: Optional[str], log=print):
    """Returns (train_rows, val_rows_or_None). Cached on disk under cache_dir."""
    key = hashlib.sha256(f"{fingerprint}|{T}|{pad_id}|{seed}|{val_split}|{val_max_docs}|v1".encode()).hexdigest()[:12]
    base = Path(cache_dir or ".") / f"packed_{key}"
    base.parent.mkdir(parents=True, exist_ok=True)
    tr, va, meta = Path(f"{base}.train"), Path(f"{base}.val"), Path(f"{base}.meta.json")
    if meta.exists():
        m = json.loads(meta.read_text())
        log(f"[pack] reusing cache {base.name}: {m['train_rows']} train rows, {m['val_rows']} val rows")
    else:
        n = len(source)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = min(int(n * val_split), val_max_docs) if val_split > 0 else 0
        val_docs, train_docs = perm[:n_val], perm[n_val:]
        log(f"[pack] packing {len(train_docs)} train docs (+{n_val} val) into rows of {T} tokens ...")
        t_rows, t_real = _pack(source, train_docs, T, pad_id, tr, log=log)
        v_rows = v_real = 0
        if n_val:
            v_rows, v_real = _pack(source, val_docs, T, pad_id, va, log=log)
        m = dict(train_rows=t_rows, train_real=t_real, val_rows=v_rows, val_real=v_real, T=T)
        meta.write_text(json.dumps(m))
        log(f"[pack] done: {t_rows} train rows, {t_real:,} real tokens "
            f"({100 * t_real / max(t_rows * T, 1):.1f}% fill); {v_rows} val rows")
    train = PackedRows(tr, T)
    val = PackedRows(va, T) if m["val_rows"] > 0 else None
    return train, val, m


def used_vocab_mask(rows: PackedRows, vocab_size: int, extra_ids=()):
    """bool [V]: ids that occur anywhere in rows (for --freeze-unseen-vocab)."""
    counts = np.zeros(vocab_size, dtype=np.int64)
    step = 4096
    for i in range(0, len(rows), step):
        counts += np.bincount(np.asarray(rows.ids[i:i + step]).ravel(), minlength=vocab_size)[:vocab_size]
    used = counts > 0
    for t in extra_ids:
        if 0 <= t < vocab_size:
            used[t] = True
    return torch.from_numpy(used)


# --------------------------------------------------------------------------- #
# batches
# --------------------------------------------------------------------------- #

def batch_from_arrays(ids, fl, seg, doc_aware: bool, need_seg: bool = False):
    """ids [B,T] u32, fl [B,T] u8, seg [B,T] u16 -> dict of CPU tensors."""
    B, T = ids.shape
    ids64 = ids.astype(np.int64)
    same = seg[:, 1:] == seg[:, :-1]
    valid = ((fl[:, 1:] & 1) == 1) & same & (seg[:, 1:] != PAD_SEG)
    labels = np.full((B, T), IGNORE, np.int64)
    labels[:, :-1] = np.where(valid, ids64[:, 1:], IGNORE)
    ans = np.zeros((B, T), np.bool_)
    ans[:, :-1] = valid & ((fl[:, 1:] & 2) != 0)
    valid_idx = np.flatnonzero(labels.ravel() != IGNORE)

    is_start = np.ones((B, T), np.bool_)
    is_start[:, 1:] = ~same
    out = dict(
        input_ids=torch.from_numpy(ids64),
        labels=torch.from_numpy(labels),
        ans_valid=torch.from_numpy(ans.ravel()[valid_idx]),
        valid_idx=torch.from_numpy(valid_idx),
        n_real=int((seg != PAD_SEG).sum()),
        n_valid=int(valid_idx.size),
    )
    if doc_aware:
        ar = np.broadcast_to(np.arange(T), (B, T))
        start = np.maximum.accumulate(np.where(is_start, ar, 0), axis=1)
        out["pos"] = torch.from_numpy((ar - start).astype(np.int64))
        starts = np.flatnonzero(is_start.ravel())
        cu = np.concatenate([starts, [B * T]]).astype(np.int32)
        out["cu"] = torch.from_numpy(cu)
        out["max_seqlen"] = int(np.diff(cu).max())
    if need_seg:
        out["seg"] = torch.from_numpy(seg.astype(np.int32))
    return out


class PackedLoader:
    """Background-thread producer of pinned CPU batches. Deterministic and
    resumable: epoch e uses permutation(seed + e); pass start_micro to skip
    the micro-batches already consumed."""

    def __init__(self, rows: PackedRows, batch: int, seed: int, doc_aware: bool,
                 need_seg: bool = False, shuffle: bool = True, start_micro: int = 0,
                 pin: bool = True, depth: int = 6, once: bool = False):
        self.rows, self.B, self.seed = rows, batch, seed
        self.doc_aware, self.need_seg, self.shuffle = doc_aware, need_seg, shuffle
        self.pin, self.once = pin, once
        self.per_epoch = len(rows) // batch
        if self.per_epoch == 0:
            raise ValueError(f"only {len(rows)} packed rows, batch={batch}")
        self.start = start_micro
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _make(self, idx):
        idx = np.sort(idx)  # sorted reads are friendlier to the page cache; order inside a batch is irrelevant
        b = batch_from_arrays(np.asarray(self.rows.ids[idx]), np.asarray(self.rows.fl[idx]),
                              np.asarray(self.rows.seg[idx]), self.doc_aware, self.need_seg)
        if self.pin and torch.cuda.is_available():
            b = {k: (v.pin_memory() if torch.is_tensor(v) else v) for k, v in b.items()}
        return b

    def _run(self):
        try:
            micro = self.start
            while not self._stop.is_set():
                epoch, pos = divmod(micro, self.per_epoch)
                order = (np.random.default_rng(self.seed + epoch).permutation(len(self.rows))
                         if self.shuffle else np.arange(len(self.rows)))
                while pos < self.per_epoch and not self._stop.is_set():
                    self.q.put(self._make(order[pos * self.B:(pos + 1) * self.B]))
                    pos += 1
                    micro += 1
                if self.once:
                    break
            self.q.put(None)
        except Exception as e:  # surface in the consumer
            self.q.put(e)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self):
        self._stop.set()


def to_device(b: dict, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
