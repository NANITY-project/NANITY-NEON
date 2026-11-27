#!/usr/bin/env python3
"""
stream_tokenize.py

Stream a Hugging Face dataset -> tokenize in RAM with the Rust `tokenizers`
library -> upload fixed-size token shards to a HF dataset repo.

* Nothing is written to disk (only a tiny state.json is cached by huggingface_hub on resume).
* Resumable: progress (dataset position + counters) is committed to the repo in the
  SAME commit as each shard, so the repo is always consistent. Re-run the same command
  to continue after a crash / Ctrl-C.
* Pipeline: main thread streams + tokenizes, a background thread uploads, so tokenizing
  the next shard overlaps with uploading the previous one.

Shard format: flat token ids, little-endian, uint16 if vocab <= 65536 else uint32,
documents separated by the EOS id. Read later with
    np.memmap(path, dtype=np.uint16, mode="r")
(meta.json in the repo records the dtype, vocab size and EOS id).

Install:
    pip install -U tokenizers datasets huggingface_hub numpy
    pip install -U "huggingface_hub[hf_xet]"      # optional, faster uploads
Login:
    hf auth login        (or set the HF_TOKEN environment variable)

Example:
    python stream_tokenize.py \
        --dataset HuggingFaceFW/fineweb-edu --config sample-100BT --text-field text \
        --tokenizer ./tokenizer.json --eos-token "<|endoftext|>" \
        --repo-id YOURNAME/nectar8-tokens --prefix fineweb-edu \
        --max-tokens 16e9
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import argparse
import itertools
import json
import queue
import sys
import threading
import time

import numpy as np


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def load_tokenizer(spec):
    from tokenizers import Tokenizer

    if os.path.isfile(spec):
        return Tokenizer.from_file(spec)
    return Tokenizer.from_pretrained(spec)  # HF repo id containing tokenizer.json


def tokenize_batch(tok, texts, eos_id, dtype):
    """Tokenize a list of documents (Rust, multi-threaded) -> flat array with EOS after each doc."""
    encs = tok.encode_batch(texts, add_special_tokens=False)
    return np.fromiter(
        itertools.chain.from_iterable([*e.ids, eos_id] for e in encs),
        dtype=dtype,
    )


class Uploader(threading.Thread):
    """Uploads (shard, state) pairs in one commit each, with retry/backoff."""

    def __init__(self, api, repo_id, prefix, retries=12):
        super().__init__(daemon=True)
        self.api, self.repo_id, self.prefix, self.retries = api, repo_id, prefix, retries
        self.q = queue.Queue(maxsize=1)  # at most 1 shard waiting while 1 uploads
        self.error = None
        self.shards_done = 0
        self.bytes_done = 0
        self.upload_secs = 0.0

    def submit(self, item):
        while True:
            if self.error:
                raise RuntimeError(self.error)
            try:
                self.q.put(item, timeout=2)
                return
            except queue.Full:
                continue

    def run(self):
        from huggingface_hub import CommitOperationAdd

        while True:
            item = self.q.get()
            if item is None:
                return
            idx, arr, state = item
            data = arr.tobytes()
            path = f"{self.prefix}/{self.prefix}_{idx:05d}.bin"
            for attempt in range(self.retries):
                try:
                    t = time.time()
                    self.api.create_commit(
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        operations=[
                            CommitOperationAdd(path_in_repo=path, path_or_fileobj=data),
                            CommitOperationAdd(
                                path_in_repo=f"{self.prefix}/state.json",
                                path_or_fileobj=json.dumps(state).encode(),
                            ),
                        ],
                        commit_message=f"{path} ({arr.size:,} tokens)",
                    )
                    self.upload_secs += time.time() - t
                    self.shards_done += 1
                    self.bytes_done += len(data)
                    break
                except Exception as e:  # network errors, rate limits, ...
                    wait = min(600, 5 * 2**attempt)
                    print(f"[upload] {path} failed ({type(e).__name__}: {e}); retrying in {wait}s", flush=True)
                    time.sleep(wait)
            else:
                self.error = f"giving up on {path} after {self.retries} attempts"
                return
            del data, arr


# ----------------------------------------------------------------------------
# main loop (kept separate from main() so it can be tested with fakes)
# ----------------------------------------------------------------------------
def run(args, ds, tok, api, state):
    vocab = tok.get_vocab_size()
    dtype = np.uint16 if vocab <= 65536 else np.uint32
    eos_id = tok.token_to_id(args.eos_token)
    if eos_id is None:
        sys.exit(f"EOS token {args.eos_token!r} is not in the tokenizer vocabulary")

    shard_idx = state["next_shard"] if state else 0
    total_tokens = state["tokens"] if state else 0
    total_docs = state["docs"] if state else 0
    if state:
        ds.load_state_dict(state["ds_state"])
        print(f"[resume] shard {shard_idx}, {total_tokens:,} tokens, {total_docs:,} docs already done", flush=True)
    committed = ds.state_dict() if state else None

    up = Uploader(api, args.repo_id, args.prefix)
    up.start()

    buf, buf_n = [], 0

    def flush():
        nonlocal buf, buf_n, shard_idx
        if not buf_n:
            return
        arr = np.concatenate(buf)
        snap = {
            "next_shard": shard_idx + 1,
            "tokens": total_tokens,
            "docs": total_docs,
            "ds_state": committed,
            "dataset": args.dataset,
            "config": args.config,
        }
        up.submit((shard_idx, arr, snap))
        shard_idx += 1
        buf, buf_n = [], 0

    it = iter(ds)
    t0 = last_print = time.time()
    session_tokens = 0
    try:
        while not up.error:
            docs = list(itertools.islice(it, args.batch_docs))
            if not docs:
                print("[done] dataset exhausted", flush=True)
                break
            pending = ds.state_dict()
            texts = [d[args.text_field] for d in docs if d.get(args.text_field)]
            if texts:
                arr = tokenize_batch(tok, texts, eos_id, dtype)
                buf.append(arr)
                buf_n += arr.size
                total_tokens += arr.size
                session_tokens += arr.size
            total_docs += len(docs)
            committed = pending  # state now matches everything in `buf`
            if buf_n >= args.shard_tokens:
                flush()
            now = time.time()
            if now - last_print >= 15:
                last_print = now
                print(
                    f"[{now - t0:7.0f}s] {total_tokens / 1e9:7.3f}B tokens | "
                    f"{session_tokens / (now - t0) / 1e6:5.2f}M tok/s tokenized | "
                    f"{up.shards_done} shards uploaded | buffer {buf_n / 1e6:.0f}M tok",
                    flush=True,
                )
            if args.max_tokens and total_tokens >= args.max_tokens:
                print("[done] reached --max-tokens", flush=True)
                break
    except KeyboardInterrupt:
        print("\n[stop] Ctrl-C: uploading the partial shard, please wait (Ctrl-C again to abort)...", flush=True)

    try:
        if not up.error:
            flush()
            up.submit(None)
    except RuntimeError:
        pass
    up.join()
    if up.error:
        sys.exit(f"ERROR: {up.error}. Re-run the same command to resume.")
    print(
        f"[finished] {total_tokens:,} tokens total, {up.shards_done} shards uploaded this run "
        f"({up.bytes_done / 1e9:.2f} GB)",
        flush=True,
    )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def fetch_state(api, args):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        p = hf_hub_download(
            args.repo_id, f"{args.prefix}/state.json", repo_type="dataset", force_download=True
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="HF dataset id, e.g. HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default=None, help="dataset config/subset name, e.g. sample-100BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--tokenizer", required=True, help="path to tokenizer.json, or a HF repo id")
    ap.add_argument("--eos-token", required=True, help="document-separator token, must exist in the tokenizer")
    ap.add_argument("--repo-id", required=True, help="destination HF dataset repo, e.g. you/nectar8-tokens")
    ap.add_argument("--prefix", required=True, help="folder inside the repo, e.g. fineweb-edu")
    ap.add_argument("--shard-tokens", type=lambda s: int(float(s)), default=100_000_000,
                    help="approx tokens per shard (default 100M = 200MB as uint16). Avoid tiny shards: HF rate-limits commits")
    ap.add_argument("--batch-docs", type=int, default=1000, help="documents tokenized per batch")
    ap.add_argument("--max-tokens", type=lambda s: int(float(s)), default=0,
                    help="stop after this many tokens in total (across resumes), e.g. 16e9; 0 = whole dataset")
    ap.add_argument("--public", action="store_true", help="create the repo as public (default: private)")
    ap.add_argument("--fresh", action="store_true", help="ignore saved progress and start over (overwrites shards)")
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi

    tok = load_tokenizer(args.tokenizer)
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)

    state = None
    if not args.fresh:
        state = fetch_state(api, args)
        if state is None:
            existing = [f for f in api.list_repo_files(args.repo_id, repo_type="dataset")
                        if f.startswith(f"{args.prefix}/{args.prefix}_")]
            if existing:
                sys.exit(f"Repo already has {len(existing)} shards under '{args.prefix}/' but no state.json. "
                         f"Refusing to overwrite; use a different --prefix or pass --fresh.")

    ds = load_dataset(args.dataset, name=args.config, split=args.split, streaming=True)
    if not hasattr(ds, "state_dict"):
        sys.exit("Your `datasets` version has no IterableDataset.state_dict(); run: pip install -U datasets")

    if state is None:
        vocab = tok.get_vocab_size()
        meta = {
            "dtype": "uint16" if vocab <= 65536 else "uint32",
            "vocab_size": vocab,
            "eos_token": args.eos_token,
            "eos_id": tok.token_to_id(args.eos_token),
            "tokenizer": os.path.basename(args.tokenizer),
            "source_dataset": args.dataset,
            "source_config": args.config,
            "format": "flat token ids, documents separated by eos_id",
        }
        api.upload_file(path_or_fileobj=json.dumps(meta, indent=2).encode(),
                        path_in_repo=f"{args.prefix}/meta.json", repo_id=args.repo_id, repo_type="dataset")
        if os.path.isfile(args.tokenizer):  # keep the exact tokenizer next to the shards
            api.upload_file(path_or_fileobj=args.tokenizer, path_in_repo="tokenizer.json",
                            repo_id=args.repo_id, repo_type="dataset")

    run(args, ds, tok, api, state)


if __name__ == "__main__":
    main()
