# NEON — the NANITY Runtime

NEON is a from-scratch C++ inference runtime for **NANITY**, a single, fixed
transformer architecture. It is deliberately *not* a "load any GGUF" runtime
like llama.cpp: NEON only loads a model that declares
`general.architecture = "nanity"`, stored either as a **GGUF** file
(GPT-Generated Unified Format) or as an **`.nctr`** file, NANITY's own
container format. The training pipeline, the converters and the
specification are all designed around that one fixed shape on purpose.

This repository and its code are provided 'as is' without any warranty.

- Full technical contract (tensor names, metadata keys, container layouts):
  [`docs/NANITY_ARCHITECTURE_SPEC.md`](./docs/NANITY_ARCHITECTURE_SPEC.md)
- Beyond language models, NEON also runs a **vision encoder** (Vision
  Transformer) and **Motion VAE** (Variational AutoEncoder) models — see
  [Vision and motion models](#vision-and-motion-models).

SmolLM2 (1.7B parameters) has been tested on NEON: it produces coherent
text, follows prompts, and also works somewhat with the idle-loop feature.

> Licensed under the
> **[GNU Affero General Public License v3.0](./LICENSE)** (AGPLv3).

## Contents

- [What works right now](#what-works-right-now)
- [Vision and motion models](#vision-and-motion-models)
- [Known limitations](#known-limitations)
- [Out of scope](#out-of-scope)
- [Repository layout](#repository-layout)
- [Build](#build)
- [Testing](#testing)
- [Quick start](#quick-start)
- [Training your own NANITY model](#training-your-own-nanity-model)
- [Glossary](#glossary)

## What works right now

- **CPU inference.** No GPU or ROCm is needed to build or run. Validated
  end to end against real converted models (Llama-3.2-family and
  TinyLlama-family architectures load and generate coherent output).
- **GGUF loading**, including third-party models converted to NANITY
  format with [`conversion/nanity_convert.py`](./conversion/nanity_convert.py)
  (use its `--detect-only` flag to check compatibility before converting).
- **`.nctr` loading.** `NEON-3.cpp` reads the first four bytes of the model
  file, picks `GGUFLoader` or `NCTRLoader`, and runs the same generation
  path for both (`run_model<Loader>()`). `.nctr` export is done by
  `export_nctr()` in
  [`training/train_nanity_fixed.py`](./training/train_nanity_fixed.py).
- **Tokenization** for both byte-level BPE vocabularies (Byte-Pair
  Encoding; GPT-2/tiktoken family — Qwen, Llama-3, Phi) and SentencePiece
  vocabularies (Llama/TinyLlama family).
- **Idle-loop / companion-overlay mode** — continuous background
  generation, disk-backed context eviction and Hyprland vision hooks.
  Optional and off by default; enabled with `-DNANITY_ENABLE_IDLE_LOOP`.
  The core chat/generation path never needs it. The live key-value (KV)
  cache window (StreamingLLM-style attention sink plus shift,
  `ensure_cache_room()` in `runtime/NEON-3.cpp` and `kv_cache_shift()` in
  `runtime/core/rawllm_forward.hpp`) is bounded, so a fixed
  `--ctx-len` / `kv_window` never grows without limit, even with
  `--idle-max-tokens 0`.
- **zRAM compression tier** (`runtime/idle_loop/nectar_zram.hpp`; opt in
  with `--idle-compression` / `--idle-chunk-tokens`; needs
  `-DNANITY_ENABLE_IDLE_LOOP`). As chunks age out of the live window, a
  pluggable compressor can fold a summary of them permanently into the
  pinned implant region instead of dropping them.
  - `--idle-compression=text` works today: the model summarizes its own
    outgoing chunk, no training needed.
  - `--idle-compression=slots` (reserved KV-row compression, in the style
    of Gist Tokens / AutoCompressors / ICAE) has its core cache primitive,
    `kv_cache_compact()` in `runtime/core/rawllm_forward.hpp`, but is not
    wired into the idle loop yet. See that function's doc comment for the
    training and masking work still needed before it produces more than
    noise.

## Vision and motion models

NEON is no longer text-only. Two additional model families are supported
as C++ libraries, each with its own GGUF architecture tag and its own
exporter script:

| Model family | Runtime header | Exporter | `general.architecture` |
|---|---|---|---|
| **Vision encoder** (Vision Transformer tower: image → tokens in the language model's embedding space) | [`runtime/vision/rawllm_vit.hpp`](./runtime/vision/rawllm_vit.hpp) | [`conversion/nanity_export_vit.py`](./conversion/nanity_export_vit.py) | `nanity-vit` |
| **Motion VAE** (Motion Variational AutoEncoder: pose-by-pose character motion generation) | [`runtime/motion/rawllm_mvae.hpp`](./runtime/motion/rawllm_mvae.hpp) | [`conversion/nanity_export_mvae.py`](./conversion/nanity_export_mvae.py) | `nanity-mvae` |

Both are CPU-only, run on top of the shared neural-network kernels in
[`runtime/nn/rawllm_nn.hpp`](./runtime/nn/rawllm_nn.hpp), and read weights
in F32, F16, BF16 or Q8_0 directly from the memory-mapped GGUF file.

**Vision encoder.** Takes an RGB image (PPM bytes, exactly what the
idle-loop's screen capture already produces), resizes it with
PIL-compatible bilinear/bicubic resampling so Hugging Face reference
outputs can be matched, splits it into patches, runs the transformer
blocks, and projects the result to the language model's embedding width.
Supported towers are CLIP and SigLIP, with LLaVA-style (linear or
two-layer MLP) connectors or Idefics3/SmolVLM-style (pixel-shuffle plus
linear) connectors. Not supported: dynamic-resolution towers (Qwen2-VL /
NaViT) and cross-attention connectors (Flamingo / Perceiver resamplers).
Export presets: `siglip`, `clip`, `llava`.

**Motion VAE.** Implements the inference graphs from *Character
Controllers Using Motion VAEs* (Ling et al., 2020) as shipped in
`electronicarts/character-motion-vaes`: the mixture-of-experts pose model
(`PoseMixtureVAE`), the plain `PoseVAE`, the controller network (a small
multilayer perceptron, MLP), and the encoder for turning a motion-capture
frame into a latent vector. The experts of each layer are evaluated as a
single stacked matrix-vector product and then blended, which is
mathematically identical to blending the weights first but never builds
the mixed weight matrix. Lossy options (F16/BF16/Q8_0 re-encoding, top-k
expert selection) are opt-in and off by default.

**Integration status.** Both families are wired into `NEON-3.cpp` through
[`runtime/multimodal/neon_multimodal.hpp`](./runtime/multimodal/neon_multimodal.hpp),
and both run on the engine's own `ThreadPool` and the shared `nn` kernels.

*Vision.* `--vision-model <nanity-vit.gguf>` loads the tower at boot (also
under `--probe`). Its output width must equal the language model's `n_embd`
(export with a connector that projects to it); a mismatch is refused with a
message. Images are binary PPM (P6) or the literal `screen` (captured with
`grim`). The image tokens are fed into the residual stream in place of
token-table lookups (`fwd::EmbdOverride` in `rawllm_forward.hpp`), so they
take RoPE positions and KV-cache slots like text. Place an image with an
`<image>` marker in the prompt; with no marker, images go right after BOS.
Image positions carry content-derived sentinel ids, so cross-turn KV-prefix
reuse still works, and recent encodings are cached (re-sending an image every
turn costs a hash). Image tokens count against `kv_window`.
```
neon --model m.gguf --vision-model vit.gguf --image shot.ppm --prompt "what is this? <image>"
echo '{"messages":[{"role":"user","content":"<image> describe"}],"images":["a.ppm"]}' | neon --model m.gguf --vision-model vit.gguf --interactive
# images: ["path.ppm", "screen", {"path":..}, {"b64":..}, {"screen":true}]
```

*Motion.* `--motion-model` (repeatable) loads a pose VAE and/or a controller
MLP. In `--interactive` mode, `{"motion":{"op":...}}` requests are answered
with `{"motion":{...}}` and `{"done":true}`: `info`, `reset` (mean pose, or
`frame` / `frames`), `step` (`n`, `seed`, `temperature`, or `z`, `z_seq`, or
`obs_seq` through the controller), `decode`, `encode`. `{"capabilities":true}`
reports what is loaded. A NERVA conditional VQ-VAE (`nanity-cvqvae`, from `nanity_export_cvqvae.py`)
loads the same way and adds `cvq_encode` (frames -> codes), `cvq_decode`
(codes -> frames) and `cvq_recon`, each with an optional `action` (name, index
or null) and `normalized`. Frame counts must be a multiple of 4.

*Prior (generating motion without an input clip).* The cVQ-VAE's autoregressive
prior (`CodePrior` in NERVA's `tools/train_cvqvae.py`, checkpoint
`prior_best.pt`) is exported with `conversion/nanity_export_cvqprior.py` to a
`nanity-cvqprior` GGUF and loaded with another `--motion-model`. It is a small causal
transformer over the VAE's codes with a KV cache; sampling mirrors the trainer
(temperature, then top-k, then top-p). Ops: `cvq_sample` (`action`, `windows`,
`temperature`, `top_k`, `top_p`, `seed`, optional `"decode":true` to also run the
matching VAE and return `frames` per window), and for endless generation
`cvq_prior_reset` (`action`, `seed`) followed by one `cvq_prior_next` per window
(returns `codes`, the prior's `probs` and `entropy` per code, `stride`, optional
`frames`). `info` reports the prior and whether it matches the loaded VAE (code
count, codes per window, action list); `"decode":true` is refused when it does not.
A prior trained with goal numbers (`info.cvqprior.goal` is not null: `dyaw, dx, dz, speed` of a
window, raw numbers) steers by them instead of, or on top of, an action label: `cvq_sample`
takes `goal` / `goals`, `cvq_prior_reset` takes the same as a plan for the session (window *i*
uses entry *min(i, size-1)*; the reply's `planned_windows` is its length), and
`cvq_prior_next` takes `goal` for its own window, which wins over the plan (`null` = none for
that window). `"action":"null"` plus a goal is movement described by numbers only.
`tests/test_cvqprior_goal_ops.py` drives these through a real `neon --interactive`.
Windows are returned as sampled: overlapping/cross-fading them into a clip is the
caller's job (NERVA's `stitch`; the exporter stores `stride`).
```
neon --motion-model vae.gguf --motion-model prior.gguf --interactive     # no --model needed for motion-only requests
{"motion":{"op":"cvq_sample","action":"walk","windows":6,"temperature":0.9,"top_k":40,"seed":1,"decode":true}}
```
Standalone: `g++ -std=c++20 -O2 -pthread -mavx2 -mfma -mf16c runtime/motion/cvqprior_cli.cpp -o cvqprior`
(`info`, `sample [--vae VAE.gguf]`, `logits`, `probs`, `selftest`, `bench`);
`tests/verify_cvqprior.py` checks it against PyTorch (`--nerva`) or a numpy twin.

Motion works without `--model`: `--motion-info` / `--motion-rollout` (pose VAE) and, with
`--interactive`, the `{"motion":{...}}` ops above:
```
neon --motion-model pose.gguf --motion-info --motion-rollout 60 --motion-seed 1
```

Not yet wired: the idle loop (`--idle-vision`) still uses OCR text deltas. It
can call `g_vision` / `fwd::EmbdOverride` the same way, but that code path
needs `idle_loop/nectar_cot.hpp`, which is not in this repository.

## Known limitations

- **Vulkan backend** (`runtime/backends/rawllm_vulkan.hpp`): experimental.
  Matrix-vector only (no batched matrix-matrix path — NEON is single-user,
  single-prompt, so there is no batching workload to justify one). Enabled
  with `-DUSE_VULKAN`. Verified correct and benchmarked on real GPU
  hardware (NVIDIA T4): the optimized backend came in about 5–11% slower
  than cuBLAS's single-precision matrix-vector routine on the same
  hardware. The full validation process and results are in
  [`docs/VULKAN_BACKEND_NOTES.md`](./docs/VULKAN_BACKEND_NOTES.md) (and on
  the [`VULKAN-EXPERIMENTAL` branch](https://github.com/NANITY-project/NEON-Runtime-Beta-Variation/blob/VULKAN-EXPERIMENTAL/VULKAN_BACKEND_NOTES.md)).
  It is still **not wired into the generation hot path**:
  `--model ... --probe` on a Vulkan-enabled build initializes the backend
  and logs whether a usable device was found, but every token is still
  computed on the CPU path (`runtime/simd/rawllm_simd_dispatch.hpp`).
  Deciding when the GPU beats the CPU for a given tensor size, and routing
  `proj_all_positions()` through it, is the next piece of work.
- **AMD ROCm hardware** (the project's target GPU family): the ROCm build
  has produced byte-identical generation output to the CPU build on a
  converted TinyLlama GGUF on an MI300X, but broader testing is still
  pending.
- **`.nctr` models:** loading is wired in, but no large trained `.nctr`
  model has been run through NEON yet.
- Single maintainer, no independent security audit yet (see
  [`SECURITY.md`](./SECURITY.md)).

## Performance (CPU AVX2 Only Tested)
 <h2>CPU Performance — AVX2</h2>

<p>
  <strong>CPU:</strong> Intel Core i5-10400 @ 2.90 GHz<br>
  <strong>Model:</strong> Qwen2.5-0.5B, Q4_0<br>
  <strong>Test:</strong> 512 prompt tokens, 256 generated tokens, 5 repetitions<br>
  <strong>NEON:</strong> AVX2, FMA · <strong>llama.cpp:</strong> Release, AVX2, FMA, F16C, BMI2
</p>

<table>
  <thead>
    <tr>
      <th>Threads</th>
      <th>Runtime</th>
      <th>Prefill (tok/s) ↑</th>
      <th>Decode (tok/s) ↑</th>
      <th>Host RSS (MB) ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">6</td>
      <td><strong>NEON</strong></td>
      <td><strong>639.18 ± 10.78</strong></td>
      <td>45.90 ± 2.11</td>
      <td>757</td>
    </tr>
    <tr>
      <td>llama.cpp</td>
      <td>531.03 ± 10.16</td>
      <td><strong>52.18 ± 0.21</strong></td>
      <td>674</td>
    </tr>
    <tr>
      <td rowspan="2">12</td>
      <td><strong>NEON</strong></td>
      <td><strong>726.57 ± 4.26</strong></td>
      <td>42.38 ± 7.74</td>
      <td>757</td>
    </tr>
    <tr>
      <td>llama.cpp</td>
      <td>397.85 ± 19.58</td>
      <td>37.46 ± 13.64</td>
      <td><strong>674</strong></td>
    </tr>
  </tbody>
</table>

<h3>Relative Performance</h3>

<table>
  <thead>
    <tr>
      <th>Threads</th>
      <th>Prefill Ratio</th>
      <th>Decode Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>6</td>
      <td><strong>1.20× (+20.3%)</strong></td>
      <td>0.88× (−12.0%)</td>
    </tr>
    <tr>
      <td>12</td>
      <td><strong>1.83× (+82.6%)</strong></td>
      <td><strong>1.13× (+13.1%)</strong></td>
    </tr>
  </tbody>
</table>

<p>
  <sub>
    Prefill measures prompt-processing throughput; decode measures
    token-generation throughput. Ratios above 1.00× indicate higher
    NEON throughput. Results are specific to this hardware, model pair,
    build configuration, and benchmark run.
  </sub>
</p>

<h3>Memory &amp; Benchmark Methodology</h3>

- **Host RSS (MB)** is the Resident Set Size: the amount of physical system RAM currently occupied by a process, including resident model-weight pages, runtime allocations, and buffers. It is **system RAM, not GPU VRAM**. Lower RSS means a smaller resident memory footprint; it does not directly indicate inference speed.
- **Test conditions:** Both runtimes were tested on the same Intel Core i5-10400, using corresponding Qwen2.5-0.5B Q4_0 model files, matching prompt/output token targets, thread counts, and 5 repetitions.
- **Build configuration:** NEON uses `-O2 -mavx2 -mfma`; llama.cpp uses a Release build with AVX2, FMA, F16C, and BMI2 enabled. Compiler settings are therefore not identical.
- **Measurement:** Throughput is reported in tokens per second. NEON is timed through its interactive interface, while llama.cpp uses `llama-bench`.

Results reflect the tested revisions, build configurations, model files, and hardware. They should not be interpreted as universal performance guarantees.

## Out of scope

Two different kinds of "not here": things this project will never be, and
things that are simply sequenced after other work. Mixing the two is how a
focused runtime slowly turns into a second llama.cpp, so the line is
written down here.

**Permanently out of scope — not planned, not "later":**

- **Continuous batching / paged attention** (vLLM-style block-table KV
  cache, prefix sharing across concurrent requests). This is request
  scheduling and memory pooling, not a kernel you add. NEON is a
  single-user runtime (one companion, one conversation at a time).
- **Serving many models or many requests on one GPU.** NEON is not an
  inference server.
- **Mixture-of-Experts language models.** NANITY is one fixed, dense
  architecture on purpose (see the spec). (The Motion VAE's expert layers
  are a separate, small model family and not affected by this.)
- **GPU flash-attention** (finishing the composable_kernel `DeviceMHAFwd`
  stub in `runtime/backends/rawllm_rocm.hpp`). It is its own project. A
  CPU-side online-softmax attempt was tried and reverted after
  benchmarking showed it regressed at long context; see the git history of
  the attention loop in `runtime/core/rawllm_forward.hpp`.

**Deferred, not ruled out:**

- **K-quant (Q4_K / Q5_K / Q6_K) fused integer kernels.** Only Q4_0 and
  Q8_0 get the fast fused path today; K-quants fall back to dequantizing
  to F32. Adding more quantization formats on top of one that has not yet
  been proven to give reliably coherent, instruction-following output
  end to end means debugging two things at once, so this waits until
  Q4_0/Q8_0 are confirmed on a real instruct model.

## Repository layout

```
.
├── README.md  LICENSE  CONTRIBUTING.md  SECURITY.md
├── runtime/                 C++ inference runtime (NEON)
│   ├── NEON-3.cpp           command-line program, generation loop, tokenizer, format dispatch
│   ├── core/                loaders, forward pass, shared utilities
│   ├── simd/                CPU vector-instruction backends
│   ├── backends/            optional GPU backends (ROCm, Vulkan)
│   ├── nn/                  shared neural-network kernels (vision + motion)
│   ├── vision/              vision encoder (Vision Transformer)
│   ├── motion/              Motion VAE
│   └── idle_loop/           optional idle-loop / companion-overlay feature
├── training/                training pipeline and reference model (Python)
├── conversion/              model converters and exporters (Python)
├── tests/                   standalone tests
├── docs/                    specification and backend notes
```

**`runtime/core/` — loaders and forward pass**

| File | What it is |
|---|---|
| `rawllm_loader.hpp` | GGUF loader (memory-mapped), including metadata accessors used by the vision and motion models |
| `rawllm_nctr_loader.hpp` | `.nctr` loader (`NCTRLoader`) |
| `rawllm_forward.hpp` | Transformer forward pass: attention, rotary position embedding (RoPE), grouped-query attention (GQA), SwiGLU feed-forward, KV cache (including `kv_cache_compact()`, the zRAM-tier primitive) |
| `rawllm_repack.hpp` | Interleaved 8-row Q4_0/Q8_0 weight layouts and the matrix-matrix kernels that use them |
| `rawllm_common.hpp` | Shared types and helpers used across loaders and the forward pass |
| `rawllm_util.hpp` | Small standalone utilities, including the thread pool |
| `rawllm_json.hpp` | Minimal JSON parsing and writing for `--interactive` requests and responses |

**`runtime/simd/` — CPU vector-instruction (SIMD) backends**

| File | What it is |
|---|---|
| `rawllm_simd_scalar.hpp` | Portable fallback, and the correctness reference every vectorized backend is checked against |
| `rawllm_simd_avx2.hpp` | AVX2 + FMA dot-product and fused quantized dot-product kernels |
| `rawllm_simd_avx512.hpp` | AVX-512 kernels, with an AVX-512 VNNI fast path |
| `rawllm_simd_dispatch.hpp` | Picks AVX-512 > AVX2 > scalar at compile time; the only SIMD header that `rawllm_forward.hpp` includes |

**`runtime/backends/` — optional GPU backends**

| File | What it is |
|---|---|
| `rawllm_rocm.hpp` | ROCm/HIP backend (AMD MI300X target) |
| `rawllm_vulkan.hpp` | Vulkan compute backend (experimental; F32/Q4_0/Q8_0 matrix-vector only; shader sources live on the `VULKAN-EXPERIMENTAL` branch) |

**`runtime/nn/`, `runtime/vision/`, `runtime/motion/`**

| File | What it is |
|---|---|
| `nn/rawllm_nn.hpp` | Neural-network kernels: matrix-vector and matrix-matrix products, layer normalization, softmax, activations |
| `vision/rawllm_vit.hpp` | Vision encoder (Vision Transformer tower plus connector) |
| `motion/rawllm_mvae.hpp` | Motion VAE inference |
| `motion/rawllm_cvqprior.hpp` | cVQ-VAE prior: KV-cached causal transformer over the VAE codes, with temperature / top-k / top-p sampling (CLI: `motion/cvqprior_cli.cpp`) |

**`runtime/idle_loop/` — optional (`-DNANITY_ENABLE_IDLE_LOOP`)**

| File | What it is |
|---|---|
| `nectar_diskmem.hpp` | Disk-backed context eviction |
| `nectar_vision.hpp` | Hyprland screen-capture (vision) hooks |
| `nectar_splice.hpp` | Idle-loop generation splicing |
| `nectar_zram.hpp` | zRAM compression tier: chunk-pointer ring buffer plus pluggable compression backend |
| `nectar_cot.hpp` | Standby chain-of-thought cycle (`--idle-cot`) |

**`training/` — training pipeline (Python)**

| File | What it is |
|---|---|
| `train_nanity_fixed.py` | Training pipeline (warmup, distillation, fine-tuning; LoRA support) plus `export_gguf()` / `export_nctr()` |
| `modeling_nanity.py` | Reference PyTorch implementation of the NANITY architecture, with the `4b`, `1_5b` and `tiny` presets |
| `prepare_data.py` | Training-data preparation into pre-tokenized `.bin` files |
| `nanity_data_format.py` | `.bin` dataset format and shared chat-template code |

**`conversion/` — converters and exporters (Python)**

| File | What it is |
|---|---|
| `nanity_convert.py` | Converts third-party GGUF models into NANITY-conformant GGUF (`--detect-only` checks compatibility first) |
| `nanity_export_vit.py` | Exports a CLIP / SigLIP vision tower (plus optional connector) to a `nanity-vit` GGUF |
| `nanity_export_mvae.py` | Exports a Motion VAE checkpoint (safetensors) to a `nanity-mvae` GGUF |
| `nanity_export_cvqprior.py` | Exports a NERVA cVQ-VAE prior checkpoint (`prior_best.pt`) to a `nanity-cvqprior` GGUF |

**`tests/`, `docs/`**

| File | What it is |
|---|---|
| `tests/test_nctr_loader.cpp` | Standalone smoke test for the `.nctr` loader |
| `docs/NANITY_ARCHITECTURE_SPEC.md` | The specification — the full technical contract |
| `docs/VULKAN_BACKEND_NOTES.md` | Vulkan backend performance log and real-hardware (NVIDIA T4) validation results |
| `docs/CREDITS.md` | Inspirations and ideas taken/reimplemented from... |

Project meta files at the root: [`CONTRIBUTING.md`](./CONTRIBUTING.md)
(read before sending a pull request, especially the backend-review notes),
[`SECURITY.md`](./SECURITY.md) (vulnerability reporting, best effort, no
service-level agreement) and [`LICENSE`](./LICENSE) (AGPLv3).

## Build

All commands are run from the repository root. The runtime is a single
translation unit, `runtime/NEON-3.cpp`; all headers are found through
relative includes, so no `-I` flags are needed.

CPU only (no GPU needed):
```
g++ -std=c++20 -O2 -pthread runtime/NEON-3.cpp -o neon
```

AVX2 (detected from the compiler flags, no extra `-D` needed):
```
g++ -std=c++20 -O2 -pthread -mavx2 -mfma runtime/NEON-3.cpp -o neon
```

AVX-512 (opt-in with `-DUSE_AVX512`; `-mavx512vl` is required alongside
`-mavx512vnni`, because VNNI's 256-bit forms need the vector-length
extension — without it the build falls back to plain AVX-512F instead of
failing):
```
g++ -std=c++20 -O2 -pthread -mavx512f -mavx512bw -mavx512vnni -mavx512vl -DUSE_AVX512 runtime/NEON-3.cpp -o neon
```

With the optional idle-loop features:
```
g++ -std=c++20 -O2 -pthread -DNANITY_ENABLE_IDLE_LOOP runtime/NEON-3.cpp -o neon
```

ROCm (AMD GPU dispatch; not required for CPU inference). Needs a
HIP/ROCm toolchain (`hipcc`) and linking against `rocblas` and `amdhip64`.
The exact flags may differ on your ROCm install:
```
hipcc -std=c++20 -O2 -pthread -DUSE_ROCBLAS -DUSE_ROCM runtime/NEON-3.cpp -lrocblas -lamdhip64 -o neon
```

Vulkan (experimental; see [Known limitations](#known-limitations)). The
`.comp` shader sources, the compiled-`.spv` build recipe and the
standalone Vulkan test programs live on the **`VULKAN-EXPERIMENTAL`**
branch; `main` only ships `rawllm_vulkan.hpp`. On `main`, a `-DUSE_VULKAN`
build compiles and runs, but with no `.spv` files to find it prints
`[Hardware] Vulkan: unavailable (...)` and falls back to the CPU:
```
g++ -std=c++20 -O2 -pthread -DUSE_VULKAN runtime/NEON-3.cpp -lvulkan -o neon
```
For the full Vulkan workflow: `git checkout VULKAN-EXPERIMENTAL`.

Vision and motion headers on their own (header-only; include them from your own
program — `NEON-3.cpp` already does this for you):
```
g++ -std=c++20 -O2 -pthread -mavx2 -mfma -mf16c -I. my_program.cpp -o my_program
# my_program.cpp:  #include "runtime/vision/rawllm_vit.hpp"
#                  #include "runtime/motion/rawllm_mvae.hpp"
```

### CPU SIMD self-test

Build any program that includes `runtime/simd/rawllm_simd_dispatch.hpp`
with `-DRAWLLM_SIMD_SELFTEST` to get `simd::run_selftest()`. It checks
whichever backend was selected against the scalar reference over
randomized Q4_0/Q8_0 blocks and plain dot-product inputs. It is a
development aid and is not compiled in by default.

## Testing

`tests/test_nctr_loader.cpp` is a standalone smoke test for the `.nctr`
loader. It takes a `.nctr` file to load and validate:
```
g++ -std=c++20 -O0 -g -pthread tests/test_nctr_loader.cpp -o test_nctr_loader
./test_nctr_loader path/to/model.nctr
```

## Quick start

```
# 1. Check compatibility before converting anything
python3 conversion/nanity_convert.py --detect-only your-model.gguf

# 2. Convert (if the report is clean, or you've reviewed what --drop-*-anyway loses)
python3 conversion/nanity_convert.py your-model.gguf your-model-nanity.gguf --quant Q4_0

# 3. Sanity check: loads and passes spec validation, no generation
./neon --model your-model-nanity.gguf --probe

# 4. Generate
./neon --model your-model-nanity.gguf --prompt "Explain photosynthesis in one sentence."
```

`--interactive` runs a persistent standard-input/output loop: the model
stays loaded between prompts and accepts JSON `{"messages": [...]}` for
chat-template formatting and per-request sampling parameters.
`NANITY_DEBUG_LOGITS=1` dumps the top-10 first-token candidates before
sampling, which helps tell a bad conversion apart from a runtime bug.

## Training your own NANITY model

[`training/train_nanity_fixed.py`](./training/train_nanity_fixed.py)
trains the reference PyTorch model in
[`training/modeling_nanity.py`](./training/modeling_nanity.py) and exports
it with `export_gguf()` / `export_nctr()`. Three size presets are built in
(`4b`, `1_5b`, `tiny`) and three phases (`warmup`, `distill`, `finetune`);
`--lora-rank` enables LoRA (Low-Rank Adaptation) fine-tuning.
[`training/prepare_data.py`](./training/prepare_data.py) builds the
pre-tokenized `.bin` datasets the trainer loads. The scripts import each
other by file name, so keep the four files together in `training/`. Anyone
can reuse the same data pipeline and architecture to train their own model against
the same spec.

## Glossary

| Term | Meaning |
|---|---|
| **NANITY** | The single fixed transformer architecture this project defines |
| **NEON** | The C++ runtime that runs NANITY models |
| **GGUF** | GPT-Generated Unified Format, a common single-file model container |
| **`.nctr`** | NANITY's own model container format, hard-coded to the NANITY spec |
| **KV cache** | Key-value cache: stored attention keys/values for already-processed tokens |
| **GQA** | Grouped-query attention: several query heads share one key/value head |
| **RoPE** | Rotary position embedding |
| **SwiGLU** | The gated feed-forward activation used in NANITY's feed-forward blocks |
| **SIMD** | Single instruction, multiple data: CPU vector instructions (AVX2, AVX-512) |
| **VNNI** | Vector Neural Network Instructions: fast 8-bit integer dot products in AVX-512 |
| **ROCm / HIP** | AMD's GPU compute platform and its CUDA-like programming interface |
| **ViT** | Vision Transformer |
| **VAE** | Variational AutoEncoder |
| **MLP** | Multilayer perceptron: a plain stack of fully connected layers |
| **MoE** | Mixture of Experts |
| **LoRA** | Low-Rank Adaptation: fine-tuning by training small add-on matrices |
| **zRAM tier** | This project's term for the idle loop's compression stage for evicted context |
| **Idle loop** | Optional mode where the model keeps generating in the background between user inputs |
