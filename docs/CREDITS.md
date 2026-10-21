# Credits and Prior Art

NEON is written from scratch, but it stands on a lot of public work. This
document says exactly what is original, what is borrowed in idea or format,
and what depends on outside software, so nobody has to guess.

## 1. What "from scratch" means here

"From scratch" applies to the **NEON runtime** (`runtime/`) only.

- **Original code:** the GGUF parser, the `.nctr` loader, both tokenizers
  (byte-level BPE and SentencePiece-style), the JSON reader/writer, the
  thread pool, the AVX2 / AVX-512 / scalar kernels, the fused Q4_0 / Q8_0
  kernels, the transformer forward pass, the KV cache, the vision encoder
  and the Motion VAE inference code are written in this repository.
- **No inference library or ML framework underneath.** No llama.cpp, ggml,
  ONNX Runtime, PyTorch or similar is linked, vendored or called by the
  runtime.
- **CPU build dependencies:** a C++20 compiler and the standard system
  libraries (libstdc++, libm, libgcc_s, libc). Nothing else is required.
  Optional libnuma support is used only if it is present.
- **Vulkan backend:** the compute shaders are written by hand, one by one,
  and use the plain Vulkan API. The shader sources are not on `main`; they
  live on the `VULKAN-EXPERIMENTAL` branch. `main` ships only the host-side
  header.

"From scratch" does **not** mean the ideas are new. See sections 3 and 4.

## 2. Outside software that is used, but is not part of NEON

### The optional ROCm backend

The ROCm/HIP backend (`runtime/backends/rawllm_rocm.hpp`) is optional and
is **not** from scratch. It needs the HIP runtime and sends its matrix
products through **rocBLAS**. RCCL (multi-GPU) and Composable Kernel
(flash attention, an unfinished stub) are referenced behind compile-time
switches. Development on this backend has stopped in favor of Vulkan. It
produced byte-identical output to the CPU build on one converted TinyLlama
model on an MI300X, and has had little testing beyond that.

### The training and conversion tooling (not part of NEON)

The `training/` and `conversion/` folders are Python tools that **produce
or convert** models for NEON. They are separate from the runtime. NEON does
not import, call or link any of them, and the runtime builds and runs
without Python installed.

These tools use well-known open-source libraries:

| Tool area | Uses |
|---|---|
| Model definition and training (`training/`) | PyTorch (ROCm build for AMD hardware), Hugging Face `transformers`, `huggingface_hub`, `datasets`, NumPy, tqdm |
| Tokenization during data preparation | `tiktoken`, Hugging Face `tokenizers`, SentencePiece |
| Export and conversion (`training/`, `conversion/`) | the `gguf` Python package, `safetensors`, NumPy |
| Training methods | AdamW optimizer, LoRA fine-tuning, KL-divergence knowledge distillation from a teacher model |

The training code is an independent implementation of the architecture in
`docs/NANITY_ARCHITECTURE_SPEC.md`. The spec, not shared code, is what
makes a trained model load correctly in NEON. Distillation and conversion
may involve third-party models; those models keep their own licenses and
terms, which apply to anyone who uses them this way.

## 3. Formats and standards NEON is compatible with

These are common knowledge in the field or are file formats NEON reads. No
code from these projects is copied into NEON.

- **GGUF** container format and the **ggml quantization block layouts**
  (Q4_0, Q8_0, and the K-quant sizes the loader recognizes), created by the
  ggml / llama.cpp project. NEON reads these files; it is not a ggml
  runtime and refuses models that are not NANITY-architecture.
- **Transformer components:** rotary position embedding (RoPE),
  grouped-query attention (GQA), SwiGLU feed-forward, RMS normalization,
  softmax attention, as used across the Llama-family literature.
- **Tokenization:** byte-level BPE (GPT-2 / tiktoken family) and
  SentencePiece unigram-style segmentation.
- **Hugging Face safetensors** for exporting vision and motion checkpoints.

## 4. Specific ideas and designs that came from others

NEON's code was written independently, but these specific designs are
inspired by or follow existing work, and the source comments already say so.

| In NEON | Inspired by |
|---|---|
| Interleaved 8-row Q4_0 / Q8_0 weight repacking and the matrix-matrix kernels that use it (`rawllm_repack.hpp`) | The "Q4_0_8x8 / repack" approach in **llama.cpp** |
| Attention sink plus shifted window for the live KV cache (`ensure_cache_room()`, `kv_cache_shift()`) | **StreamingLLM** (Xiao et al., 2023), and llama.cpp's context shift |
| Vulkan backend structure: one matrix-vector dispatch per call | The design of **ggml-vulkan** (llama.cpp), per the header comments |
| Reserved-slot KV compression in the zRAM tier (`kv_cache_compact()`, not yet wired in) | **Gist Tokens**, **AutoCompressors**, **ICAE** |
| Motion VAE: mixture-of-experts pose model, plain PoseVAE, controller network | **Character Controllers Using Motion VAEs** (Ling et al., 2020) and Electronic Arts' `character-motion-vaes` reference code |
| Vision encoder towers and connectors | **CLIP**, **SigLIP**, **LLaVA**-style connectors, **Idefics3 / SmolVLM**-style pixel-shuffle connectors |
| Performance comparison | **llama.cpp** (`llama-bench`) was used as the benchmark baseline, nothing more |

## 5. Models used for testing

NEON has been tested on converted third-party models, including
SmolLM2, Llama-3.2-family, TinyLlama-family and Qwen2.5-0.5B (for
benchmarks). These models belong to their creators and keep their own
licenses. They are not distributed with this repository.

## 6. Licensing

NEON is licensed under the GNU Affero General Public License v3.0 (see
[`LICENSE`](../LICENSE)). Third-party projects named here keep their own
licenses. This document is a statement of credit and is not legal advice.

## 7. Corrections

If you think something here is missing, wrong or unfair to the original
authors, open an issue. Credit corrections will be made quickly.
