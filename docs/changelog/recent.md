# Recent release notes

[Project overview](../../README.md) · [Older releases](archive.md)

Full highlights for 7.8.0–7.11.0. The homepage keeps only short summaries.

## 7.11.0

  * **MACaQ:** Added [native MXFP4 W4A4 quantization](../macaq.md) with activation-aware scaling, block-Hadamard rotation, and Hessian-guided refinement. Includes packed checkpoints, CUDA and RDNA4 (`gfx1201`) Triton kernels, native MLX/Metal inference, a W4A16 option, and a portable Torch fallback.
  * **NF12:** Added lossless 12-bit BF16 weight compression, now the default source storage during calibration for eligible weights, with exact restoration before quantization. Packed linear, embedding, and fused inference APIs use optimized NVIDIA CUDA and AMD RDNA4 HIP kernels or a portable Torch fallback. Added native [MLX/Metal inference](../mlx-runtime.md#macaq-and-nf12-metal-kernels) on Apple silicon, including fused decode and small-token kernels.
  * **Tenstorrent:** Added packed GPTQ inference through Torch/XLA/PJRT and native TTNN, including multi-card execution and opt-in `bf8_b` weights. Added block-float GPTQ calibration and checkpoint export, plus device-resident Llama decode with paged KV caching and chunked prefill up to 64K context. `bf4_b`/`bf8_b` checkpoints also run on NVIDIA CUDA and AMD ROCm; bit-exact CPU kernels support `bf8_b`/`bf4_b`/`bf2_b` without Tenstorrent hardware.
  * **Quantization and recovery:** Added resumable, distributed EXL3 quantization with mixed-tier allocation, projection-level recovery, and disk-backed calibration replay. Expanded RCO allocation to AWQ, EXL3, and GGUF under packed-weight budgets. Added rule-scoped [SmoothQuant protocol actions](../protocol-actions.md) with calibration-driven scale folding.
  * **Security:** Added startup checks for known-vulnerable PyTorch runtimes, explicit trust requirements for checkpoint-provided code, checkpoint path and index validation, and bounded server requests and remote media downloads.
  * **Calibration and accuracy:** Fixed AWQ scaling across dense and MoE families, variable-length and packed-sample calibration, and per-sample attention masks and positions. Excluded padding from expert Hessians and restored multimodal and recurrent-layer replay. Corrected FP8 scale semantics, QQQ BF16 inference, and low-precision reductions and checkpoint precision.
  * **Inference:** Added opt-in QKV and gate/up projection fusion driven by model-defined module plans, preserving native backend selection and packed-weight storage. Accelerated quantized embedding lookups, AWQ/Triton small-batch inference, and packed zero-point conversion. Fixed split-K coverage, strided activations, large-index arithmetic, and backend selection for GGUF and ROCm AWQ.
  * **Models:** Added OLMo Hybrid, Granite SWA, Granite MoE SWA, and Dots3 Note quantization support. Restored Transformers 5.22 model layouts, remote-code loading and generation, and dense/MoE weight coverage. Fixed routed-expert coverage in Jamba and ERNIE 4.5 VL MoE, and Gemma 3 checkpoint loading.
  * **Multi-GPU and ROCm:** Added GIL-safe peer copies and bounded disk-backed activation staging. Reduced retained calibration activations and fixed worker cleanup, Hessian placement, and concurrent Triton compilation. Fixed ROCm allocator corruption and calibration reproducibility; HIP builds now stage sources safely, and deterministic BLAS selection is scoped to GPTQ calibration.
  * **Compat:** Fixed checkpoint and in-memory state round trips, GPTQ v1 zero points, AWQ mixed-bit handling, FP8 save/reload, EXL3 serialization, and dynamic exclusions in Transformers/MLX exports. Unsupported formats now fail clearly instead of loading uninitialized weights. Repaired CPU-only placement and optimization, requantization, rotation, GSQ, LoRA/EoRA, and vLLM/SGLang/OpenAI-compatible generation. Deferred Torch compilation failures fall back to eager inference; NumPy is optional for the core.

## 7.10.0

  * **Models:** Added Ricursive Ripple (`ricursive`) quantization support for the 65B preview checkpoint, including load-time Transformers 5 generation/masking patches, per-head Q/K norm and gated-attention handling, and `v_proj` Hessian isolation. Fixed pre-lm-head norm resolution across multimodal model definitions.
  * **Performance:** Accelerated Ricursive Ripple decode by up to 10x per row at batch 16 with fused RMSNorm and native GQA attention, and added chunked MLP prefill to bound activation memory. The CUDA expandable-segments allocator is now installed before Torch initializes, so wide-layer multi-GPU runs actually use it.
  * **Checkpointing:** Added `gptqmodel.reshard()` for per-layer safetensors sharding with one global shard, byte-for-byte tensor copies, preserved metadata, and config/tokenizer copying. Added a one-time warning when quantization shards are not layer-wise.
  * **Multi-GPU:** Honored explicit device lists for quantization placement, routed cross-device moves per direction to avoid asymmetric ROCm SIGSEGVs, and documented `cudaMallocAsync` for wide-layer Hessian allocation.
  * **ROCm:** Made AMD device families explicit and enforced kernel support contracts; gated W4A FP8/NVFP4 and ExLlamaV2/VPTQ off unsupported HIP paths with fast failures; fixed QQQ kernel launches to use the tensor's device context.
  * **Quantization:** Added method-specific integer bit-width validation across GPTQ, AWQ, QQQ, Paro, FP8, bitsandbytes, EXL3, GGUF, and VPTQ; EXL3 now rejects fractional trellis widths; MoE routing-bypass batch sizes are validated.
  * **Correctness:** Fixed Ricursive Ripple cached decode, mask alignment, per-row positions, and labeled-loss handling, plus EXL3 tile-scratch bounds and sign-vector precision.

## 7.9.0

  * **Correctness:** Fixed packing/inference overflow, underflow, FP32 reductions, BF16 bias cancellation, 8-bit BitBLAS zero points, and stale fused Torch metadata after checkpoint loads or tensor changes.
  * **MoE:** Fixed skipped routed experts in MiniMax M2, Granite MoE/Hybrid, Bailing Hybrid, and AfMoE; added warnings for unresolved module paths.
  * **Memory:** Removed retained FP8 weight expansion from the experimental NVFP4 path and dense-weight scratch from ExLlamaV2.
  * **Kernels:** Added measured RTX 4090 (Ada, SM 8.9) and RDNA4 (gfx1200/gfx1201/gfx1250) launch tables to every tuned Triton pipeline — GPTQ GEMV/GEMM/dequant, Paro GEMM, AWQ GEMM/GEMV, GGUF Triton, and bitsandbytes 4-bit — enforced by a hardware-free parity contract; the fused bitsandbytes 4-bit GEMM measured 1.4x-6.3x faster than the vendor HIP kernel on RDNA4 for `M >= 16`, while Ada keeps the vendor kernels. Added opt-in AllSpark W8A16 for compatible per-channel 8-bit GPTQ/AWQ on CUDA and RDNA4 (gfx1200/gfx1201). EXL3 2-bit quantization now fits RDNA4's 64 KiB shared memory.
  * **MLX:** Accelerated GPTQ/RTN quantization, GPTQ/AWQ group-16 decode and short prefill, and GGUF K-quant/ternary packing.
  * **GSQ/RCO:** Added teacher-KL optimization, per-projection bit allocation, and native MLX search for dense Llama/Qwen3, with token-based batching and device/disk teacher caches.
  * **Models:** Added LFM3, LFM3-VL, and LFM3-Omni text-decoder quantization, OLMo/Helium support, and Qwen2 quantization resume.
  * **Compat:** Added native Transformers/Optimum load hooks preserving dynamic settings, fused/MoE structure, rotation, and W4A policies. Replaced method-specific Marlin/Machete/Swordfish enums with `BACKEND.MARLIN`, `BACKEND.MACHETE`, and `BACKEND.SWORDFISH`; legacy string aliases remain supported by the `hf_` APIs.

## 7.8.0

  * **GSQ:** Added staged GSQ training with public method adapters, distributed staged training that reuses captured activations, and per-block bit allocation driven by an RCO search verified bit-identical against the IST-DASLab reference (measured results on Llama 3.2 1B). Added a paper-reproduction harness, best-epoch restore, HIPRTC-compiled fused scalar relaxation on ROCm, and segment-sum scale gradients.
  * **ROCm:** Added RDNA4 (gfx1200/gfx1201/gfx1250) kernel and quantization support with per-architecture kernel tuning. MI350/gfx950 behavior is unchanged.
  * **GGUF:** Added CPU coverage for every ggml tensor type (all K-quant and IQ variants plus explicit MXFP4/NVFP4) and a native ggml SIMD inference path with per-shape autotuning, measured 3–100x faster than the torch paths on AVX2. New `gguf` install extra.
  * **Multi-GPU:** Added automatic GPU peer-access enablement on CUDA and ROCm with measured-link validation, so multi-GPU quantization copies stay on the interconnect instead of the driver's staging path.
  * **Experimental W4A:** Revalidated the NVFP4 packed layout at load time, removed the retained E4M3 weight expansion, and relaxed the Triton build requirement on the Gluon FP8 GEMM.
  * **Performance:** Package import now builds the kernel registry lazily and defers optional backend probes (previously ~9s and 6767 modules). Vectorized the GPTQ/QQQ MSE shrink-grid and AWQ clip-grid evaluation; grid workspace budgets moved from environment variables to a new MSE configuration option.
  * **Models:** Added RecursiveSI Flint and Raon-1 mini support, Gemma 3 multimodal checkpoint compatibility, Qwen2.5-Omni thinker lm_head quantization, and the Transformers 5.22 Qwen-Omni layout.
  * **Compat:** Kept the AVX512-BF16 GPTQ/AWQ CPU kernels trainable for PEFT LoRA; fixed OpenAI-server auth, stop handling, and ChatCompletion schema; surfaced compatibility-shim failures instead of failing silently; fixed the Marlin CPU-offload snapshot refresh; fixed the calibration minimum-length setting not reaching the dataset filter; re-checked for an emptied calibration set after min-length filtering; tolerated MoE early-stop in data-parallel forward; bounded remote image fetches; read accelerator memory telemetry from the active device; fixed config serialization mutating dynamic overrides; fixed GPTAQ loss-log capture and evaluation-metadata coercion for fast eval tasks.
