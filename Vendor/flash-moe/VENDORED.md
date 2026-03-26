# Vendored Flash-MoE Source

Source: https://github.com/Anemll/flash-moe (iOS-App branch)
Date vendored: 2026-03-26
Commit: 9d1d602bceca11b2e1b9fd6cf89057858cfbe6e4

Files:
- infer.m (10,488 lines) — Complete inference engine
- shaders.metal (2,377 lines) — Metal compute kernels
- FlashMoEEngine.h/m — iOS wrapper API
- tokenizer.h — C BPE tokenizer
- batched_prefill.h — Batched prefill support
- gguf_iq_shared.h — GGUF IQ quantization defs

These files are included via unity build in FlashMoERuntime.
Modifications are tracked in this repo's git history.
