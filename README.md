# PrivateAgent

**Run 35B+ LLMs on your iPhone.** Fully offline, fully private.

PrivateAgent is a native iOS app that runs large language models (35B and beyond) entirely on-device. No cloud, no API keys, no data leaves your phone.

## How is this possible?

Most of a 35B model's parameters sit in **Mixture-of-Experts (MoE)** layers. For each token, only a small subset of experts (~3B out of 35B) are actually needed. PrivateAgent exploits this sparsity:

- **Flash-MoE** streams experts from NVMe storage on demand -- only the active 3B parameters live in RAM at any moment, while the full 35B model sits on disk.
- **TurboQuant** compresses the KV cache so you get longer conversations within iPhone's memory constraints.
- **Metal GPU acceleration** handles matrix ops, attention, and normalization at ~3 tok/s on iPhone 16 Pro.

This means a 35B-parameter model runs comfortably in ~1.4 GB of RAM. The same architecture scales to **400B+ MoE models** (like Qwen3-235B-A22B or DeepSeek-V3) as long as the active parameters fit in memory.

## Supported Models

| Model | Total Params | Active Params | RAM Usage | Speed |
|-------|-------------|---------------|-----------|-------|
| Qwen3.5-35B-A3B | 35B | 3B | ~1.4 GB | ~3 tok/s |
| More coming... | | | | |

Models are downloaded within the app and stored locally. No HuggingFace account required.

## Architecture

```
PrivateAgent/
├── Sources/
│   ├── FlashMoECore/          # C -- routing math, GatedDeltaNet, tokenizer
│   ├── FlashMoEMetal/         # Metal -- GPU kernels (dequant, matvec, attention, norms)
│   ├── TurboQuantCore/        # C -- KV cache compression format & transforms
│   ├── TurboQuantMetal/       # Metal -- compressed-domain attention (no full decompress)
│   ├── FlashMoERuntime/       # C/ObjC -- session state machine, memory planner, expert pager
│   ├── FlashMoEBridge/        # Swift -- engine facade (@Observable, async streams)
│   ├── ModelPack/             # Swift -- manifest parsing, prompt compiler
│   ├── ModelHub/              # Swift -- model catalog, download manager
│   └── PrivateAgentUI/        # SwiftUI -- chat interface, model manager
├── Vendor/
│   └── flash-moe/             # Vendored C/ObjC inference engine (~10K LOC)
├── Apps/
│   └── PrivateAgentiOS/       # iOS app target
└── Package.swift              # Swift 6.0, iOS 18+
```

### Key Design Decisions

- **FlashMoE-first**: The inference engine owns the hot path. SSD expert streaming via `pread` with page-aligned I/O and GCD fanout.
- **TurboQuant-native**: KV cache compression participates in the attention fast path -- Q scores against compressed K directly, no full decompression.
- **Metal GPU pipeline**: Dequantization, matrix-vector multiply, SwiGLU, RMSNorm, RoPE, and attention all run on GPU. Delta-net (linear attention) layers use Accelerate BLAS on CPU.
- **Smart context management**: Auto-detects follow-up questions vs independent queries. Follow-ups reuse KV cache; new topics reset cleanly.

## Build

**Requirements**: Xcode 16+, iOS 18+, Swift 6.0

```bash
# Clone
git clone https://github.com/nicklama/PrivateAgent.git
cd PrivateAgent

# Open in Xcode
open Apps/PrivateAgentiOS/PrivateAgent.xcodeproj
# Or build with SPM
swift build
```

Deploy to your iPhone, download a model from the in-app model manager, and start chatting.

## Roadmap

- [ ] More model support (Qwen3-235B-A22B, DeepSeek-V3)
- [ ] Tiered quantization (mixed 2/3/4-bit for memory-speed tradeoff)
- [ ] NAX Metal 4 acceleration (M5+ / A19+)
- [ ] Conversation export / import
- [ ] System prompt templates
- [ ] Context window extension via TurboQuant 3.5-bit KV

## How it scales to 400B+

The secret is **sparsity**. A 400B MoE model like DeepSeek-V3 only activates ~37B parameters per token. With Flash-MoE's SSD streaming:

- Active parameters (~37B at 4-bit) = ~18 GB compute memory
- On a device with sufficient RAM (e.g., iPad Pro M4 with 16GB), this is feasible
- The full 400B model sits on storage (~200 GB), streamed on demand

The architecture is storage-bound, not memory-bound. As phone storage gets faster (NVMe on A-series chips already hits 2-3 GB/s sequential read), larger models become practical.

## License

Apache 2.0 -- see [LICENSE](LICENSE)
