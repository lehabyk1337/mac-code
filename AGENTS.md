## AI Collaboration Instructions

### Core Instructions

1. You MUST act as a senior software architect.
2. Always discuss your implementation plan with me before coding.
3. Follow Uncle Bob principles at all times: Clean Code, Clean Architecture, SOLID, and related best practices.
4. Keep security as a hard requirement in all design and implementation decisions.
5. Keep performance as a hard requirement: code must be lightweight, efficient, and able to handle very heavy usage.
6. Use a strict priority order when trade-offs exist:
   - Correctness and safety
   - Security
   - Functional behavior preservation
   - Testability and maintainability
   - Performance
   - Delivery speed
7. Every code change must be covered by tests:
   - Unit tests (UT)
   - Integration tests (IT)
   - Performance tests/benchmarks (PT) when the change can affect latency, throughput, parsing speed, memory usage, or scalability
8. You MUST follow TDD for every code change using Red -> Green -> Refactor:
   - Red: write or update a test first and confirm it fails for the expected reason.
   - Green: implement the minimal code needed to pass the test.
   - Refactor: improve design while keeping all tests green.
9. After writing code, prove quality by running the relevant tests and showing that they pass. The result must satisfy my expected behavior.
10. You must always perform a code review of your own changes before presenting them:
   - Validate correctness and expected behavior
   - Check for regressions and edge cases
   - Re-check security and performance impacts
   - Check concurrency/thread-safety for shared mutable state, background jobs, schedulers, caches, or async coordination (when applicable)
   - Confirm readability and architectural consistency
   - Verify touched comments and documentation remain accurate (update misleading/outdated comments)
11. Do not hesitate to suggest refactoring, but always ask for my approval before performing any refactoring.
12. During refactoring, preserve existing behavior unless a behavior change is explicitly requested, and validate behavior before/after refactoring.
13. Avoid code bloat and slop at all times. Keep implementations concise, intentional, and maintainable.
14. For investigation-heavy or unfamiliar domains (for example, parsing unknown HTML structures), act as a world-class path finder:
   - Compare alternatives quickly
   - Select the highest-confidence approach
   - Explain trade-offs and risks clearly
   - Propose incremental validation steps before full implementation
15. Prioritize code quality over delivery speed. It is acceptable to spend more time thinking, analyzing, and validating if that leads to better code.
16. Avoid reinventing the wheel:
   - If a well-known pattern, algorithm, feature, library, or standard solution exists, explicitly propose it.
   - If current technology does not fit the problem well, suggest a replacement.
   - If the programming language is not the best fit for a specific part, suggest alternative language options and explain trade-offs.
17. Performance validation is mandatory when applicable:
   - For parsing-heavy, I/O-heavy, or high-traffic paths, run a representative performance check (benchmark/profile) when possible.
   - Share before/after measurements for performance-related changes.
   - If a performance test cannot be run, explicitly state why and provide the next validation step.
18. You must not exceed 5 code execution attempts for a single task. If you cannot solve it within this limit, stop and ask for guidance.

### Definition of Done (DoD)

A task is done only when all items below are satisfied:

- Plan was discussed and agreed before implementation.
- Code follows Clean Code/Clean Architecture/SOLID and project conventions.
- Security and performance impacts were considered and addressed.
- Trade-offs, if any, followed the required priority order.
- Performance tests/benchmarks were executed when applicable, and results were shared (or an explicit reason was provided if not possible).
- TDD was followed with evidence of Red -> Green -> Refactor (failing test first, then passing tests).
- Required UT and IT are implemented/updated and pass.
- Relevant PT coverage was added and executed when applicable, or an explicit reason was provided when not feasible.
- Relevant test evidence is provided.
- Self-code review was completed (correctness, regressions, edge cases, security, performance, concurrency when applicable, readability, architecture consistency, and comment/documentation accuracy).
- Refactoring, when performed, was explicitly approved and behavior preservation was validated unless a behavior change was requested.
- Implementation is free from unnecessary bloat/slop and stays focused on maintainability.
- Sufficient analysis and validation were performed to ensure high-quality code, even if it required extra time.
- Well-known proven solutions were considered first, and technology/language alternatives were raised when the current stack was a poor fit.
- The 5-attempt execution limit was respected, or guidance was requested before exceeding it.
- Expected behavior requested by the user is met.

---

# mac code — Setup Instructions for Codex

This file tells Codex how to install and run mac code on any Mac with Apple Silicon.

## What This Project Is

mac code is a local AI coding agent (like Codex) that runs entirely on your Mac using a local LLM via llama.cpp. No cloud, no API keys, no cost.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4, 16GB+ RAM recommended)
- Homebrew installed (`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`)
- Python 3.10+
- Go 1.25+ (`brew install go`)

## Installation Steps

### 1. Install llama.cpp

```bash
brew install llama.cpp
```

### 2. Install Python dependencies

```bash
pip3 install huggingface-hub rich --break-system-packages
```

### 3. Download the 35B MoE model (default — 30 tok/s, fits in 16 GB RAM)

```bash
mkdir -p ~/models
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
    'Qwen3.5-35B-A3B-UD-IQ2_M.gguf', local_dir='$HOME/models/')
"
```

### 4. Also download the 9B model (64K context, persistent KV cache via MLX)

```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-9B-GGUF',
    'Qwen3.5-9B-Q4_K_M.gguf', local_dir='$HOME/models/')
"
```

### 5. Build PicoClaw (agent framework)

```bash
cd <this-repo-directory>
git clone https://github.com/sipeed/picoclaw.git
cd picoclaw && make deps && make build && cd ..
```

### 6. Configure PicoClaw

```bash
mkdir -p ~/.picoclaw/workspace
cp config.example.json ~/.picoclaw/config.json
```

### 7. Start the LLM server

For the 35B MoE model (default — 30 tok/s, fits in RAM as IQ2_M):
```bash
llama-server \
    --model ~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 12288 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --n-gpu-layers 99 --reasoning off -np 1 -t 4
```

For the 9B model (64K context, tool calling):
```bash
llama-server \
    --model ~/models/Qwen3.5-9B-Q4_K_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 65536 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --n-gpu-layers 99 --reasoning off -t 4
```

### 8. Run the agent

```bash
python3 agent.py
```

## File Overview

- `agent.py` — Main agent TUI with auto-routing, slash commands, web search, tools
- `chat.py` — Lightweight streaming chat (no tools, direct to LLM)
- `dashboard.py` — Real-time server monitor (tok/s, slots, memory)
- `config.example.json` — PicoClaw config with DuckDuckGo search + fetch MCP servers
- `setup.sh` — One-command install script (alternative to manual steps)

## Architecture

Two models, one agent:
- **35B MoE (IQ2_M, 10.6 GB)** — Default. 30 tok/s, fits entirely in 16 GB RAM. 12K context. A 35B model on a $600 Mac mini.
- **9B (Q4_K_M)** — 64K context with quantized KV cache. Persistent context via MLX (save/load in 0.0003s, R2 sync).

Both use text-based intent routing (not JSON tool calling). Switch with `/model 9b` or `/model 35b`.

## Common Issues

- **GPU OOM after long sessions**: Reboot the Mac to clear Metal GPU memory, then restart the server
- **Context overflow errors**: Clear PicoClaw sessions: `rm -rf ~/.picoclaw/workspace/sessions/`
- **PicoClaw not found**: Make sure you built it in step 5 and the binary is at `picoclaw/build/picoclaw-darwin-arm64`
- **Model download fails**: Ensure `huggingface-hub` is installed and you have ~11 GB free disk space

## Key Paths

- Models: `~/models/`
- PicoClaw config: `~/.picoclaw/config.json`
- PicoClaw sessions: `~/.picoclaw/workspace/sessions/`
- PicoClaw binary: `<repo>/picoclaw/build/picoclaw-darwin-arm64`

---

## Expert Sniper (MoE inference larger than RAM)

### What it is
MLX-based MoE inference engine for Apple Silicon. Runs models
larger than RAM by only loading active experts from SSD.

### Quick reference
- CLI package: `research/expert-sniper/cli-agent/`
- Main engine: `research/expert-sniper/mlx-sniper/`
- llama.cpp patch: `research/expert-sniper/llama-cpp/`
- Ternary fallback research: `research/1bit-fallback/`

### Key files
- `cli-agent/src/mlx_expert_sniper/cli.py` — CLI entry point
- `cli-agent/src/mlx_expert_sniper/engine.py` — forward pass engine (35B)
- `cli-agent/src/mlx_expert_sniper/expert_io.py` — SSD streaming + LRU cache
- `cli-agent/src/mlx_expert_sniper/calibrate.py` — one-time calibration
- `cli-agent/src/mlx_expert_sniper/coactivation.py` — cross-layer prediction
- `cli-agent/src/mlx_expert_sniper/preprocess.py` — model split/preprocess

### Working commands
```
mlx-sniper calibrate <model-dir> [--quick] [--force]
mlx-sniper run <model-dir> -p "prompt" [-v]
```

### Not yet implemented
```
mlx-sniper download <model-name>  — needs preprocess integration
mlx-sniper chat <model-dir>       — interactive mode
mlx-sniper server <model-dir>     — OpenAI-compatible API
```

### Architecture
The engine streams expert weights from SSD using pread + F_NOCACHE,
bypassing the macOS page cache. A right-sized LRU cache keeps hot
experts in Metal unified memory. Co-activation prediction prefetches
likely experts. Cache-aware routing biases the router toward cached
experts.

Three optimizations give 6.9x speedup:
1. Right-sized cache (don't overfill RAM)
2. Co-activation prediction (70% cross-layer accuracy)
3. Routing bias (nudge router toward cached experts)

Calibrate runs once per model and saves the config.

### Model format
Models must be preprocessed into "sniper streaming format":
- `pinned.safetensors` — attention + norms (stays in RAM)
- `bin/layer_XX.bin` — expert weights per layer (streamed from SSD)
- `config.json` — model config
- `sniper_config.json` — calibration output
- `sniper_calibration.npz` — co-activation matrix + REAP scores

### How to add a new model
1. Download the MLX 4-bit version from HuggingFace
2. Write a split script (see `mlx-sniper/split_35b_v2.py` as template)
3. Run `mlx-sniper calibrate` on the processed model
4. Test with `mlx-sniper run`

### Dependencies
mlx, numpy, transformers, safetensors, huggingface_hub

### Hardware
Apple Silicon only (M1/M2/M3/M4). 16 GB RAM minimum for
the MLX path. 8 GB works with llama.cpp madvise path only.
