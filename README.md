# pico-mini

**Run a 35-billion parameter AI agent on a $600 Mac mini — for free.**

No cloud. No API keys. No subscription. Just your Mac.

---

## The Discovery

We proved that Apple Silicon's unified memory architecture lets you run models that **don't fit in RAM** by paging from SSD — and it's fast. This is the "LLM in a Flash" thesis tested in practice.

| Setup | Speed | Cost/hr | How |
|---|---|---|---|
| **pico-mini (Mac mini M4, 16GB)** | **29.8 tok/s** | **$0.00** | **SSD paging** |
| NVIDIA GPU in-VRAM (RunPod) | 42.5 tok/s | $0.34 | Fits in VRAM |
| NVIDIA GPU + NVMe (Vast.ai) | 1.6 tok/s | $0.44 | NVMe paging |
| NVIDIA GPU + FUSE (RunPod) | 0.075 tok/s | $0.44 | Network paging |

**Apple Silicon SSD paging is 18.6x faster than NVIDIA NVMe paging.**

Why? On NVIDIA, paging forces computation onto the CPU — bottleneck. On Apple Silicon, the GPU processes all layers via unified memory regardless of whether data is in RAM or paging from SSD.

### The Model

**Qwen3.5-35B-A3B** — a Mixture-of-Experts model with 35 billion total parameters but only **3 billion active per token**. This is key: MoE means only a fraction of the model is "hot" at any time, making SSD paging practical.

- 10.6 GB on disk (IQ2_M quantization)
- 16 GB RAM available, ~11 GB used by model, ~5 GB paging from SSD
- 256 experts, 8 selected per token
- Outperforms dense models 3x its active size on reasoning benchmarks

### Math Accuracy

We ran 212 math problems verified computationally with SymPy:

| Category | Score |
|---|---|
| Linear Algebra | **100%** (22/22) |
| Number Theory | **100%** (22/22) |
| Logic | **100%** (20/20) |
| Differential Equations | 95% |
| Geometry | 91% |
| Algebra | 86% |
| **Overall** | **86.3%** (183/212) |

---

## Quick Start

### What you need

- Mac with Apple Silicon (M1 or later, 16GB+ RAM)
- [Homebrew](https://brew.sh)

### Option A: One-command setup

```bash
git clone https://github.com/walter-grace/pico-mini.git
cd pico-mini
chmod +x setup.sh && ./setup.sh
```

### Option B: Step by step

**Step 1 — Install llama.cpp and download the model**

```bash
brew install llama.cpp
pip3 install huggingface-hub rich --break-system-packages

mkdir -p ~/models
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
    'Qwen3.5-35B-A3B-UD-IQ2_M.gguf', local_dir='$HOME/models/')
"
```

> The model is 10.6 GB. Download takes 5-15 minutes.

**Step 2 — Start the inference server**

```bash
llama-server \
    --model ~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 8192 \
    --n-gpu-layers 99 --reasoning off -np 1 -t 4
```

> Wait for "server is listening on http://127.0.0.1:8000". Takes ~20 seconds.

**Step 3 — Build PicoClaw (agent framework)**

```bash
git clone https://github.com/sipeed/picoclaw.git
cd picoclaw && make deps && make build && cd ..
```

> Requires Go 1.25+. Install with `brew install go` if needed.

**Step 4 — Configure the agent**

```bash
mkdir -p ~/.picoclaw/workspace
cp config.example.json ~/.picoclaw/config.json
```

**Step 5 — Run**

```bash
python3 agent.py
```

That's it. You're chatting with a 35B AI agent running on your desk.

---

## What's Included

| File | What it does |
|---|---|
| `agent.py` | Interactive agent with web search, file ops, code execution, animated loading, markdown rendering |
| `chat.py` | Lightweight streaming chat — direct to the LLM, no tools |
| `dashboard.py` | Real-time server monitor — tok/s, GPU slots, memory, sparkline graphs |
| `config.example.json` | Agent config pointing at local llama-server with DuckDuckGo search + fetch MCP servers |
| `setup.sh` | One-command install script |

### Agent Commands

| Command | Action |
|---|---|
| `/agent` | Agent mode — web search, file ops, shell exec (default) |
| `/raw` | Raw mode — direct streaming to LLM, no tools |
| `/tools` | List available tools |
| `/stats` | Session statistics (tok/s, tokens, turns) |
| `/clear` | Reset conversation |
| `/system <msg>` | Set system prompt |
| `/quit` | Exit |

### Agent Tools

All tools run locally. No API keys required.

- **Web search** — DuckDuckGo via MCP server
- **URL fetch** — Read any webpage
- **Shell exec** — Run commands
- **File read/write/edit** — Full filesystem access
- **Subagent** — Spawn sub-tasks

---

## Architecture

```
┌──────────────────────────────────────────────┐
│  pico-mini TUI          Python + Rich        │
│  Animated loading, markdown, streaming       │
├──────────────────────────────────────────────┤
│  PicoClaw               Go agent framework   │
│  github.com/sipeed/picoclaw                  │
│  Tool use, web search, MCP, sessions         │
├──────────────────────────────────────────────┤
│  llama.cpp              C++ + Metal GPU      │
│  github.com/ggergov/llama.cpp                │
│  OpenAI-compatible API @ localhost:8000      │
├──────────────────────────────────────────────┤
│  Qwen3.5-35B-A3B        MoE model           │
│  huggingface.co/Qwen                         │
│  34.7B params, 3B active, IQ2_M quant       │
├──────────────────────────────────────────────┤
│  Apple Silicon           M1/M2/M3/M4         │
│  Unified memory + Metal GPU + SSD paging     │
└──────────────────────────────────────────────┘
```

pico-mini is a **recipe and proof-of-concept**, not a framework. The unique contribution is:

1. **The benchmark data** proving Apple Silicon flash-paging works 18.6x faster than NVIDIA
2. **The specific model/quant/config combination** that makes 35B parameters work on 16GB RAM
3. **The TUI wrapper** with animated loading, live logs, and markdown rendering
4. **The 212-problem math eval** with SymPy verification

The heavy lifting is done by open-source projects we gratefully build on.

---

## Smaller Hardware?

Only 8GB RAM? Use the dense 9B model instead (fits entirely in memory):

```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-9B-GGUF',
    'Qwen3.5-9B-Q4_K_M.gguf', local_dir='$HOME/models/')
"

llama-server \
    --model ~/models/Qwen3.5-9B-Q4_K_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 4096 \
    --n-gpu-layers 99 --reasoning off -t 4
```

The agent auto-detects whichever model is running.

---

## Bigger Hardware?

| Mac | RAM | What you can run | Expected speed |
|---|---|---|---|
| Mac mini M4 | 16 GB | Qwen3.5-35B-A3B (this project) | ~30 tok/s |
| Mac mini M4 Pro | 48 GB | Qwen3.5-35B-A3B Q4_K_M (better quality) | ~40+ tok/s |
| Mac Studio M4 Ultra | 192 GB | Qwen3.5-397B-A17B (frontier model) | ~15-30 tok/s |

The same recipe scales. More RAM = bigger models = less paging = faster.

---

## How It Works: "LLM in a Flash"

Apple's 2023 paper ["LLM in a Flash"](https://arxiv.org/abs/2312.11514) proposed running LLMs larger than available RAM by intelligently loading model weights from flash storage. Our results validate this on real hardware:

1. **Unified memory** — Apple Silicon shares RAM between CPU and GPU. No PCIe copy bottleneck.
2. **Metal GPU** — All model layers run on the GPU via Metal, even when data pages from SSD.
3. **MoE sparsity** — Only 3B of 35B parameters activate per token. Hot experts stay cached in RAM; cold experts page from SSD on demand.
4. **SSD bandwidth** — The M4's SSD reads at 3-5 GB/s, fast enough to feed the GPU between token generations.

The result: **29.8 tok/s while actively paging 5.4 GB from SSD** (9,704 pageouts observed). This is conversational speed — fast enough for interactive use.

---

## License

MIT

## Credits

This project is built entirely on open-source work:

- **[Qwen3.5](https://huggingface.co/Qwen)** by Alibaba — the model
- **[llama.cpp](https://github.com/ggergov/llama.cpp)** by Georgi Gerganov — inference engine with Metal support
- **[PicoClaw](https://github.com/sipeed/picoclaw)** by Sipeed — agent framework with tool use and MCP
- **[Unsloth](https://huggingface.co/unsloth)** — optimized GGUF quantizations
- **[Rich](https://github.com/Textualize/rich)** by Will McGugan — terminal UI rendering
