# 🍎 mac code

**Claude Code, but it runs on your Mac for free.**

No cloud. No API keys. No monthly bill. A 35-billion parameter AI agent running on a $600 Mac mini.

---

## Why this exists

Every AI coding agent today — Claude Code, Cursor, Copilot — sends your code to someone else's server and charges you per token. We wanted to know: **can a Mac mini on your desk do the same job?**

The answer is yes. And the reason is Apple Silicon.

## What makes this different

**The model doesn't fit in RAM.** That's the whole point.

Qwen3.5-35B-A3B is a 10.6 GB model. A Mac mini M4 has 16 GB of RAM. After macOS takes its share, there's not enough room. The overflow pages from the SSD.

On any other hardware, this kills performance. We tested it:

| Setup | Speed | Cost/hr | What happens |
|---|---|---|---|
| **Mac mini M4 + SSD paging** | **29.8 tok/s** | **$0.00** | **GPU processes everything via unified memory** |
| NVIDIA GPU + NVMe paging | 1.6 tok/s | $0.44 | CPU bottleneck — GPU can't access paged data |
| NVIDIA GPU + FUSE paging | 0.075 tok/s | $0.44 | Network storage — barely functional |
| NVIDIA GPU in-VRAM (no paging) | 42.5 tok/s | $0.34 | Fast, but costs money and needs big GPU |
| Claude Code (API) | ~80 tok/s | ~$0.50+ | Fastest, but every token costs money |

**Apple Silicon is 18.6x faster than NVIDIA when the model doesn't fit in memory.**

Why? Because of **unified memory**. On a Mac, the CPU, GPU, and SSD all share the same memory address space. When macOS pages model weights from the SSD, the Metal GPU can still process them directly — no copy to a separate GPU memory, no CPU bottleneck. The data flows SSD → unified memory → GPU at 3-5 GB/s.

On NVIDIA, paging forces the data through the CPU first, then across the PCIe bus to the GPU. The CPU becomes the bottleneck and the $10,000 GPU sits idle.

**This is Apple's "LLM in a Flash" thesis running in practice on a $600 computer.**

## Why MoE is the key

The model is **Qwen3.5-35B-A3B** — a Mixture-of-Experts architecture:
- 35 billion total parameters
- 256 experts
- Only **8 experts (3B parameters) activate per token**

This means at any moment, 90% of the model is "cold." Cold experts sit on the SSD. Hot experts stay cached in RAM. The GPU only needs the active 3B to generate each token, so the SSD paging overhead is minimal.

Dense models can't do this. A 35B dense model would need every parameter for every token. MoE + Apple Silicon SSD paging is the combination that makes local AI practical on consumer hardware.

## The result

A fully autonomous AI agent with web search, file operations, code execution, and 19 slash commands — running at **30 tokens per second** on a Mac mini that costs $599 once and $4/month in electricity.

---

## Quick Start

### What you need

- Mac with Apple Silicon (M1 or later, 16GB+ RAM)
- [Homebrew](https://brew.sh)

### One-command setup

```bash
git clone https://github.com/walter-grace/pico-mini.git
cd pico-mini
chmod +x setup.sh && ./setup.sh
```

### Or step by step

**1 — Install llama.cpp and download the model**

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

> 10.6 GB download. Takes 5-15 minutes.

**2 — Start the inference server**

```bash
llama-server \
    --model ~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 8192 \
    --n-gpu-layers 99 --reasoning off -np 1 -t 4
```

> Wait for "server is listening". ~20 seconds.

**3 — Build the agent backend**

```bash
git clone https://github.com/sipeed/picoclaw.git
cd picoclaw && make deps && make build && cd ..
```

> Needs Go 1.25+. Install with `brew install go` if needed.

**4 — Configure**

```bash
mkdir -p ~/.picoclaw/workspace
cp config.example.json ~/.picoclaw/config.json
```

**5 — Run**

```bash
python3 agent.py
```

That's it. You have a local AI coding agent on your Mac.

---

## What it looks like

```
  🍎 mac code
  claude code, but it runs on your Mac for free

  model  Qwen3.5-35B-A3B  MoE 34.7B · 3B active · IQ2_M
  tools  search · fetch · exec · files
  cost   $0.00/hr  Apple M4 Metal · localhost:8000

  ──────────────────────────────────────────────────

  agent > search the web for the latest Qwen 3.5 news

  ⠹ searching the web  8s
    llm_request model=qwen3.5-35b-a3b
    tool_call web_search "Qwen 3.5 latest news"
    tool_result received

  thinking → searching the web → thinking → finishing up

  Alibaba released Qwen3.5, a new series of AI models with
  advanced agentic capabilities...

  29.7 tok/s  ·  142 tokens  ·  4.8s
```

## Commands

| Command | Action |
|---|---|
| `/agent` | Agent mode — web search, file ops, shell exec (default) |
| `/raw` | Raw mode — direct streaming to LLM, no tools |
| `/tools` | List available tools |
| `/stats` | Session statistics |
| `/clear` | Reset conversation |
| `/system <msg>` | Set system prompt |
| `/quit` | Exit |

## Tools

All local. No API keys.

| Tool | What it does |
|---|---|
| `web_search` | DuckDuckGo search |
| `web_fetch` | Read any URL |
| `exec` | Run shell commands |
| `read_file` | Read local files |
| `write_file` | Create files |
| `edit_file` | Modify files |
| `list_dir` | Browse directories |
| `subagent` | Spawn sub-tasks |

---

## Files

| File | What |
|---|---|
| `agent.py` | The main agent — animated loading, live logs, markdown, tools |
| `chat.py` | Lightweight streaming chat (no tools) |
| `dashboard.py` | Real-time server monitor with tok/s sparklines |
| `config.example.json` | Agent config with DuckDuckGo + fetch MCP servers |
| `setup.sh` | One-command install |

---

## How it works

### "LLM in a Flash" on Apple Silicon

The model is 10.6 GB. Your Mac has 16 GB RAM. After the OS takes ~4 GB, there's not enough room. macOS pages the overflow from the SSD.

On NVIDIA, this kills performance (1.6 tok/s) because paging forces computation onto the CPU. On Apple Silicon, the GPU processes all layers via **unified memory** regardless of whether data is in RAM or paging from SSD. Result: **29.8 tok/s while paging 5.4 GB from SSD**.

### Why MoE matters

Qwen3.5-35B-A3B has 256 experts but only activates 8 per token (3B of 35B parameters). This means:
- Only a small fraction of the model is "hot" at any time
- Hot experts stay cached in RAM
- Cold experts page from SSD on demand
- Effective compute is 3B per token, but you get 35B-class intelligence

### Architecture

```
┌──────────────────────────────────────────────┐
│  mac code TUI           Python + Rich        │
├──────────────────────────────────────────────┤
│  PicoClaw               Go agent framework   │
│  github.com/sipeed/picoclaw                  │
├──────────────────────────────────────────────┤
│  llama.cpp              C++ + Metal GPU      │
│  github.com/ggergov/llama.cpp                │
├──────────────────────────────────────────────┤
│  Qwen3.5-35B-A3B        MoE model           │
│  34.7B params, 3B active, IQ2_M quant       │
├──────────────────────────────────────────────┤
│  Apple Silicon           Unified Memory      │
│  Metal GPU + SSD flash paging                │
└──────────────────────────────────────────────┘
```

---

## Scaling

| Mac | RAM | Model | Speed |
|---|---|---|---|
| Mac mini M4 | 16 GB | 35B-A3B IQ2_M (this project) | ~30 tok/s |
| Mac mini M4 Pro | 48 GB | 35B-A3B Q4_K_M | ~40+ tok/s |
| Mac Studio M4 Ultra | 192 GB | 397B-A17B (frontier) | ~15-30 tok/s |

### Smaller hardware (8GB)

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

mac code auto-detects whichever model is running.

---

## Benchmarks

212 math problems verified with SymPy:

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

## License

MIT

## Credits

- **[Qwen3.5](https://huggingface.co/Qwen)** — the model (Alibaba)
- **[llama.cpp](https://github.com/ggergov/llama.cpp)** — inference engine (Georgi Gerganov)
- **[PicoClaw](https://github.com/sipeed/picoclaw)** — agent framework (Sipeed)
- **[Unsloth](https://huggingface.co/unsloth)** — GGUF quantizations
- **[Rich](https://github.com/Textualize/rich)** — terminal UI (Will McGugan)
