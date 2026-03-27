"""
MoE Expert Sniper v3 — Use HuggingFace transformers for correct attention.

v1: proved concept (0.531 tok/s, garbled output)
v2: added VRAM caching (0.609 tok/s, still garbled — attention wrong)
v3: use transformers' native Qwen3.5 attention (GatedDeltaNet + GQA),
    monkey-patch MoE forward to snipe experts from NVMe.

Strategy:
  1. Load model architecture from transformers (correct attention for free)
  2. Dequantize our MLX 4-bit pinned weights → inject into model
  3. Monkey-patch MoE layers to load 8/256 experts from disk
  4. Cache first N layers' experts in VRAM for speed

Usage:
    python3 sniper_122b_v3.py --model-dir /workspace/qwen35-122b-stream \\
        --original-dir /workspace/qwen35-122b-a10b-4bit
"""

import os
import sys
import gc
import json
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

BITS = 4
GROUP_SIZE = 64


def dequantize_4bit(weight, scales, biases, group_size=64):
    """Dequantize MLX 4-bit to bfloat16."""
    if weight.dtype not in (torch.uint32, torch.int32):
        return weight
    out_features = weight.shape[0]
    w = weight.to(torch.int32)
    unpacked = []
    for i in range(8):
        unpacked.append((w >> (4 * i)) & 0xF)
    unpacked = torch.stack(unpacked, dim=-1)
    in_features = unpacked.shape[1] * 8
    unpacked = unpacked.reshape(out_features, in_features).float()
    num_groups = in_features // group_size
    unpacked = unpacked.reshape(out_features, num_groups, group_size)
    scales_exp = scales.float().unsqueeze(-1)
    biases_exp = biases.float().unsqueeze(-1)
    dequantized = unpacked * scales_exp + biases_exp
    return dequantized.reshape(out_features, in_features).to(torch.bfloat16)


def load_and_inject_pinned_weights(model, pinned_path, device="cuda"):
    """
    Load pinned weights (MLX 4-bit), dequantize, inject into transformers model.
    Only loads non-expert weights (attention, router, shared expert, embeddings, norms).
    """
    print("Loading and dequantizing pinned weights...")
    t0 = time.time()

    loaded = 0
    skipped = 0

    with safe_open(str(pinned_path), framework="pt", device="cpu") as f:
        pinned_keys = list(f.keys())

    # Build a map of model param names
    model_params = dict(model.named_parameters())
    model_buffers = dict(model.named_buffers())
    all_model_keys = set(model_params.keys()) | set(model_buffers.keys())

    with safe_open(str(pinned_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())

        # Group by base name (strip .weight/.scales/.biases)
        base_keys = {}
        for k in keys:
            if k.endswith(".scales") or k.endswith(".biases"):
                base = k.rsplit(".", 1)[0]
                if base not in base_keys:
                    base_keys[base] = {}
                base_keys[base][k.split(".")[-1]] = k
            elif k.endswith(".weight"):
                base = k.rsplit(".", 1)[0]
                if base not in base_keys:
                    base_keys[base] = {}
                base_keys[base]["weight"] = k
            else:
                # Non-quantized param (e.g., norm weights, A_log, dt_bias)
                base_keys[k] = {"raw": k}

        for base, components in base_keys.items():
            if "raw" in components:
                # Non-quantized: load directly
                raw_key = components["raw"]
                tensor = f.get_tensor(raw_key)

                if raw_key in model_params:
                    model_params[raw_key].data = tensor.to(device)
                    loaded += 1
                elif raw_key in model_buffers:
                    # Set buffer
                    parts = raw_key.rsplit(".", 1)
                    parent = model
                    for p in parts[0].split("."):
                        parent = getattr(parent, p)
                    setattr(parent, parts[1], tensor.to(device))
                    loaded += 1
                else:
                    skipped += 1

            elif "weight" in components and "scales" in components:
                # Quantized: dequantize then inject
                w = f.get_tensor(components["weight"])
                s = f.get_tensor(components["scales"])
                b = f.get_tensor(components["biases"])

                dequantized = dequantize_4bit(w, s, b, GROUP_SIZE)

                weight_key = base + ".weight"
                if weight_key in model_params:
                    param = model_params[weight_key]
                    if dequantized.shape == param.shape:
                        param.data = dequantized.to(device)
                        loaded += 1
                    else:
                        print(f"  Shape mismatch: {weight_key} model={param.shape} file={dequantized.shape}")
                        skipped += 1
                else:
                    skipped += 1

                del dequantized
            elif "weight" in components:
                # Non-quantized weight
                tensor = f.get_tensor(components["weight"])
                weight_key = base + ".weight"
                if weight_key in model_params:
                    model_params[weight_key].data = tensor.to(device)
                    loaded += 1
                else:
                    skipped += 1

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded: {loaded}, Skipped: {skipped}, VRAM: {vram:.2f} GB [{time.time()-t0:.1f}s]")
    return loaded


class ExpertSniper:
    """Loads active experts from per-layer safetensors files on NVMe."""

    def __init__(self, expert_dir, num_layers, device="cuda", cache_layers=15):
        self.expert_dir = Path(expert_dir)
        self.device = device
        self.handles = {}
        self.vram_cache = {}
        self.num_layers = num_layers
        self.cache_layers = cache_layers

    def cache_in_vram(self):
        """Pre-load first N layers' experts into VRAM."""
        print(f"Caching expert layers 0-{self.cache_layers-1} in VRAM...")
        t0 = time.time()
        for i in range(min(self.cache_layers, self.num_layers)):
            path = self.expert_dir / f"layer_{i:02d}.safetensors"
            if not path.exists():
                continue
            data = {}
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for k in f.keys():
                    data[k] = f.get_tensor(k).to(self.device)
            self.vram_cache[i] = data
        cached_gb = sum(sum(t.nbytes for t in d.values()) for d in self.vram_cache.values()) / 1e9
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"  Cached: {cached_gb:.2f} GB, Total VRAM: {vram:.2f} GB [{time.time()-t0:.1f}s]")

    def _get_handle(self, layer_idx):
        if layer_idx not in self.handles:
            path = self.expert_dir / f"layer_{layer_idx:02d}.safetensors"
            self.handles[layer_idx] = safe_open(str(path), framework="pt", device="cpu")
        return self.handles[layer_idx]

    def get_expert_weights(self, layer_idx, expert_ids):
        """
        Get dequantized weights for active experts.
        Returns dict of {proj_name: [top_k, out, in]} tensors on GPU.
        """
        ids = expert_ids if isinstance(expert_ids, list) else expert_ids.tolist()

        result = {}
        if layer_idx in self.vram_cache:
            # Fast path: index from VRAM
            data = self.vram_cache[layer_idx]
            idx = torch.tensor(ids, dtype=torch.long, device=self.device)
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                w = torch.index_select(data[f"{proj}.weight"], 0, idx)
                s = torch.index_select(data[f"{proj}.scales"], 0, idx)
                b = torch.index_select(data[f"{proj}.biases"], 0, idx)
                result[proj] = dequantize_4bit(w, s, b, GROUP_SIZE)
        else:
            # Cold path: load from NVMe
            handle = self._get_handle(layer_idx)
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                full_w = handle.get_tensor(f"{proj}.weight")
                full_s = handle.get_tensor(f"{proj}.scales")
                full_b = handle.get_tensor(f"{proj}.biases")
                w = torch.stack([full_w[i] for i in ids]).to(self.device)
                s = torch.stack([full_s[i] for i in ids]).to(self.device)
                b = torch.stack([full_b[i] for i in ids]).to(self.device)
                result[proj] = dequantize_4bit(w, s, b, GROUP_SIZE)

        return result


def make_sniped_moe_forward(original_forward, layer_idx, sniper, top_k=8):
    """
    Create a patched MoE forward that loads experts from disk/cache
    instead of expecting them in VRAM.
    """
    def sniped_forward(hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)

        # Route using the original router (already in VRAM)
        # Access the parent module's gate
        moe_block = sniped_forward._moe_block
        router_logits = moe_block.gate(x)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(routing_weights, top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden_states.dtype)

        # Get unique experts needed
        needed = topk_indices.unique().tolist()

        # Load expert weights from sniper
        expert_weights = sniper.get_expert_weights(layer_idx, needed)

        # Build local index map
        id_to_local = {eid: i for i, eid in enumerate(needed)}

        # Compute expert outputs
        output = torch.zeros_like(x)

        for local_idx, expert_id in enumerate(needed):
            # Which tokens use this expert?
            mask = (topk_indices == expert_id)  # [tokens, top_k]
            token_mask = mask.any(dim=-1)
            token_indices = token_mask.nonzero(as_tuple=True)[0]

            if len(token_indices) == 0:
                continue

            # Get routing weight for this expert
            weights = (topk_weights * mask.float()).sum(dim=-1)  # [tokens]

            # Expert forward: SwiGLU
            token_input = x[token_indices]
            gate_w = expert_weights["gate_proj"][local_idx]
            up_w = expert_weights["up_proj"][local_idx]
            down_w = expert_weights["down_proj"][local_idx]

            gate_out = F.silu(token_input @ gate_w.t())
            up_out = token_input @ up_w.t()
            expert_out = (gate_out * up_out) @ down_w.t()

            # Weighted contribution
            output[token_indices] += weights[token_indices].unsqueeze(-1) * expert_out

        # Shared expert (already in model VRAM)
        if hasattr(moe_block, 'shared_expert') and moe_block.shared_expert is not None:
            shared_out = moe_block.shared_expert(x)
            if hasattr(moe_block, 'shared_expert_gate') and moe_block.shared_expert_gate is not None:
                shared_gate = torch.sigmoid(moe_block.shared_expert_gate(x))
                shared_out = shared_out * shared_gate
            output = output + shared_out

        del expert_weights
        return output.view(batch_size, seq_len, hidden_dim)

    return sniped_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/qwen35-122b-stream")
    parser.add_argument("--original-dir", default="/workspace/qwen35-122b-a10b-4bit")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--cache-layers", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("  MoE EXPERT SNIPER v3 — HF Transformers + Expert Sniping")
    print("  Correct attention (GatedDeltaNet + GQA) for free")
    print("=" * 60)

    model_dir = Path(args.model_dir)
    original_dir = Path(args.original_dir)

    # Step 1: Load model architecture with empty weights
    print("\n[1/5] Loading model architecture...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    config = AutoConfig.from_pretrained(str(original_dir), trust_remote_code=True)

    # Use text_config for the causal LM if it's a VLM config
    if hasattr(config, 'text_config'):
        text_model_config = config.text_config
    else:
        text_model_config = config

    # Load just the text/language model on meta device
    from transformers import AutoModelForCausalLM as AMCLM
    with torch.device("meta"):
        # Try loading from text config directly
        try:
            model = AMCLM.from_config(text_model_config, trust_remote_code=True)
        except Exception as e:
            print(f"  Direct config failed: {e}")
            print("  Trying with full config...")
            model = AMCLM.from_config(config, trust_remote_code=True)

    # Move to device with empty tensors
    model = model.to_empty(device=args.device)
    model.eval()

    text_config = config.text_config if hasattr(config, 'text_config') else config
    num_layers = text_config.num_hidden_layers
    print(f"  Architecture: {type(model).__name__}, {num_layers} layers")

    # Step 2: Inject pinned weights
    print("\n[2/5] Injecting dequantized pinned weights...")
    pinned_path = model_dir / "pinned.safetensors"
    loaded = load_and_inject_pinned_weights(model, pinned_path, device=args.device)

    # Step 3: Set up expert sniper
    print("\n[3/5] Setting up Expert Sniper...")
    sniper = ExpertSniper(
        model_dir / "experts", num_layers,
        device=args.device, cache_layers=args.cache_layers
    )
    sniper.cache_in_vram()

    # Step 4: Monkey-patch MoE layers
    print("\n[4/5] Patching MoE layers for expert sniping...")
    patched = 0
    for i in range(num_layers):
        # Find the MoE block in this layer
        # Handle both VLM (language_model.model.layers) and text-only (model.layers)
        if hasattr(model, 'language_model'):
            layer = model.language_model.model.layers[i]
        elif hasattr(model, 'model'):
            layer = model.model.layers[i]
        else:
            print(f"  Cannot find layers in model")
            break
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            # This is an MoE layer — patch it
            moe_block = layer.mlp
            top_k = text_config.num_experts_per_tok if hasattr(text_config, 'num_experts_per_tok') else 8

            new_forward = make_sniped_moe_forward(
                moe_block.forward, i, sniper, top_k=top_k
            )
            new_forward._moe_block = moe_block

            # Delete expert weights from model to free VRAM
            if hasattr(moe_block, 'experts'):
                del moe_block.experts
                moe_block.experts = None

            moe_block.forward = new_forward
            patched += 1

    print(f"  Patched {patched}/{num_layers} MoE layers")
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM: {vram:.2f} GB")

    # Step 5: Generate!
    print("\n[5/5] Generating...")
    tokenizer = AutoTokenizer.from_pretrained(str(original_dir), trust_remote_code=True)

    messages = [
        {"role": "system", "content": "Answer briefly and directly."},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(args.device)
    print(f"  Prompt: {input_ids.shape[1]} tokens")

    # Generate
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=False,
            temperature=1.0,
        )
    t_total = time.time() - t0

    new_tokens = output_ids[0][input_ids.shape[1]:]
    n = len(new_tokens)
    tps = n / t_total if t_total > 0 else 0
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    vram = torch.cuda.memory_allocated() / 1e9

    print(f"\n{'='*60}")
    print(f"Q: {args.prompt}")
    print(f"A: {output_text}")
    print(f"{'='*60}")
    print(f"  Model: Qwen3.5-122B-A10B (69.6 GB, 4-bit)")
    print(f"  VRAM: {vram:.1f} GB")
    print(f"  Cached layers: 0-{args.cache_layers-1}")
    print(f"  Speed: {tps:.3f} tok/s")
    print(f"  Tokens: {n}")
    print(f"  Time: {t_total:.1f}s")


if __name__ == "__main__":
    main()
