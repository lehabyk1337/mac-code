"""
MoE Expert Sniper v2 — Qwen3.5-122B-A10B on NVIDIA L40.

v1 proved the concept (0.531 tok/s, 4.4 GB VRAM).
v2 targets 5-10 tok/s via:
  1. Hybrid expert caching: layers 0-14 fully in VRAM, hot experts for 15-47
  2. Proper full attention with GQA + RoPE + KV cache
  3. Simplified GatedDeltaNet with recurrent state

Usage:
    python3 sniper_122b_v2.py --model-dir /workspace/qwen35-122b-stream --prompt "What is the capital of France?"
"""

import os
import sys
import gc
import json
import time
import math
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from safetensors import safe_open

# ── Config ──────────────────────────────────────────────

BITS = 4
GROUP_SIZE = 64


def dequantize_4bit(weight, scales, biases, group_size=64):
    """Dequantize MLX 4-bit to float16. Unsigned 0-15, LSB-first."""
    if weight.dtype != torch.uint32 and weight.dtype != torch.int32:
        return weight.to(torch.float16)

    out_features = weight.shape[0]
    w = weight.to(torch.int32)
    unpacked = []
    for i in range(8):
        unpacked.append((w >> (4 * i)) & 0xF)
    unpacked = torch.stack(unpacked, dim=-1)
    in_features = unpacked.shape[1] * 8
    unpacked = unpacked.reshape(out_features, in_features).to(torch.float16)
    num_groups = in_features // group_size
    unpacked = unpacked.reshape(out_features, num_groups, group_size)
    scales_exp = scales.unsqueeze(-1).to(torch.float16)
    biases_exp = biases.unsqueeze(-1).to(torch.float16)
    dequantized = unpacked * scales_exp + biases_exp
    return dequantized.reshape(out_features, in_features)


def dequant_weight(pinned, prefix, device="cuda"):
    """Dequantize a quantized weight from pinned dict."""
    w_key = f"{prefix}.weight"
    if w_key not in pinned:
        return None
    w = pinned[w_key]
    if w.dtype == torch.uint32:
        s = pinned.get(f"{prefix}.scales")
        b = pinned.get(f"{prefix}.biases")
        if s is not None and b is not None:
            return dequantize_4bit(w.to(device), s.to(device), b.to(device), GROUP_SIZE)
    return w.to(torch.float16).to(device)


def rms_norm(x, weight, eps=1e-6):
    """RMSNorm returning float16."""
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(torch.float16)


def apply_rope_partial(q, k, position, head_dim, rotary_dim, rope_theta):
    """
    Apply RoPE to the first rotary_dim dimensions only.
    q: [B, num_heads, 1, head_dim]
    k: [B, num_kv_heads, 1, head_dim]
    """
    # Compute frequencies
    half_rot = rotary_dim // 2
    freqs = 1.0 / (rope_theta ** (torch.arange(0, half_rot, dtype=torch.float32, device=q.device) / half_rot))
    t = position.float() * freqs  # [half_rot]

    cos_t = torch.cos(t).to(q.dtype)
    sin_t = torch.sin(t).to(q.dtype)

    def rotate(x):
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x1 = x_rot[..., :half_rot]
        x2 = x_rot[..., half_rot:]
        rotated = torch.cat([x1 * cos_t - x2 * sin_t, x1 * sin_t + x2 * cos_t], dim=-1)
        return torch.cat([rotated, x_pass], dim=-1)

    return rotate(q), rotate(k)


class SniperV2:
    def __init__(self, model_dir, device="cuda"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.expert_handles = {}

        with open(self.model_dir / "config.json") as f:
            self.config = json.load(f)

        tc = self.config.get("text_config", self.config)
        self.num_layers = tc.get("num_hidden_layers", 48)
        self.hidden_size = tc.get("hidden_size", 3072)
        self.num_experts = tc.get("num_experts", 256)
        self.top_k = tc.get("num_experts_per_tok", 8)
        self.expert_intermediate = tc.get("moe_intermediate_size", 1024)
        self.num_heads = tc.get("num_attention_heads", 32)
        self.num_kv_heads = tc.get("num_key_value_heads", 2)
        self.head_dim = tc.get("head_dim", 256)
        self.rotary_dim = int(self.head_dim * tc.get("rope_parameters", {}).get("partial_rotary_factor", 0.25))
        self.rope_theta = tc.get("rope_parameters", {}).get("rope_theta", 10000000)
        self.layer_types = tc.get("layer_types", [])

        # Cache config: layers 0-14 fully cached in VRAM
        self.full_cache_layers = 15

        # KV cache for full attention layers
        self.kv_cache = {}  # layer_idx -> (k_cache, v_cache)

        # SSM state for linear attention layers
        self.ssm_state = {}  # layer_idx -> state tensor

        # Expert VRAM cache
        self.expert_vram_cache = {}  # layer_idx -> dict of tensors (full [256,...] on GPU)

        self.position = 0  # current position for RoPE

        print(f"Config: {self.num_layers} layers, {self.num_experts} experts, "
              f"top-{self.top_k}, hidden={self.hidden_size}")
        print(f"Attention: {self.num_heads} Q heads, {self.num_kv_heads} KV heads, "
              f"head_dim={self.head_dim}, rotary_dim={self.rotary_dim}")
        print(f"Full VRAM cache: layers 0-{self.full_cache_layers-1}")

    def load_pinned(self):
        """Load all non-expert weights onto GPU."""
        print("\nLoading pinned weights...")
        t0 = time.time()
        pinned_path = self.model_dir / "pinned.safetensors"
        self.pinned = {}
        with safe_open(str(pinned_path), framework="pt", device="cpu") as f:
            for i, key in enumerate(f.keys()):
                self.pinned[key] = f.get_tensor(key).to(self.device)
                if (i + 1) % 200 == 0:
                    print(f"  {i+1} tensors...")
        gb = sum(t.nbytes for t in self.pinned.values()) / 1e9
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"  Pinned: {gb:.2f} GB, VRAM: {vram:.2f} GB [{time.time()-t0:.1f}s]")

    def cache_experts_in_vram(self):
        """Load first N layers' experts fully into VRAM for instant access."""
        print(f"\nCaching experts for layers 0-{self.full_cache_layers-1} in VRAM...")
        t0 = time.time()
        for layer_idx in range(min(self.full_cache_layers, self.num_layers)):
            path = self.model_dir / "experts" / f"layer_{layer_idx:02d}.safetensors"
            if not path.exists():
                continue
            layer_data = {}
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    layer_data[key] = f.get_tensor(key).to(self.device)
            self.expert_vram_cache[layer_idx] = layer_data
        cached_gb = sum(
            sum(t.nbytes for t in d.values())
            for d in self.expert_vram_cache.values()
        ) / 1e9
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"  Cached: {cached_gb:.2f} GB, Total VRAM: {vram:.2f} GB [{time.time()-t0:.1f}s]")

    def _get_expert_handle(self, layer_idx):
        if layer_idx not in self.expert_handles:
            path = self.model_dir / "experts" / f"layer_{layer_idx:02d}.safetensors"
            self.expert_handles[layer_idx] = safe_open(str(path), framework="pt", device="cpu")
        return self.expert_handles[layer_idx]

    def load_active_experts(self, layer_idx, expert_ids_list):
        """Load active experts — from VRAM cache or NVMe."""
        if layer_idx in self.expert_vram_cache:
            # Instant: index from VRAM cache
            data = self.expert_vram_cache[layer_idx]
            result = {}
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                for comp in ["weight", "scales", "biases"]:
                    key = f"{proj}.{comp}"
                    full = data[key]  # [256, ...]
                    result[key] = full[expert_ids_list]
            return result
        else:
            # Cold: load from NVMe (full tensor, index on CPU, transfer to GPU)
            handle = self._get_expert_handle(layer_idx)
            result = {}
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                for comp in ["weight", "scales", "biases"]:
                    key = f"{proj}.{comp}"
                    full = handle.get_tensor(key)
                    slices = torch.stack([full[i] for i in expert_ids_list])
                    result[key] = slices.to(self.device)
            return result

    def route(self, x, layer_idx):
        """Router: pick top-K experts."""
        prefix = f"language_model.model.layers.{layer_idx}.mlp.gate"
        router_w = dequant_weight(self.pinned, prefix, self.device)
        if router_w is None:
            return None, None

        x_flat = x.squeeze(0).squeeze(0).to(torch.float16)
        logits = x_flat @ router_w.t()
        scores = F.softmax(logits.float(), dim=-1).to(torch.float16)
        top_k_scores, top_k_ids = torch.topk(scores, self.top_k)
        top_k_scores = top_k_scores / top_k_scores.sum()
        return top_k_ids, top_k_scores

    def run_moe_ffn(self, x, layer_idx):
        """Run MoE FFN with expert sniping."""
        prefix = f"language_model.model.layers.{layer_idx}"
        norm_w = self.pinned.get(f"{prefix}.post_attention_layernorm.weight")
        if norm_w is None:
            return x
        normed = rms_norm(x, norm_w)

        expert_ids, expert_weights = self.route(normed, layer_idx)
        if expert_ids is None:
            # Dense layer
            gate_w = dequant_weight(self.pinned, f"{prefix}.mlp.gate_proj", self.device)
            up_w = dequant_weight(self.pinned, f"{prefix}.mlp.up_proj", self.device)
            down_w = dequant_weight(self.pinned, f"{prefix}.mlp.down_proj", self.device)
            if gate_w is not None:
                h = normed.squeeze(0).squeeze(0).to(torch.float16)
                out = F.silu(h @ gate_w.t()) * (h @ up_w.t())
                out = out @ down_w.t()
                del gate_w, up_w, down_w
                return x + out.unsqueeze(0).unsqueeze(0)
            return x

        ids_list = expert_ids.cpu().tolist()
        data = self.load_active_experts(layer_idx, ids_list)

        # Compute each expert
        output = torch.zeros(self.hidden_size, device=self.device, dtype=torch.float16)
        h = normed.squeeze(0).squeeze(0).to(torch.float16)

        for i in range(self.top_k):
            gate_w = dequantize_4bit(data["gate_proj.weight"][i], data["gate_proj.scales"][i], data["gate_proj.biases"][i], GROUP_SIZE)
            up_w = dequantize_4bit(data["up_proj.weight"][i], data["up_proj.scales"][i], data["up_proj.biases"][i], GROUP_SIZE)
            down_w = dequantize_4bit(data["down_proj.weight"][i], data["down_proj.scales"][i], data["down_proj.biases"][i], GROUP_SIZE)

            gate_out = F.silu(h @ gate_w.t())
            up_out = h @ up_w.t()
            expert_out = (gate_out * up_out) @ down_w.t()
            output += expert_weights[i] * expert_out
            del gate_w, up_w, down_w

        # Shared expert
        shared_prefix = f"{prefix}.mlp.shared_expert"
        sg = dequant_weight(self.pinned, f"{shared_prefix}.gate_proj", self.device)
        if sg is not None:
            su = dequant_weight(self.pinned, f"{shared_prefix}.up_proj", self.device)
            sd = dequant_weight(self.pinned, f"{shared_prefix}.down_proj", self.device)
            shared_out = F.silu(h @ sg.t()) * (h @ su.t())
            output += (shared_out @ sd.t())
            del sg, su, sd

        del data
        return x + output.unsqueeze(0).unsqueeze(0)

    def forward_full_attention(self, h, layer_idx):
        """Proper GQA + RoPE + KV cache."""
        prefix = f"language_model.model.layers.{layer_idx}"
        norm_w = self.pinned.get(f"{prefix}.input_layernorm.weight")
        if norm_w is None:
            return h

        normed = rms_norm(h, norm_w)
        x = normed.squeeze(0).squeeze(0).to(torch.float16)  # [hidden]

        attn_prefix = f"{prefix}.self_attn"
        q_w = dequant_weight(self.pinned, f"{attn_prefix}.q_proj", self.device)
        k_w = dequant_weight(self.pinned, f"{attn_prefix}.k_proj", self.device)
        v_w = dequant_weight(self.pinned, f"{attn_prefix}.v_proj", self.device)
        o_w = dequant_weight(self.pinned, f"{attn_prefix}.o_proj", self.device)

        if q_w is None:
            return h

        # Project
        q = (x @ q_w.t()).view(1, self.num_heads, 1, self.head_dim)
        k = (x @ k_w.t()).view(1, self.num_kv_heads, 1, self.head_dim)
        v = (x @ v_w.t()).view(1, self.num_kv_heads, 1, self.head_dim)
        del q_w, k_w, v_w

        # QK-norm
        qn_w = self.pinned.get(f"{attn_prefix}.q_norm.weight")
        kn_w = self.pinned.get(f"{attn_prefix}.k_norm.weight")
        if qn_w is not None:
            q = rms_norm(q, qn_w)
            k = rms_norm(k, kn_w)

        # RoPE (partial)
        pos = torch.tensor([self.position], device=self.device)
        q, k = apply_rope_partial(q, k, pos, self.head_dim, self.rotary_dim, self.rope_theta)

        # KV cache
        if layer_idx in self.kv_cache:
            k_cache, v_cache = self.kv_cache[layer_idx]
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        self.kv_cache[layer_idx] = (k.detach(), v.detach())

        # GQA expand
        num_groups = self.num_heads // self.num_kv_heads
        k_exp = k.unsqueeze(2).expand(-1, -1, num_groups, -1, -1).reshape(1, self.num_heads, -1, self.head_dim)
        v_exp = v.unsqueeze(2).expand(-1, -1, num_groups, -1, -1).reshape(1, self.num_heads, -1, self.head_dim)

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k_exp.transpose(-2, -1)) * scale
        attn = F.softmax(attn.float(), dim=-1).to(torch.float16)
        out = torch.matmul(attn, v_exp)

        # Reshape and project
        out = out.squeeze(2).reshape(1, 1, -1)
        out = out.squeeze(0).squeeze(0) @ o_w.t()
        del o_w
        return h + out.unsqueeze(0).unsqueeze(0)

    def forward_linear_attention(self, h, layer_idx):
        """Simplified GatedDeltaNet / Mamba-2 with recurrent state."""
        prefix = f"language_model.model.layers.{layer_idx}"
        norm_w = self.pinned.get(f"{prefix}.input_layernorm.weight")
        if norm_w is None:
            return h

        normed = rms_norm(h, norm_w)
        x = normed.squeeze(0).squeeze(0).to(torch.float16)

        attn_prefix = f"{prefix}.linear_attn"

        # Combined QKV projection
        qkv_w = dequant_weight(self.pinned, f"{attn_prefix}.in_proj_qkv", self.device)
        if qkv_w is None:
            return h
        qkv = x @ qkv_w.t()
        del qkv_w

        # Z gate
        z_w = dequant_weight(self.pinned, f"{attn_prefix}.in_proj_z", self.device)
        z = torch.sigmoid(x @ z_w.t()) if z_w is not None else torch.ones_like(qkv[:qkv.shape[0]//3])
        del z_w

        # Split QKV
        third = qkv.shape[0] // 3
        q, k, v = qkv[:third], qkv[third:2*third], qkv[2*third:3*third]

        # A and dt for SSM
        A_log = self.pinned.get(f"{attn_prefix}.A_log")
        dt_bias = self.pinned.get(f"{attn_prefix}.dt_bias")

        # B projections (input/output gates)
        a_w = dequant_weight(self.pinned, f"{attn_prefix}.in_proj_a", self.device)
        b_w = dequant_weight(self.pinned, f"{attn_prefix}.in_proj_b", self.device)

        # Simplified SSM: use recurrent state
        if A_log is not None and dt_bias is not None:
            A = -torch.exp(A_log.to(torch.float16).to(self.device))
            dt = dt_bias.to(torch.float16).to(self.device)

            # Get or init state
            state_dim = v.shape[0]
            if layer_idx not in self.ssm_state:
                self.ssm_state[layer_idx] = torch.zeros(state_dim, device=self.device, dtype=torch.float16)

            state = self.ssm_state[layer_idx]

            # SSM update: h_t = A * h_{t-1} + B_t * x_t
            # Simplified: use v as input, q*k as gating
            gate = torch.sigmoid(q[:state_dim] * k[:state_dim]) if q.shape[0] >= state_dim else torch.ones(state_dim, device=self.device, dtype=torch.float16)
            decay = torch.sigmoid(A[:state_dim] + dt[:state_dim]) if A.shape[0] >= state_dim else torch.ones(state_dim, device=self.device, dtype=torch.float16) * 0.9
            state = decay * state + gate * v[:state_dim]
            self.ssm_state[layer_idx] = state.detach()

            out = state * z[:state_dim] if z.shape[0] >= state_dim else state
        else:
            # Fallback: simple linear attention
            out = v * z[:v.shape[0]] if z.shape[0] >= v.shape[0] else v

        # Norm
        norm_w_attn = self.pinned.get(f"{attn_prefix}.norm.weight")
        if norm_w_attn is not None and out.shape[0] == norm_w_attn.shape[0]:
            out = rms_norm(out.unsqueeze(0), norm_w_attn).squeeze(0)

        # Output projection
        o_w = dequant_weight(self.pinned, f"{attn_prefix}.out_proj", self.device)
        if o_w is not None:
            # Match dimensions
            if out.shape[0] < o_w.shape[1]:
                out = F.pad(out, (0, o_w.shape[1] - out.shape[0]))
            elif out.shape[0] > o_w.shape[1]:
                out = out[:o_w.shape[1]]
            out = out @ o_w.t()
        del o_w

        return h + out.unsqueeze(0).unsqueeze(0)

    def forward_token(self, token_id):
        """Full forward pass for one token."""
        # Embed
        embed_key = None
        for k in self.pinned:
            if "embed_tokens" in k and k.endswith(".weight"):
                embed_key = k
                break

        embed_w = self.pinned[embed_key]
        if embed_w.dtype == torch.uint32:
            base = embed_key.replace(".weight", "")
            embed_w = dequantize_4bit(embed_w, self.pinned[f"{base}.scales"], self.pinned[f"{base}.biases"], GROUP_SIZE)

        h = embed_w[token_id].unsqueeze(0).unsqueeze(0).to(torch.float16)

        for i in range(self.num_layers):
            layer_type = self.layer_types[i] if i < len(self.layer_types) else "linear_attention"

            if layer_type == "full_attention":
                h = self.forward_full_attention(h, i)
            else:
                h = self.forward_linear_attention(h, i)

            h = self.run_moe_ffn(h, i)

            torch.cuda.empty_cache()

        # Final norm
        for k in self.pinned:
            if k.endswith("model.norm.weight"):
                h = rms_norm(h, self.pinned[k])
                break

        # LM head
        for k in self.pinned:
            if "lm_head" in k and k.endswith(".weight"):
                lm_w = self.pinned[k]
                if lm_w.dtype == torch.uint32:
                    base = k.replace(".weight", "")
                    lm_w = dequantize_4bit(lm_w, self.pinned[f"{base}.scales"], self.pinned[f"{base}.biases"], GROUP_SIZE)
                return (h.squeeze(0).squeeze(0) @ lm_w.t())

        return h.squeeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.path.expanduser("~/models/qwen35-122b-stream"))
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-layers", type=int, default=15)
    args = parser.parse_args()

    print("=" * 60)
    print("  MoE EXPERT SNIPER v2 — Qwen3.5-122B-A10B")
    print("  Hybrid VRAM cache + NVMe streaming")
    print("=" * 60)

    engine = SniperV2(args.model_dir, device=args.device)
    engine.full_cache_layers = args.cache_layers
    engine.load_pinned()
    engine.cache_experts_in_vram()

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"\nTotal VRAM: {vram:.2f} GB")

    print(f"\nLoading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-122B-A10B", trust_remote_code=True)

    messages = [
        {"role": "system", "content": "Think briefly, answer directly."},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(text)
    print(f"Prompt: {len(tokens)} tokens")

    # Prefill
    print(f"\n--- Prefill ---")
    t0 = time.time()
    for tid in tokens:
        logits = engine.forward_token(tid)
        engine.position += 1
    prefill_time = time.time() - t0
    print(f"Prefill: {prefill_time:.1f}s ({len(tokens)} tokens, {len(tokens)/prefill_time:.2f} tok/s)")
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Decode
    print(f"\n--- Decode (max {args.max_tokens}) ---")
    generated = []
    t_decode = time.time()

    for step in range(args.max_tokens):
        t_step = time.time()
        next_token = torch.argmax(logits).item()
        if next_token in (248044, 248046):
            break

        generated.append(next_token)
        chunk = tokenizer.decode([next_token])
        print(chunk, end="", flush=True)

        logits = engine.forward_token(next_token)
        engine.position += 1

        step_time = time.time() - t_step
        if (step + 1) % 5 == 0:
            vram = torch.cuda.memory_allocated() / 1e9
            tps = (step + 1) / (time.time() - t_decode)
            print(f" [{tps:.2f} tok/s, {vram:.1f}GB]", flush=True)

    t_total = time.time() - t_decode
    n = len(generated)
    tps = n / t_total if t_total > 0 else 0
    output = tokenizer.decode(generated)
    vram = torch.cuda.memory_allocated() / 1e9

    print(f"\n\n{'='*60}")
    print(f"Q: {args.prompt}")
    print(f"A: {output}")
    print(f"{'='*60}")
    print(f"  Model: Qwen3.5-122B-A10B (69.6 GB, 4-bit)")
    print(f"  VRAM: {vram:.1f} GB")
    print(f"  Cached layers: 0-{engine.full_cache_layers-1} (full VRAM)")
    print(f"  Streamed layers: {engine.full_cache_layers}-{engine.num_layers-1} (NVMe)")
    print(f"  Speed: {tps:.3f} tok/s")
    print(f"  Tokens: {n}")


if __name__ == "__main__":
    main()
