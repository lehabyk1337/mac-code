"""
Flash MoE — The Agentic Speed Breakthrough.

Qwen3.5-35B-A3B at Q4_K_M (22 GB) on 16 GB Mac.
Only 1.1 GB pinned in RAM. 256 experts on SSD.
Router picks 8 active experts → pread only those → compute → discard.

Per-token SSD: 0.57 GB → projected 5-10 tok/s.
"""

import time
import os
import sys
import json
import gc

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from expert_io import MoEExpertReader

MODEL_DIR = "/Users/bigneek/models/qwen35-35b-moe-stream"
BITS = 4
GROUP_SIZE = 64


def run_expert_ffn(x, expert_data, top_k_indices, top_k_weights):
    """
    Compute MoE FFN for active experts only.

    x: [B, L, D] hidden state
    expert_data: dict[expert_id] → dict of weight/scales/biases arrays
    top_k_indices: [B, L, K] expert indices
    top_k_weights: [B, L, K] expert weights (softmaxed)

    Returns: [B, L, D] weighted sum of expert outputs
    """
    B, L, D = x.shape
    K = top_k_indices.shape[-1]

    output = mx.zeros_like(x)

    for k in range(K):
        expert_ids = top_k_indices[..., k]  # [B, L]
        weights = top_k_weights[..., k:k+1]  # [B, L, 1]

        # For single-token decode, there's one expert_id per position
        eid = expert_ids[0, 0].item()

        if eid not in expert_data:
            continue

        ed = expert_data[eid]

        # SwiGLU: down(silu(gate(x)) * up(x))
        gate = mx.quantized_matmul(
            x, ed["mlp.switch_mlp.gate_proj.weight"],
            scales=ed["mlp.switch_mlp.gate_proj.scales"],
            biases=ed["mlp.switch_mlp.gate_proj.biases"],
            transpose=True, group_size=GROUP_SIZE, bits=BITS,
        )
        up = mx.quantized_matmul(
            x, ed["mlp.switch_mlp.up_proj.weight"],
            scales=ed["mlp.switch_mlp.up_proj.scales"],
            biases=ed["mlp.switch_mlp.up_proj.biases"],
            transpose=True, group_size=GROUP_SIZE, bits=BITS,
        )
        hidden = nn.silu(gate) * up
        expert_out = mx.quantized_matmul(
            hidden, ed["mlp.switch_mlp.down_proj.weight"],
            scales=ed["mlp.switch_mlp.down_proj.scales"],
            biases=ed["mlp.switch_mlp.down_proj.biases"],
            transpose=True, group_size=GROUP_SIZE, bits=BITS,
        )

        output = output + weights * expert_out

    return output


def main():
    print("=" * 60)
    print("  FLASH MOE — 35B Model at Agentic Speed")
    print("  22 GB model · 16 GB RAM · 8/256 experts streamed")
    print("=" * 60)

    with open(f"{MODEL_DIR}/config.json") as f:
        config = json.load(f)

    num_layers = config["num_hidden_layers"]
    streaming = config["streaming"]

    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    args = TextModelArgs(
        model_type=config.get("model_type", "qwen3_5_moe_text"),
        hidden_size=config["hidden_size"],
        num_hidden_layers=num_layers,
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        rms_norm_eps=config["rms_norm_eps"],
        vocab_size=config["vocab_size"],
        max_position_embeddings=config["max_position_embeddings"],
        head_dim=config.get("head_dim"),
        tie_word_embeddings=config["tie_word_embeddings"],
        num_experts=config["num_experts"],
        num_experts_per_tok=config["num_experts_per_tok"],
        shared_expert_intermediate_size=config["shared_expert_intermediate_size"],
        moe_intermediate_size=config["moe_intermediate_size"],
        linear_num_value_heads=config.get("linear_num_value_heads", 48),
        linear_num_key_heads=config.get("linear_num_key_heads", 16),
        linear_key_head_dim=config.get("linear_key_head_dim", 128),
        linear_value_head_dim=config.get("linear_value_head_dim", 128),
        linear_conv_kernel_dim=config.get("linear_conv_kernel_dim", 4),
        full_attention_interval=config.get("full_attention_interval", 4),
        rope_parameters=config.get("rope_parameters"),
    )

    print(f"\nCreating model...")
    model = TextModel(args)

    # Surgical quantization: only quantize large Linear layers, NOT SSM params
    SSM_PROTECT = {"conv1d"}  # Only conv1d needs special handling

    def should_quantize(path, module):
        # Always quantize Embedding
        if isinstance(module, nn.Embedding):
            return True
        if not isinstance(module, nn.Linear):
            return False
        # Protect small SSM layers from quantization
        if any(k in path for k in SSM_PROTECT):
            return False
        # Don't quantize if input dim < group_size (last dim is input for Linear)
        if module.weight.shape[-1] < GROUP_SIZE:
            return False
        return True

    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS, class_predicate=should_quantize)

    mx.set_memory_limit(10 * 1024**3)
    mx.set_cache_limit(512 * 1024**2)

    # Load pinned weights — now SSM params are still nn.Linear, so float16 weights load fine
    print("Loading pinned weights...")
    t0 = time.time()
    pinned = mx.load(f"{MODEL_DIR}/pinned.safetensors")
    model.load_weights(list(pinned.items()), strict=False)

    # Eval only non-expert params
    params = [p for name, p in tree_flatten(model.parameters()) if "switch_mlp" not in name]
    mx.eval(*params)
    del pinned
    gc.collect()
    mx.clear_cache()

    pinned_gb = sum(p.nbytes for p in params) / 1e9
    print(f"  {pinned_gb:.2f} GB in {time.time()-t0:.1f}s")
    print(f"  Active memory: {mx.get_active_memory()/1e9:.2f} GB")

    # Initialize expert reader
    print("\nInitializing MoE expert reader (8 threads, F_NOCACHE)...")
    reader = MoEExpertReader(
        f"{MODEL_DIR}/{streaming['expert_dir']}",
        num_layers, num_workers=8,
    )

    # Tokenizer
    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B", trust_remote_code=True)

    cache = model.make_cache()

    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    def forward_moe_streaming(input_ids):
        h = model.model.embed_tokens(input_ids)

        ssm_idx = model.model.ssm_idx
        fa_idx = model.model.fa_idx
        fa_mask = create_attention_mask(h, cache[fa_idx])
        ssm_mask = create_ssm_mask(h, cache[ssm_idx])

        for i in range(num_layers):
            layer = model.model.layers[i]

            # === ATTENTION (pinned in RAM) ===
            mask = ssm_mask if layer.is_linear else fa_mask
            normed = layer.input_layernorm(h)
            if layer.is_linear:
                attn_out = layer.linear_attn(normed, mask=mask, cache=cache[i])
            else:
                attn_out = layer.self_attn(normed, mask=mask, cache=cache[i])
            h = h + attn_out
            mx.eval(h)

            # === MoE FFN (experts streamed from SSD) ===
            normed = layer.post_attention_layernorm(h)

            # Router (pinned in RAM — tiny linear)
            gates = layer.mlp.gate(normed)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = layer.mlp.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if layer.mlp.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)
            mx.eval(inds, scores)

            # Get active expert IDs
            active_ids = set()
            inds_np = np.array(inds).flatten()
            for eid in inds_np:
                active_ids.add(int(eid))

            # Prefetch next layer's experts (heuristic: same experts often repeat)
            if i + 1 < num_layers:
                reader.prefetch_experts(i + 1, list(active_ids))

            # Read active experts from SSD
            expert_data = reader.get_experts(i, list(active_ids))

            # Compute expert FFN
            expert_out = run_expert_ffn(normed, expert_data, inds, scores)

            # Shared expert (pinned in RAM)
            shared_out = layer.mlp.shared_expert(normed)
            shared_gate = mx.sigmoid(layer.mlp.shared_expert_gate(normed))
            if shared_gate.ndim < shared_out.ndim:
                shared_gate = shared_gate[..., None]
            expert_out = expert_out + shared_gate * shared_out

            h = h + expert_out
            mx.eval(h)

            del expert_data, expert_out, normed, attn_out
            mx.clear_cache()

            if i % 10 == 0:
                print(f"    [layer {i}] mem={mx.get_active_memory()/1e9:.2f} GB", flush=True)

        h = model.model.norm(h)
        return model.lm_head(h)

    # Test
    prompt = "Explain how mixture of experts models work in one paragraph."
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    import subprocess
    subprocess.run(["sudo", "purge"], capture_output=True)

    print(f"\n--- Prefill ({len(tokens)} tokens) ---")
    t0 = time.time()
    logits = forward_moe_streaming(input_ids)
    mx.eval(logits)
    t_pf = time.time() - t0
    print(f"  {t_pf:.2f}s ({len(tokens)/t_pf:.2f} tok/s)")
    print(f"  {reader.stats()}")

    # Decode
    temperature = 0.7
    rep_penalty = 1.2
    max_tokens = 50

    print(f"\n--- Decode (max {max_tokens}) ---")
    generated = []
    t_decode = time.time()

    for step in range(max_tokens):
        next_logits = logits[:, -1, :]
        if generated:
            seen = mx.array(list(set(generated[-50:])))
            pl = next_logits[:, seen]
            pl = mx.where(pl > 0, pl / rep_penalty, pl * rep_penalty)
            next_logits[:, seen] = pl

        probs = mx.softmax(next_logits / temperature, axis=-1)
        token = mx.random.categorical(mx.log(probs + 1e-10))
        mx.eval(token)
        token_id = token.item()

        if token_id in (248044, 248045):
            break

        generated.append(token_id)
        print(tokenizer.decode([token_id]), end="", flush=True)

        logits = forward_moe_streaming(token.reshape(1, 1))
        mx.eval(logits)

        if (step + 1) % 10 == 0:
            elapsed = time.time() - t_decode
            tps = (step + 1) / elapsed
            mem = mx.get_active_memory() / 1e9
            print(f" [{tps:.2f} tok/s, {mem:.1f}GB]", flush=True)

    t_total = time.time() - t_decode
    n = len(generated)
    tps = n / t_total if t_total > 0 else 0

    output = tokenizer.decode(generated)
    print(f"\n\nDecode: {n} tokens in {t_total:.1f}s ({tps:.2f} tok/s)")
    print(f"{reader.stats()}")
    print(f"Memory: {mx.get_active_memory()/1e9:.2f} GB")

    r = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True)
    print(r.stdout.strip())

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")
    print(f"{'='*60}")
    print(f"\n  Model: Qwen3.5-35B-A3B Q4_K_M (22 GB total)")
    print(f"  Pinned: {pinned_gb:.1f} GB")
    print(f"  Experts per token: 8/256 streamed from SSD")
    print(f"  Speed: {tps:.2f} tok/s")

    reader.close()


if __name__ == "__main__":
    main()
