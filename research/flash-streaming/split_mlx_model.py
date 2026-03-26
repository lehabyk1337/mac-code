"""
Split a pre-converted MLX model into Flash Stream format.

Input: mlx-community/Qwen3.5-35B-A3B-4bit (correctly converted safetensors)
Output: pinned.safetensors + per-layer expert binary files

No GGUF parsing. No custom transposition. The MLX model has correct weights.
We just split them by access pattern.
"""

import os
import sys
import gc
import json
import time
import numpy as np
import mlx.core as mx

MLX_MODEL_DIR = "/Users/bigneek/models/qwen35-35b-mlx-4bit"
OUTPUT_DIR = "/Users/bigneek/models/qwen35-35b-moe-stream"
PAGE_SIZE = 16384
NUM_LAYERS = 40
NUM_EXPERTS = 256


def align_up(offset):
    return (offset + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)


def main():
    os.makedirs(f"{OUTPUT_DIR}/experts", exist_ok=True)

    print("=" * 60)
    print("  Split MLX Model → Flash Stream Format")
    print("  Source: mlx-community/Qwen3.5-35B-A3B-4bit")
    print("=" * 60)

    # Load all safetensors shards
    print("\nLoading MLX model safetensors...")
    t0 = time.time()

    import glob
    shard_files = sorted(glob.glob(f"{MLX_MODEL_DIR}/model-*.safetensors"))
    print(f"  {len(shard_files)} shards")

    all_weights = {}
    for sf in shard_files:
        shard = mx.load(sf)
        all_weights.update(shard)
        print(f"  Loaded {os.path.basename(sf)}: {len(shard)} arrays")
        del shard

    print(f"  Total: {len(all_weights)} arrays in {time.time()-t0:.1f}s")

    # Categorize: pinned vs expert
    pinned = {}
    expert_layers = {}  # layer_idx → {local_name: array}

    for raw_name, arr in all_weights.items():
        # Strip "language_model." prefix if present
        name = raw_name.replace("language_model.", "") if raw_name.startswith("language_model.") else raw_name

        if "switch_mlp" in name:
            # Expert tensor — extract layer index and local name
            parts = name.split(".")
            layer_idx = int(parts[2])  # model.layers.XX.mlp.switch_mlp...
            local_name = ".".join(parts[3:])  # mlp.switch_mlp.gate_proj.weight etc

            if layer_idx not in expert_layers:
                expert_layers[layer_idx] = {}
            expert_layers[layer_idx][local_name] = arr
        else:
            # Store with stripped name (no language_model. prefix)
            pinned[name] = arr

    print(f"\n  Pinned: {len(pinned)} arrays")
    print(f"  Expert layers: {len(expert_layers)}")

    # Save pinned
    print("\n  Saving pinned.safetensors...")
    mx.save_safetensors(f"{OUTPUT_DIR}/pinned.safetensors", pinned)
    pinned_bytes = sum(v.nbytes for v in pinned.values())
    print(f"    {pinned_bytes/1e9:.2f} GB")

    # Verify critical SSM tensors
    print("\n  Verifying SSM tensors...")
    for k in ["model.layers.0.linear_attn.A_log",
              "model.layers.0.linear_attn.dt_bias",
              "model.layers.0.linear_attn.in_proj_qkv.weight"]:
        if k in pinned:
            v = pinned[k]
            mx.eval(v)
            nz = mx.abs(v.astype(mx.float32)).sum().item() > 0
            print(f"    {'✓' if nz else '✗'} {k}: {v.shape} nonzero={nz}")
        else:
            # Check quantized
            base = k.replace(".weight", "")
            if f"{base}.weight" in pinned:
                print(f"    ✓ {k}: QUANTIZED")
            else:
                print(f"    ✗ {k}: MISSING")

    del pinned
    gc.collect()

    # Save expert layers as 16KB-aligned binary
    print(f"\n  Saving {len(expert_layers)} expert layers...")
    total_expert_bytes = 0

    for layer_idx in sorted(expert_layers.keys()):
        data = expert_layers[layer_idx]

        # Expert weights are stored as [num_experts, out, in_packed] for QuantizedSwitchLinear
        # We need to save per-expert slices for selective reading

        # Determine tensor order and per-expert sizes
        proj_names = sorted(set(k.split(".")[2] for k in data.keys() if "." in k))
        # e.g., ["gate_proj", "down_proj", "up_proj"]

        tensor_order = []
        for proj in ["mlp.switch_mlp.gate_proj", "mlp.switch_mlp.up_proj", "mlp.switch_mlp.down_proj"]:
            for suffix in [".weight", ".scales", ".biases"]:
                key = f"{proj}{suffix}"
                if key in data:
                    tensor_order.append(key)

        # Calculate per-expert block size
        expert_block_size = 0
        tensor_sizes = {}
        for key in tensor_order:
            arr = data[key]
            mx.eval(arr)
            per_expert = arr[0].nbytes  # first expert slice
            tensor_sizes[key] = per_expert
            expert_block_size += per_expert
        expert_block_size = align_up(expert_block_size)

        # Build layout
        inner_offset = 0
        layout_tensors = {}
        for key in tensor_order:
            arr = data[key]
            layout_tensors[key] = {
                "inner_offset": inner_offset,
                "nbytes": tensor_sizes[key],
                "dtype": str(arr[0].dtype),
                "shape_per_expert": list(arr[0].shape),
            }
            inner_offset += tensor_sizes[key]

        total_size = PAGE_SIZE + NUM_EXPERTS * expert_block_size

        header = json.dumps({
            "format": "moe_flash_v1",
            "page_size": PAGE_SIZE,
            "layer_idx": layer_idx,
            "total_size": total_size,
            "layout": {
                "num_experts": NUM_EXPERTS,
                "expert_block_size": expert_block_size,
                "data_start": PAGE_SIZE,
                "tensors": layout_tensors,
            }
        }).encode()

        out_path = f"{OUTPUT_DIR}/experts/layer_{layer_idx:02d}.bin"
        with open(out_path, "wb") as f:
            f.write(header + b"\x00" * (PAGE_SIZE - len(header)))

            for expert_id in range(NUM_EXPERTS):
                block_start = PAGE_SIZE + expert_id * expert_block_size
                if f.tell() < block_start:
                    f.write(b"\x00" * (block_start - f.tell()))

                for key in tensor_order:
                    arr = data[key]
                    expert_slice = arr[expert_id]
                    mx.eval(expert_slice)
                    # Convert to bytes via mx.save then read, or use numpy
                    raw_bytes = memoryview(expert_slice)
                    f.write(bytes(raw_bytes))

            if f.tell() < total_size:
                f.write(b"\x00" * (total_size - f.tell()))

        total_expert_bytes += total_size

        if layer_idx % 10 == 0:
            print(f"    Layer {layer_idx}: {total_size/1e6:.1f} MB")

        del data
        gc.collect()

    print(f"    Total experts: {total_expert_bytes/1e9:.2f} GB")

    # Copy config
    config_src = f"{MLX_MODEL_DIR}/config.json"
    if os.path.exists(config_src):
        import shutil
        with open(config_src) as f:
            orig_config = json.load(f)

        # Create our streaming config
        tc = orig_config.get("text_config", orig_config)
        stream_config = {
            "model_type": tc.get("model_type", "qwen3_5_moe_text"),
            "hidden_size": tc.get("hidden_size", 2048),
            "num_hidden_layers": tc.get("num_hidden_layers", 40),
            "num_attention_heads": tc.get("num_attention_heads", 16),
            "num_key_value_heads": tc.get("num_key_value_heads", 2),
            "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
            "vocab_size": tc.get("vocab_size", 248320),
            "max_position_embeddings": tc.get("max_position_embeddings", 262144),
            "head_dim": tc.get("head_dim", 256),
            "tie_word_embeddings": orig_config.get("tie_word_embeddings", False),
            "num_experts": tc.get("num_experts", 256),
            "num_experts_per_tok": tc.get("num_experts_per_tok", 8),
            "shared_expert_intermediate_size": tc.get("shared_expert_intermediate_size", 512),
            "moe_intermediate_size": tc.get("moe_intermediate_size", 512),
            "linear_num_value_heads": tc.get("linear_num_value_heads", 32),
            "linear_num_key_heads": tc.get("linear_num_key_heads", 16),
            "linear_key_head_dim": tc.get("linear_key_head_dim", 128),
            "linear_value_head_dim": tc.get("linear_value_head_dim", 128),
            "linear_conv_kernel_dim": tc.get("linear_conv_kernel_dim", 4),
            "full_attention_interval": tc.get("full_attention_interval", 4),
            "rope_parameters": tc.get("rope_parameters"),
            "quantization": orig_config.get("quantization", {"bits": 4, "group_size": 64}),
            "streaming": {
                "pinned_file": "pinned.safetensors",
                "expert_dir": "experts/",
                "num_layers": NUM_LAYERS,
                "num_experts": NUM_EXPERTS,
                "experts_per_tok": 8,
            }
        }
        with open(f"{OUTPUT_DIR}/config.json", "w") as f:
            json.dump(stream_config, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s!")
    print(f"  Pinned: {pinned_bytes/1e9:.2f} GB")
    print(f"  Experts: {total_expert_bytes/1e9:.2f} GB")


if __name__ == "__main__":
    main()
