"""
THE DEFINITIVE TEST: single forward pass vs model.generate()

Test 1: model(input_ids, use_cache=False) — single forward, no cache
  → If top prediction is "Paris": model works, generate()'s cache is broken
  → If not "Paris": bug is cumulative across 48 layers

Test 2: Manual greedy loop with use_cache=False (slow but correct)
  → If coherent: proves the sniper works, cache is the only issue

Test 3: model.generate() for comparison
  → Shows whether cache-based generation differs from cacheless
"""

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from pathlib import Path
import time

GROUP_SIZE = 64

def dequantize_mlx_4bit(weight, scales, biases, group_size=64):
    if weight.dtype not in (torch.uint32, torch.int32):
        return weight.to(torch.bfloat16)
    orig_shape = weight.shape
    if weight.ndim == 3:
        batch = orig_shape[0]
        weight = weight.reshape(-1, orig_shape[-1])
        scales = scales.reshape(-1, scales.shape[-1])
        biases = biases.reshape(-1, biases.shape[-1])
    else:
        batch = None
    out_features = weight.shape[0]
    w = weight.to(torch.int32)
    shifts = torch.arange(0, 32, 4, device=w.device)
    unpacked = (w.unsqueeze(-1) >> shifts.view(1, 1, -1)) & 0xF
    in_features = unpacked.shape[1] * 8
    unpacked = unpacked.reshape(out_features, in_features).float()
    num_groups = in_features // group_size
    unpacked = unpacked.reshape(out_features, num_groups, group_size)
    dq = unpacked * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)
    result = dq.reshape(out_features, in_features).to(torch.bfloat16)
    if batch is not None:
        result = result.reshape(batch, orig_shape[1], in_features)
    return result

def remap_key(k):
    if k.startswith("language_model."):
        return k[len("language_model."):]
    return k

device = "cuda"
original_dir = "/workspace/qwen35-122b-a10b-4bit"
pinned_path = "/workspace/qwen35-122b-stream/pinned.safetensors"
expert_dir = "/workspace/qwen35-122b-stream/experts"

config = AutoConfig.from_pretrained(original_dir, trust_remote_code=True)
text_cfg = config.text_config if hasattr(config, 'text_config') else config

print("=" * 60)
print("  GENERATE vs FORWARD DIAGNOSTIC")
print("=" * 60)

# ── Build model ──
print("\n[1] Building model...")
t0 = time.time()

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(
        text_cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
    for i in range(text_cfg.num_hidden_layers):
        model.model.layers[i].mlp.experts = torch.nn.ModuleList()

for name, param in list(model.named_parameters()):
    if param.device == torch.device("meta"):
        set_module_tensor_to_device(model, name, device=device,
            value=torch.zeros(param.shape, dtype=torch.bfloat16))
for name, buf in list(model.named_buffers()):
    if buf.device == torch.device("meta") or buf.device != torch.device(device):
        try:
            set_module_tensor_to_device(model, name, device=device,
                value=torch.zeros(buf.shape, dtype=buf.dtype))
        except:
            pass

model = model.to(device)

# ── Inject weights ──
print("[2] Injecting weights...")
model_param_names = set(n for n, _ in model.named_parameters())
model_buffer_names = set(n for n, _ in model.named_buffers())
loaded = 0

with safe_open(pinned_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    bases = {}
    for k in keys:
        if k.endswith(".scales"):
            bases.setdefault(k[:-7], {})["scales"] = k
        elif k.endswith(".biases"):
            bases.setdefault(k[:-7], {})["biases"] = k
        elif k.endswith(".weight"):
            bases.setdefault(k[:-7], {})["weight"] = k
        else:
            bases.setdefault(k, {})["raw"] = k

    for base, parts in bases.items():
        if "raw" in parts:
            raw_key = parts["raw"]
            mapped = remap_key(raw_key)
            tensor = f.get_tensor(raw_key)
            if mapped in model_param_names or mapped in model_buffer_names:
                try:
                    val = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
                    set_module_tensor_to_device(model, mapped, device=device, value=val)
                    loaded += 1
                except:
                    pass
        elif "weight" in parts and "scales" in parts:
            w = f.get_tensor(parts["weight"])
            s = f.get_tensor(parts["scales"])
            b = f.get_tensor(parts["biases"])
            dq = dequantize_mlx_4bit(w, s, b, GROUP_SIZE)
            target = remap_key(base) + ".weight"
            if target in model_param_names:
                model_shape = dict(model.named_parameters())[target].shape
                if dq.shape != model_shape and dq.numel() == model_shape.numel():
                    dq = dq.reshape(model_shape)
                try:
                    set_module_tensor_to_device(model, target, device=device, value=dq)
                    loaded += 1
                except:
                    pass
            del dq
        elif "weight" in parts:
            w = f.get_tensor(parts["weight"])
            target = remap_key(base) + ".weight"
            if target in model_param_names:
                model_shape = dict(model.named_parameters())[target].shape
                val = w.to(torch.bfloat16)
                if val.shape != model_shape and val.numel() == model_shape.numel():
                    val = val.reshape(model_shape)
                try:
                    set_module_tensor_to_device(model, target, device=device, value=val)
                    loaded += 1
                except:
                    pass

print(f"  Loaded: {loaded}")

# ── Patch MoE ──
print("[3] Patching MoE layers...")

def make_sniped_forward(gate, shared_expert, shared_expert_gate, layer_idx, top_k, expert_dir):
    def forward(hidden_states):
        B, L, D = hidden_states.shape
        x = hidden_states.reshape(-1, D)
        gate_out = gate(x)
        if isinstance(gate_out, tuple) and len(gate_out) == 3:
            _, topk_w, topk_idx = gate_out
            topk_w = topk_w.to(hidden_states.dtype)
        else:
            scores = F.softmax(gate_out, dim=-1, dtype=torch.float32)
            topk_w, topk_idx = torch.topk(scores, top_k, dim=-1)
            topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)

        needed = topk_idx.unique().tolist()
        ep = f"{expert_dir}/layer_{layer_idx:02d}.safetensors"
        with safe_open(ep, framework="pt", device="cpu") as ef:
            expert_w = {}
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                fw = torch.stack([ef.get_tensor(f"{proj}.weight")[e] for e in needed]).to(x.device)
                fs = torch.stack([ef.get_tensor(f"{proj}.scales")[e] for e in needed]).to(x.device)
                fb = torch.stack([ef.get_tensor(f"{proj}.biases")[e] for e in needed]).to(x.device)
                expert_w[proj] = dequantize_mlx_4bit(fw, fs, fb, GROUP_SIZE)

        output = torch.zeros_like(x)
        for local_idx, eid in enumerate(needed):
            mask = (topk_idx == eid)
            token_mask = mask.any(dim=-1)
            tidx = token_mask.nonzero(as_tuple=True)[0]
            if len(tidx) == 0:
                continue
            w = (topk_w * mask.to(topk_w.dtype)).sum(dim=-1)
            inp = x[tidx]
            g = F.silu(inp @ expert_w["gate_proj"][local_idx].t())
            u = inp @ expert_w["up_proj"][local_idx].t()
            out = (g * u) @ expert_w["down_proj"][local_idx].t()
            output[tidx] += w[tidx].unsqueeze(-1) * out

        if shared_expert is not None:
            s_out = shared_expert(x)
            if shared_expert_gate is not None:
                s_out = s_out * torch.sigmoid(shared_expert_gate(x))
            output = output + s_out
        del expert_w
        return output.reshape(B, L, D)
    return forward

patched = 0
for i in range(text_cfg.num_hidden_layers):
    layer = model.model.layers[i]
    if hasattr(layer.mlp, 'gate') and layer.mlp.gate is not None:
        moe = layer.mlp
        moe.forward = make_sniped_forward(
            moe.gate, getattr(moe, 'shared_expert', None),
            getattr(moe, 'shared_expert_gate', None),
            i, text_cfg.num_experts_per_tok, expert_dir)
        patched += 1

print(f"  Patched: {patched}/{text_cfg.num_hidden_layers}")
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"  Setup time: {time.time()-t0:.1f}s")

model.eval()
tokenizer = AutoTokenizer.from_pretrained(original_dir, trust_remote_code=True)

# ── TEST 1: Single forward pass, no cache ──
print("\n" + "=" * 60)
print("TEST 1: Single forward pass (use_cache=False)")
print("=" * 60)

prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
attention_mask = torch.ones_like(input_ids)
print(f"  Prompt: '{prompt}' ({input_ids.shape[1]} tokens)")

t0 = time.time()
with torch.no_grad():
    out = model(input_ids, attention_mask=attention_mask, use_cache=False)

logits = out.logits if hasattr(out, 'logits') else out[0]
next_logits = logits[0, -1]
top10 = torch.topk(next_logits.float(), 10)
elapsed = time.time() - t0

print(f"  Time: {elapsed:.1f}s")
print(f"  Top 10 predictions:")
for i, (val, idx) in enumerate(zip(top10.values, top10.indices)):
    token = tokenizer.decode([idx.item()])
    print(f"    {i+1}. '{token}' (logit={val.item():.2f})")

# Check if Paris is in top 10
paris_tokens = tokenizer.encode("Paris", add_special_tokens=False)
paris_in_top10 = any(idx.item() in paris_tokens for idx in top10.indices)
print(f"\n  'Paris' in top 10: {paris_in_top10}")
if paris_in_top10:
    print("  >>> MODEL WORKS! Bug is in generate()'s cache handling.")
else:
    print("  >>> MODEL BROKEN at single forward level. Bug is in weights or layer composition.")

# ── TEST 2: Manual greedy loop (no cache) ──
print("\n" + "=" * 60)
print("TEST 2: Manual greedy decode (use_cache=False, 20 tokens)")
print("=" * 60)

generated = []
current_ids = input_ids.clone()
t0 = time.time()

for step in range(20):
    with torch.no_grad():
        out = model(current_ids, attention_mask=torch.ones_like(current_ids), use_cache=False)
    logits = out.logits if hasattr(out, 'logits') else out[0]
    next_token = logits[0, -1].argmax().item()

    if next_token in (248044, 248046):
        break

    generated.append(next_token)
    current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=device)], dim=1)

    chunk = tokenizer.decode([next_token])
    print(f"  Token {step}: '{chunk}' (id={next_token})", flush=True)

elapsed = time.time() - t0
output_text = tokenizer.decode(generated, skip_special_tokens=True)
tps = len(generated) / elapsed if elapsed > 0 else 0

print(f"\n  Full output: '{output_text}'")
print(f"  Speed: {tps:.3f} tok/s")
print(f"  Time: {elapsed:.1f}s")

# ── TEST 3: model.generate() for comparison ──
print("\n" + "=" * 60)
print("TEST 3: model.generate() (with cache)")
print("=" * 60)

t0 = time.time()
with torch.no_grad():
    gen_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
elapsed = time.time() - t0

new_tokens = gen_ids[0][input_ids.shape[1]:]
gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
tps = len(new_tokens) / elapsed if elapsed > 0 else 0

print(f"  Output: '{gen_text}'")
print(f"  Speed: {tps:.3f} tok/s")
print(f"  Time: {elapsed:.1f}s")

# ── COMPARISON ──
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"  Test 1 (single forward): top prediction = '{tokenizer.decode([top10.indices[0].item()])}'")
print(f"  Test 2 (manual greedy):  '{output_text}'")
print(f"  Test 3 (model.generate): '{gen_text}'")

if paris_in_top10:
    print("\n  VERDICT: Model produces correct predictions.")
    print("  If Test 2 is coherent but Test 3 is garbled → cache bug in generate()")
    print("  If both are coherent → FULL SUCCESS")
else:
    print("\n  VERDICT: Model is broken at the forward pass level.")
    print("  The bug is in weight injection or layer composition, not in generate().")

print("\n" + "=" * 60)
print("  DIAGNOSIS COMPLETE")
print("=" * 60)
