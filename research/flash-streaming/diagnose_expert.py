"""
Single-layer expert comparison: HF native vs Sniped forward.
Determines if the bug is in expert computation, accumulation, or weight layout.

Key test: does HF use fused gate_up_proj with [gate|up] or [up|gate] ordering?
If ordering is swapped, SiLU gets applied to wrong projection → structured garbage.
"""

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

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

device = "cpu"  # run on CPU to avoid VRAM issues
original_dir = "/workspace/qwen35-122b-a10b-4bit"
expert_dir = "/workspace/qwen35-122b-stream/experts"
pinned_path = "/workspace/qwen35-122b-stream/pinned.safetensors"
TEST_LAYER = 0

config = AutoConfig.from_pretrained(original_dir, trust_remote_code=True)
text_cfg = config.text_config if hasattr(config, 'text_config') else config

print("=" * 60)
print("  EXPERT COMPARISON: HF native vs Sniped")
print("=" * 60)

# Create model — keep experts for test layer only
print("\n[1] Creating model skeleton...")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(text_cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)

for i in range(text_cfg.num_hidden_layers):
    if i != TEST_LAYER:
        model.model.layers[i].mlp.experts = torch.nn.ModuleList()

# Materialize as zeros
for name, param in list(model.named_parameters()):
    if param.device == torch.device("meta"):
        set_module_tensor_to_device(model, name, device=device,
            value=torch.zeros(param.shape, dtype=torch.bfloat16))

# Print expert module structure
moe = model.model.layers[TEST_LAYER].mlp
experts_mod = moe.experts
print(f"\n[2] Expert module structure:")
print(f"  Type: {type(experts_mod).__name__}")
for name, param in experts_mod.named_parameters():
    print(f"  {name}: {param.shape} {param.dtype}")

# Check if fused gate_up_proj
has_fused = hasattr(experts_mod, 'gate_up_proj')
print(f"\n  Fused gate_up_proj: {has_fused}")
if has_fused:
    print(f"  gate_up_proj shape: {experts_mod.gate_up_proj.shape}")
    print(f"  down_proj shape: {experts_mod.down_proj.shape}")

# Inject gate weight from pinned
print(f"\n[3] Injecting gate weight...")
with safe_open(pinned_path, framework="pt", device="cpu") as f:
    for suffix in ["weight", "scales", "biases"]:
        k = f"language_model.model.layers.{TEST_LAYER}.mlp.gate.{suffix}"
        if k in f.keys():
            t = f.get_tensor(k)
            mapped = k.replace("language_model.", "", 1)
            if suffix == "weight" and t.dtype == torch.uint32:
                sk = k.replace(".weight", ".scales")
                bk = k.replace(".weight", ".biases")
                s = f.get_tensor(sk)
                b = f.get_tensor(bk)
                dq = dequantize_mlx_4bit(t, s, b, GROUP_SIZE)
                set_module_tensor_to_device(model, mapped, device=device, value=dq)
                print(f"  Injected gate: {dq.shape}, std={dq.float().std():.6f}")

# Inject shared expert weights
print(f"\n[4] Injecting shared expert weights...")
with safe_open(pinned_path, framework="pt", device="cpu") as f:
    shared_keys = [k for k in f.keys() if f'layers.{TEST_LAYER}.mlp.shared_expert' in k]
    bases = {}
    for k in shared_keys:
        if k.endswith(".scales"):
            bases.setdefault(k[:-7], {})["scales"] = k
        elif k.endswith(".biases"):
            bases.setdefault(k[:-7], {})["biases"] = k
        elif k.endswith(".weight"):
            bases.setdefault(k[:-7], {})["weight"] = k
    for base, parts in bases.items():
        if "weight" in parts and "scales" in parts:
            w = f.get_tensor(parts["weight"])
            s = f.get_tensor(parts["scales"])
            b = f.get_tensor(parts["biases"])
            dq = dequantize_mlx_4bit(w, s, b, GROUP_SIZE)
            target = base.replace("language_model.", "", 1) + ".weight"
            try:
                set_module_tensor_to_device(model, target, device=device, value=dq)
                print(f"  Injected: {target} {dq.shape}")
            except Exception as e:
                print(f"  Failed: {target} - {e}")

# Load expert weights into HF model
print(f"\n[5] Loading all 256 experts into HF model for layer {TEST_LAYER}...")
expert_path = f"{expert_dir}/layer_{TEST_LAYER:02d}.safetensors"
with safe_open(expert_path, framework="pt", device="cpu") as f:
    gate_w = dequantize_mlx_4bit(f.get_tensor("gate_proj.weight"), f.get_tensor("gate_proj.scales"), f.get_tensor("gate_proj.biases"), GROUP_SIZE)
    up_w = dequantize_mlx_4bit(f.get_tensor("up_proj.weight"), f.get_tensor("up_proj.scales"), f.get_tensor("up_proj.biases"), GROUP_SIZE)
    down_w = dequantize_mlx_4bit(f.get_tensor("down_proj.weight"), f.get_tensor("down_proj.scales"), f.get_tensor("down_proj.biases"), GROUP_SIZE)

print(f"  gate_proj: {gate_w.shape}")  # [256, intermediate, hidden]
print(f"  up_proj:   {up_w.shape}")
print(f"  down_proj: {down_w.shape}")

# Inject into HF model's expert module
if has_fused:
    # HF uses fused [gate|up] or [up|gate] — need to determine order
    fused = torch.cat([gate_w, up_w], dim=1)  # [256, 2*intermediate, hidden]
    print(f"  Fused gate_up: {fused.shape}")
    print(f"  HF expects:    {experts_mod.gate_up_proj.shape}")
    if fused.shape == experts_mod.gate_up_proj.shape:
        experts_mod.gate_up_proj.data = fused
        print(f"  Injected gate_up_proj (gate|up order)")
    else:
        # Try swapped order
        fused_swap = torch.cat([up_w, gate_w], dim=1)
        if fused_swap.shape == experts_mod.gate_up_proj.shape:
            experts_mod.gate_up_proj.data = fused_swap
            print(f"  Injected gate_up_proj (up|gate order — SWAPPED)")

    experts_mod.down_proj.data = down_w
    print(f"  Injected down_proj")
else:
    # Separate gate/up/down
    for i in range(min(256, len(experts_mod))):
        experts_mod[i].gate_proj.weight.data = gate_w[i]
        experts_mod[i].up_proj.weight.data = up_w[i]
        experts_mod[i].down_proj.weight.data = down_w[i]
    print(f"  Injected {min(256, len(experts_mod))} separate experts")

# Create test input
print(f"\n[6] Running comparison...")
torch.manual_seed(42)
hidden = torch.randn(1, 4, text_cfg.hidden_size, dtype=torch.bfloat16, device=device)

# Run HF native forward
print("  HF native forward...")
try:
    hf_out = moe(hidden)
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]
    print(f"  HF output: shape={hf_out.shape} mean={hf_out.float().mean():.6f} std={hf_out.float().std():.6f}")
except Exception as e:
    print(f"  HF forward FAILED: {e}")
    hf_out = None

# Run sniped forward
print("  Sniped forward...")
x = hidden.reshape(-1, text_cfg.hidden_size)
gate_out = moe.gate(x)
if isinstance(gate_out, tuple) and len(gate_out) == 3:
    _, topk_w, topk_idx = gate_out
    topk_w = topk_w.to(torch.bfloat16)
else:
    scores = F.softmax(gate_out, dim=-1, dtype=torch.float32)
    topk_w, topk_idx = torch.topk(scores, text_cfg.num_experts_per_tok, dim=-1)
    topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).to(torch.bfloat16)

print(f"  Routing: indices={topk_idx[0,:3].tolist()}... weights={topk_w[0,:3].tolist()}...")

needed = topk_idx.unique().tolist()
with safe_open(expert_path, framework="pt", device="cpu") as f:
    expert_data = {}
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        w = torch.stack([f.get_tensor(f"{proj}.weight")[i] for i in needed])
        s = torch.stack([f.get_tensor(f"{proj}.scales")[i] for i in needed])
        b = torch.stack([f.get_tensor(f"{proj}.biases")[i] for i in needed])
        expert_data[proj] = dequantize_mlx_4bit(w, s, b, GROUP_SIZE)

output = torch.zeros_like(x)
for local_idx, eid in enumerate(needed):
    mask = (topk_idx == eid)
    token_mask = mask.any(dim=-1)
    tidx = token_mask.nonzero(as_tuple=True)[0]
    if len(tidx) == 0:
        continue
    w = (topk_w * mask.to(topk_w.dtype)).sum(dim=-1)
    inp = x[tidx]
    g = F.silu(inp @ expert_data["gate_proj"][local_idx].t())
    u = inp @ expert_data["up_proj"][local_idx].t()
    out = (g * u) @ expert_data["down_proj"][local_idx].t()
    output[tidx] += w[tidx].unsqueeze(-1) * out

# Add shared expert
shared = moe.shared_expert(x)
if moe.shared_expert_gate is not None:
    shared = shared * torch.sigmoid(moe.shared_expert_gate(x))
output = output + shared
sniped_out = output.reshape(hidden.shape)
print(f"  Sniped output: shape={sniped_out.shape} mean={sniped_out.float().mean():.6f} std={sniped_out.float().std():.6f}")

# Compare
if hf_out is not None:
    diff = (hf_out.float() - sniped_out.float()).abs()
    print(f"\n[7] COMPARISON:")
    print(f"  Max diff:  {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Rel error: {(diff / (hf_out.float().abs() + 1e-8)).mean():.4%}")
    if diff.max() < 0.01:
        print("  >>> MATCH — expert computation is correct")
    elif diff.max() < 0.1:
        print("  >>> CLOSE — small numerical differences (precision)")
    else:
        print("  >>> MISMATCH — bug in expert computation or weight layout")

        # Try swapped gate/up
        print("\n  Trying SWAPPED gate/up...")
        output2 = torch.zeros_like(x)
        for local_idx, eid in enumerate(needed):
            mask = (topk_idx == eid)
            token_mask = mask.any(dim=-1)
            tidx = token_mask.nonzero(as_tuple=True)[0]
            if len(tidx) == 0:
                continue
            w = (topk_w * mask.to(topk_w.dtype)).sum(dim=-1)
            inp = x[tidx]
            # SWAPPED: up gets silu, gate is linear
            u = F.silu(inp @ expert_data["up_proj"][local_idx].t())
            g = inp @ expert_data["gate_proj"][local_idx].t()
            out = (u * g) @ expert_data["down_proj"][local_idx].t()
            output2[tidx] += w[tidx].unsqueeze(-1) * out
        output2 = output2 + shared
        sniped_swap = output2.reshape(hidden.shape)
        diff2 = (hf_out.float() - sniped_swap.float()).abs()
        print(f"  Swapped max diff:  {diff2.max():.6f}")
        print(f"  Swapped mean diff: {diff2.mean():.6f}")
        if diff2.max() < diff.max():
            print("  >>> SWAPPED IS CLOSER — gate/up ordering is wrong!")
        else:
            print("  >>> SWAP DIDN'T HELP — issue is elsewhere")

print("\n" + "=" * 60)
