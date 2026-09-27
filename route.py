#!/usr/bin/env python3
"""
Stage 4: Tag-based routing for Curve Attention.

Probes tau fingerprints for concept tags, then routes information
through tag-specific regions of the B-spline sampling space.

After Formation (Stage 2), each head has learned query-dependent sampling
patterns. Different concepts produce different tau locations. Routing
shifts tau_proj.bias toward tag fingerprints so the model generates
through those conceptual channels.

Usage:
  # Generate with tags
  python route.py --prompt "Explain the theory of relativity" \
    --tags apples force pull boredom fun --alpha 0.5

  # Diagnostics: are fingerprints distinct?
  python route.py --diagnose --tags apples force pull gravity music

  # Alpha sweep: find safe routing strength
  python route.py --sweep-alpha --tags apples force

  # Compare routed vs unrouted
  python route.py --prompt "Explain relativity" --tags apples force --compare

  # Scheduled routing (cycle tags during generation)
  python route.py --prompt "Explain relativity" \
    --schedule "apples:20,force:20,fun:20" --alpha 0.5

  # Export edits.json for the inspector/simulator
  python route.py --prompt "Explain relativity" \
    --tags apples force --alpha 0.5 --export-edits routing_edits.json
"""

import argparse
import json
import math
import os
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from prototype_curve_attention import CurveAttention


# ── Model Loading ──────────────────────────────────────────────

def patch_block(model, block_idx, curve_attn):
    attn = model.model.layers[block_idx].self_attn
    nh = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    ng = nh // nkv
    hd = model.config.head_dim
    attn.curve_attention = curve_attn

    def make_fwd(am, ca, nh, nkv, ng, hd):
        def fwd(hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, cache_position=None, **kw):
            bsz, ql, _ = hidden_states.size()
            q = am.q_proj(hidden_states).view(bsz, ql, nh, hd).transpose(1, 2)
            k = am.k_proj(hidden_states).view(bsz, ql, nkv, hd).transpose(1, 2)
            v = am.v_proj(hidden_states).view(bsz, ql, nkv, hd).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if ng > 1:
                k = k.repeat_interleave(ng, dim=1)
                v = v.repeat_interleave(ng, dim=1)
            mask = attention_mask[:, :, :, :k.shape[-2]] if attention_mask is not None else None
            out = ca(q, k, v, mask=mask)
            out = out.transpose(1, 2).reshape(bsz, ql, -1)
            return am.o_proj(out), None
        return fwd

    attn.forward = make_fwd(attn, curve_attn, nh, nkv, ng, hd)


def load_curve_model(model_name, checkpoint_path, device):
    """Load model and patch with curve attention. Returns (model, curve_modules dict)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation='eager').to(device)
    model.config.use_cache = False
    model.eval()

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = ckpt['config']
    R = config['R']
    hd = config['head_dim']

    curve_modules = {}
    for block_idx, state in ckpt['curve_states'].items():
        block_idx = int(block_idx)
        ca = CurveAttention(hd, R).to(device)
        ca.load_state_dict(state)
        ca.eval()
        patch_block(model, block_idx, ca)
        curve_modules[block_idx] = ca

    return model, curve_modules


# ── Probing ────────────────────────────────────────────────────

def _install_tau_hooks(curve_modules):
    """Hook tau_proj on all curve modules to capture pre-sigmoid logits.
    Returns (captures_dict, hooks_list)."""
    captures = {}
    hooks = []
    for layer_idx, ca in curve_modules.items():
        cap = {}
        captures[layer_idx] = cap

        def make_hook(c):
            def hook_fn(module, inp, out):
                c["tau_logits"] = out.detach().cpu()
            return hook_fn

        h = ca.tau_proj.register_forward_hook(make_hook(cap))
        hooks.append(h)
    return captures, hooks


def _find_tag_positions(tokenizer, full_ids, tag_text):
    """Find token positions of tag_text within the full token sequence."""
    tag_ids = tokenizer.encode(tag_text, add_special_tokens=False)
    full_list = full_ids[0].tolist() if full_ids.dim() > 1 else full_ids.tolist()

    # Sliding window match
    for start in range(len(full_list) - len(tag_ids) + 1):
        if full_list[start:start + len(tag_ids)] == tag_ids:
            return list(range(start, start + len(tag_ids)))

    # Fallback: try matching individual tag tokens anywhere
    positions = []
    for tid in tag_ids:
        for i, fid in enumerate(full_list):
            if fid == tid and i not in positions:
                positions.append(i)
                break

    if positions:
        return positions

    # Last resort: use the last token (most likely the tag)
    return [len(full_list) - 2]  # -2 to skip EOS if present


def probe_single(model, tokenizer, curve_modules, text, device):
    """Run text through model, capture tau logits per layer.
    Returns: {layer_idx: tau_logits[H, S, R]}"""
    captures, hooks = _install_tau_hooks(curve_modules)
    ids = tokenizer.encode(text, return_tensors="pt").to(device)

    with torch.no_grad():
        model(ids, use_cache=False)

    for h in hooks:
        h.remove()

    result = {}
    for layer_idx, cap in captures.items():
        if "tau_logits" in cap:
            # Shape: [1, H, S, R] → [H, S, R]
            logits = cap["tau_logits"]
            if logits.dim() == 4:
                result[layer_idx] = logits[0]
            else:
                result[layer_idx] = logits
    return result, ids


def probe_tags(model, tokenizer, curve_modules, tags, device,
               context_template="This is about {tag}."):
    """Capture tau fingerprints for each tag.

    Returns: {tag: {layer_idx: fingerprint[H, R]}} — averaged over tag tokens.
    """
    fingerprints = {}

    for tag in tags:
        text = context_template.format(tag=tag)
        tau_logits, ids = probe_single(model, tokenizer, curve_modules, text, device)
        tag_positions = _find_tag_positions(tokenizer, ids, tag)

        fp = {}
        for layer_idx, logits in tau_logits.items():
            # logits: [H, S, R]
            # Extract tag token positions and average
            valid_pos = [p for p in tag_positions if p < logits.shape[1]]
            if valid_pos:
                tag_tau = logits[:, valid_pos, :].mean(dim=1)  # [H, R]
            else:
                tag_tau = logits[:, -1, :]  # fallback: last token
            fp[layer_idx] = tag_tau

        fingerprints[tag] = fp
        n_tok = len(tag_positions)
        print(f"  Probed '{tag}': {n_tok} token(s) at positions {tag_positions}")

    return fingerprints


def probe_baseline(model, tokenizer, curve_modules, device):
    """Probe a neutral sentence to get baseline tau fingerprint."""
    text = "This is a sentence."
    tau_logits, _ = probe_single(model, tokenizer, curve_modules, text, device)

    baseline = {}
    for layer_idx, logits in tau_logits.items():
        # Average over all token positions → [H, R]
        baseline[layer_idx] = logits.mean(dim=1)
    return baseline


# ── Routing Delta ──────────────────────────────────────────────

def compute_routing_delta(tag_fingerprints, baseline_fingerprint,
                          blend_weights=None):
    """Compute per-layer bias delta from tag fingerprints.

    tag_fingerprints: {tag: {layer: [H, R]}}
    baseline_fingerprint: {layer: [H, R]}
    blend_weights: {tag: float} or None (uniform)

    Returns: {layer_idx: delta_bias[R]}
    """
    tags = list(tag_fingerprints.keys())
    if blend_weights is None:
        blend_weights = {t: 1.0 for t in tags}
    total_w = sum(blend_weights[t] for t in tags)

    # Get all layers
    layers = set()
    for fp in tag_fingerprints.values():
        layers.update(fp.keys())

    delta = {}
    for li in sorted(layers):
        # Weighted average of tag fingerprints
        target = None
        for tag in tags:
            if li not in tag_fingerprints[tag]:
                continue
            w = blend_weights[tag] / total_w
            fp = tag_fingerprints[tag][li]  # [H, R]
            if target is None:
                target = w * fp
            else:
                target = target + w * fp

        if target is None or li not in baseline_fingerprint:
            continue

        baseline = baseline_fingerprint[li]  # [H, R]

        # Average over heads (tau_proj.bias is shared across heads)
        target_mean = target.mean(dim=0)    # [R]
        baseline_mean = baseline.mean(dim=0)  # [R]
        delta[li] = target_mean - baseline_mean

    return delta


# ── Apply Routing ──────────────────────────────────────────────

@contextmanager
def apply_static_routing(curve_modules, routing_delta, alpha=1.0):
    """Temporarily shift tau_proj.bias += alpha * delta. Restores on exit."""
    originals = {}
    try:
        for layer_idx, d in routing_delta.items():
            if layer_idx not in curve_modules:
                continue
            ca = curve_modules[layer_idx]
            originals[layer_idx] = ca.tau_proj.bias.data.clone()
            ca.tau_proj.bias.data += alpha * d.to(ca.tau_proj.bias.device)
        yield
    finally:
        for layer_idx, orig in originals.items():
            curve_modules[layer_idx].tau_proj.bias.data = orig


@contextmanager
def apply_ctrl_routing(curve_modules, ctrl_edits, alpha=1.0):
    """Temporarily patch eval_curve to add control point edits to middle positions.
    ctrl_edits: {layer_idx: delta[n_heads, head_dim]} from ControlPointPolicy.
    Middle region computed dynamically from actual sequence length.
    """
    patches = []
    try:
        for li, delta in ctrl_edits.items():
            if li not in curve_modules:
                continue
            ca = curve_modules[li]
            orig_eval = ca.eval_curve
            scaled_delta = alpha * delta.to(next(ca.parameters()).device)

            def make_patched(orig, d):
                def patched(coeffs, tau_01):
                    if coeffs.dim() == 4:
                        S = coeffs.shape[2]
                        s = S // 4
                        e = max(s + 1, 3 * S // 4)
                        mask = torch.zeros(S, 1, device=coeffs.device)
                        mask[s:e] = 1.0
                        edit = d.unsqueeze(0).unsqueeze(2) * mask.unsqueeze(0).unsqueeze(0)
                        coeffs = coeffs + edit
                    return orig(coeffs, tau_01)
                return patched

            ca.eval_curve = make_patched(orig_eval, scaled_delta)
            patches.append((ca, orig_eval))
        yield
    finally:
        for ca, orig in patches:
            ca.eval_curve = orig


# ── Generation ─────────────────────────────────────────────────

def generate(model, tokenizer, prompt, device, max_tokens=100, temperature=0.0):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    kwargs = dict(max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id,
                  eos_token_id=tokenizer.eos_token_id, use_cache=False)
    if temperature == 0.0:
        kwargs['do_sample'] = False
    else:
        kwargs.update(do_sample=True, temperature=temperature, top_k=50, top_p=0.95)
    with torch.no_grad():
        out = model.generate(ids, **kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def generate_scheduled(model, tokenizer, curve_modules, prompt,
                       tag_fingerprints, baseline_fp, schedule,
                       alpha=1.0, max_tokens=100, device=None):
    """Generate with scheduled tag routing — cycle tags during generation.

    schedule: [(tag_name, n_tokens), ...]
    """
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = []
    schedule_idx = 0
    tokens_in_tag = 0

    for step in range(max_tokens):
        tag, duration = schedule[schedule_idx % len(schedule)]
        delta = compute_routing_delta(
            {tag: tag_fingerprints[tag]}, baseline_fp)

        with apply_static_routing(curve_modules, delta, alpha):
            with torch.no_grad():
                output = model(ids, use_cache=False)
                next_logits = output.logits[0, -1, :]
                next_token = next_logits.argmax()

        generated.append(next_token.item())
        ids = torch.cat([ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)

        tokens_in_tag += 1
        if tokens_in_tag >= duration:
            schedule_idx += 1
            tokens_in_tag = 0

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)


# ── Diagnostics ────────────────────────────────────────────────

def diagnose_fingerprints(tag_fingerprints, baseline_fingerprint):
    """Analyze whether tag fingerprints are distinct enough for routing."""
    tags = list(tag_fingerprints.keys())
    n_tags = len(tags)
    layers = sorted(set().union(*[fp.keys() for fp in tag_fingerprints.values()]))

    print(f"\n{'=' * 60}")
    print(f"  TAG FINGERPRINT DIAGNOSTICS")
    print(f"{'=' * 60}")
    print(f"  Tags: {tags}")
    print(f"  Layers: {len(layers)}")

    # 1. Cross-tag cosine similarity
    print(f"\n  Cross-tag cosine similarity (flattened across layers+heads):")
    flat_fps = {}
    for tag in tags:
        parts = []
        for li in layers:
            if li in tag_fingerprints[tag]:
                parts.append(tag_fingerprints[tag][li].flatten())
        flat_fps[tag] = torch.cat(parts) if parts else torch.zeros(1)

    print(f"  {'':>12s}", end="")
    for t in tags:
        print(f"  {t[:8]:>8s}", end="")
    print()

    cosines = []
    for i, t1 in enumerate(tags):
        print(f"  {t1[:12]:>12s}", end="")
        for j, t2 in enumerate(tags):
            cos = F.cosine_similarity(
                flat_fps[t1].unsqueeze(0),
                flat_fps[t2].unsqueeze(0)
            ).item()
            print(f"  {cos:8.4f}", end="")
            if i < j:
                cosines.append(cos)
        print()

    mean_cos = sum(cosines) / len(cosines) if cosines else 0
    print(f"\n  Mean pairwise cosine: {mean_cos:.4f}")
    if mean_cos > 0.98:
        print(f"  WARNING: Fingerprints are very similar — routing may not steer effectively.")
        print(f"  Consider more formation training or higher alpha.")
    elif mean_cos > 0.95:
        print(f"  Fingerprints are somewhat similar — routing may work with higher alpha.")
    else:
        print(f"  Fingerprints are distinct — routing should work.")

    # 2. Per-layer discriminability
    print(f"\n  Per-layer discriminability (mean cross-tag L2 distance):")
    for li in layers:
        layer_fps = []
        for tag in tags:
            if li in tag_fingerprints[tag]:
                layer_fps.append(tag_fingerprints[tag][li].flatten())
        if len(layer_fps) < 2:
            continue

        dists = []
        for i in range(len(layer_fps)):
            for j in range(i + 1, len(layer_fps)):
                dists.append((layer_fps[i] - layer_fps[j]).norm().item())
        mean_dist = sum(dists) / len(dists)
        print(f"    Layer {li:>2d}: L2={mean_dist:.4f}")

    # 3. Per-head variance (which heads are most discriminative)
    print(f"\n  Most discriminative heads (highest cross-tag τ variance):")
    head_scores = []
    for li in layers:
        layer_fps = []
        for tag in tags:
            if li in tag_fingerprints[tag]:
                layer_fps.append(tag_fingerprints[tag][li])  # [H, R]
        if len(layer_fps) < 2:
            continue
        stacked = torch.stack(layer_fps)  # [n_tags, H, R]
        per_head_var = stacked.var(dim=0).mean(dim=-1)  # [H]
        for h_idx in range(per_head_var.shape[0]):
            head_scores.append((li, h_idx, per_head_var[h_idx].item()))

    head_scores.sort(key=lambda x: x[2], reverse=True)
    for li, h, score in head_scores[:10]:
        print(f"    Layer {li:>2d} Head {h:>2d}: variance={score:.6f}")

    # 4. Delta magnitude vs formation tau_max
    print(f"\n  Routing delta magnitudes:")
    delta = compute_routing_delta(tag_fingerprints, baseline_fingerprint)
    for li in sorted(delta.keys()):
        d = delta[li]
        print(f"    Layer {li:>2d}: |Δb| mean={d.abs().mean():.4f}, max={d.abs().max():.4f}")

    return mean_cos


def eval_ppl(model, eval_ids, device):
    model.eval()
    ids = eval_ids.to(device)
    with torch.no_grad():
        loss = model(ids, labels=ids, use_cache=False).loss.float().item()
    return math.exp(loss)


def alpha_sweep(model, tokenizer, curve_modules, routing_delta,
                eval_text, device):
    """Sweep alpha values and measure PPL impact."""
    eval_ids = torch.tensor(
        tokenizer.encode(eval_text, add_special_tokens=False)[:512]
    ).unsqueeze(0)

    base_ppl = eval_ppl(model, eval_ids, device)
    print(f"\n  Alpha sweep (baseline PPL: {base_ppl:.2f}):")
    print(f"  {'alpha':>8s}  {'PPL':>8s}  {'ratio':>8s}  {'status'}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*12}")

    for alpha in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        with apply_static_routing(curve_modules, routing_delta, alpha):
            ppl = eval_ppl(model, eval_ids, device)
        ratio = ppl / base_ppl
        status = "safe" if ratio < 1.5 else "degraded" if ratio < 3.0 else "BROKEN"
        print(f"  {alpha:8.2f}  {ppl:8.2f}  {ratio:8.2f}x  {status}")


# ── Export Edits (for Inspector/Simulator) ─────────────────────

def _capture_blended_v(model, tokenizer, curve_modules, text, device):
    """Run text, capture blended V coefficients per (layer, head, token).
    Returns: {layer_idx: blended[H, S, head_dim]}"""
    from inkan.basis import _bspline_basis

    captures = {}
    hooks = []

    for layer_idx, ca in curve_modules.items():
        cap = {}
        captures[layer_idx] = cap

        # Hook sample_proj input to get the curve-sampled values
        # But we actually need the blended coefficients BEFORE eval_curve
        # Hook tau_proj to get q, then we can derive blended from attention
        def make_hook(c):
            def hook_fn(module, inp, out):
                # sample_proj input is the sampled curve values: [B, H, S, R]
                # But we need the blended V coefficients (pre-eval_curve)
                # Those are the input to eval_curve, which is called in forward
                # We need a different approach — hook eval_curve
                pass
            return hook_fn

    # Better approach: monkey-patch eval_curve to capture coeffs
    for layer_idx, ca in curve_modules.items():
        cap = {}
        captures[layer_idx] = cap
        orig_eval = ca.eval_curve

        def make_patched(c, orig):
            def patched_eval(coeffs, tau_01):
                c["blended"] = coeffs.detach().cpu()
                c["tau_01"] = tau_01.detach().cpu()
                return orig(coeffs, tau_01)
            return patched_eval

        ca.eval_curve = make_patched(cap, orig_eval)
        hooks.append((ca, orig_eval))

    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        model(ids, use_cache=False)

    # Restore
    for ca, orig_eval in hooks:
        ca.eval_curve = orig_eval

    result = {}
    for layer_idx, cap in captures.items():
        if "blended" in cap:
            blended = cap["blended"]
            if blended.dim() == 4:  # [B, H, S, hd]
                result[layer_idx] = {
                    "blended": blended[0],  # [H, S, hd]
                    "tau_01": cap["tau_01"][0],  # [H, S, R]
                }
            else:
                result[layer_idx] = {
                    "blended": blended,
                    "tau_01": cap["tau_01"],
                }
    return result, ids


def export_edits(model, tokenizer, curve_modules, text, tags,
                 ctrl_edits, alpha, device, output_path,
                 model_name="meta-llama/Llama-3.2-1B-Instruct",
                 is_ctrl_point=True):
    """Export routing as edits.json compatible with the inspector.

    For control point policy: directly adds scaled delta to blended V coefficients.
    Only edits middle positions (first/last quarter anchored).
    """
    from datetime import datetime

    print(f"\n  Exporting edits to {output_path}...")

    # Capture original blended V coefficients
    orig_data, ids = _capture_blended_v(
        model, tokenizer, curve_modules, text, device)

    tokens = [tokenizer.decode([tid]) for tid in ids[0].tolist()]
    seq_len = ids.shape[1]
    s = seq_len // 4
    e = max(s + 1, 3 * seq_len // 4)

    edits = {"model": model_name, "tags": tags, "alpha": alpha, "edits": []}
    n_edits = 0

    for layer_idx, ca in curve_modules.items():
        if layer_idx not in orig_data or layer_idx not in ctrl_edits:
            continue

        blended = orig_data[layer_idx]["blended"]  # [H, S, hd]
        delta = ctrl_edits[layer_idx].cpu()         # [H, hd] or [R]

        if is_ctrl_point and delta.dim() == 2:
            # Control point edits: delta is [H, hd], add to middle positions
            nh = blended.shape[0]
            for h in range(nh):
                d_h = alpha * delta[h]  # [hd]
                if d_h.abs().max().item() < 1e-6:
                    continue
                for ti in range(s, e):
                    c_orig = blended[h, ti]  # [hd]
                    c_edited = c_orig + d_h
                    edits["edits"].append({
                        "block": layer_idx,
                        "head": h,
                        "token_idx": ti,
                        "token": tokens[ti].strip() if ti < len(tokens) else "",
                        "input_text": text,
                        "original_ctrl_pts": c_orig.tolist(),
                        "edited_ctrl_pts": c_edited.tolist(),
                        "timestamp": datetime.now().isoformat(),
                    })
                    n_edits += 1

    with open(output_path, "w") as f:
        json.dump(edits, f, indent=2)

    n_blocks = len(set(e["block"] for e in edits["edits"])) if edits["edits"] else 0
    print(f"  Exported {n_edits} edits across {n_blocks} blocks")
    print(f"  Middle positions [{s}..{e}] of {seq_len} tokens")
    print(f"  Saved: {output_path}")
    print(f"  Load in inspector: load edits {os.path.abspath(output_path)}")
    return output_path


# ── CLI ────────────────────────────────────────────────────────

def parse_schedule(schedule_str):
    """Parse 'tag1:20,tag2:20,tag3:20' into [(tag, n_tokens), ...]"""
    parts = schedule_str.split(",")
    schedule = []
    for part in parts:
        tag, n = part.strip().split(":")
        schedule.append((tag.strip(), int(n.strip())))
    return schedule


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Tag-based routing for Curve Attention")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--formation", default=None,
                        help="Formation checkpoint (default: auto-detect)")
    parser.add_argument("--calibration", default=None,
                        help="Calibration checkpoint (if no formation)")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--tags", nargs="+", default=None,
                        help="Tag concepts to route through")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Routing strength (0=none, 1=full)")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--compare", action="store_true",
                        help="Show side-by-side routed vs unrouted")
    parser.add_argument("--diagnose", action="store_true",
                        help="Run fingerprint diagnostics")
    parser.add_argument("--sweep-alpha", action="store_true",
                        help="Sweep alpha values and measure PPL")
    parser.add_argument("--schedule", default=None,
                        help="Scheduled routing: 'tag1:20,tag2:20'")
    parser.add_argument("--export-edits", default=None, metavar="PATH",
                        help="Export routing as edits.json for the inspector")
    parser.add_argument("--policy", default=None, metavar="PATH",
                        help="Trained routing policy checkpoint (from train_router.py)")
    parser.add_argument("--context-template", default="This is about {tag}.",
                        help="Template for probing tags")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Find checkpoint
    checkpoint = args.formation or args.calibration
    if checkpoint is None:
        for path in [
            "checkpoints/formation/formation.pt",
            "checkpoints/Llama-3.2-1B-Instruct-curve-attn/curve_attention.pt",
        ]:
            if os.path.exists(path):
                checkpoint = path
                break
    if checkpoint is None:
        print("Error: No checkpoint found. Provide --formation or --calibration.")
        return

    print(f"Tag-Based Routing")
    print(f"  Model: {args.model}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Device: {device}")

    # Load model
    print("\nLoading model...")
    model, curve_modules = load_curve_model(args.model, checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = len(curve_modules)
    print(f"  {n_layers} curve modules loaded")

    # Tags
    tags = args.tags
    if tags is None and args.schedule:
        schedule = parse_schedule(args.schedule)
        tags = list(set(t for t, _ in schedule))
    if tags is None:
        tags = ["apples", "force", "fun"]
        print(f"  Using default tags: {tags}")

    # ── Compute routing delta ──
    # Two modes: trained policy (fast, generalizes) or probing (no training needed)
    use_policy = False
    policy = None
    policy_deltas = None

    if args.policy:
        from train_router import ControlPointPolicy, install_ctrl_hooks, restore_ctrl_hooks
        print(f"\nLoading trained routing policy: {args.policy}")
        policy_ckpt = torch.load(args.policy, weights_only=False, map_location=device)
        policy_cfg = policy_ckpt['config']
        policy = ControlPointPolicy(
            embed_dim=policy_cfg['embed_dim'],
            layer_indices=policy_cfg['layer_indices'],
            n_heads=policy_cfg['n_heads'],
            head_dim=policy_cfg['head_dim'],
            hidden=policy_cfg['hidden_dim'],
        ).to(device)
        policy.load_state_dict(policy_ckpt['policy_state'])
        policy.eval()
        n_params = sum(p.numel() for p in policy.parameters())
        print(f"  Policy: {n_params:,} params (control point editing)")

        # Policy deltas will be computed after we have the prompt
        # (policy needs prompt hidden state + tag embeddings)
        _policy_tag_ids = []
        for tag in tags:
            tid = tokenizer.encode(tag, add_special_tokens=False)
            _policy_tag_ids.append(tid[0])
        use_policy = True

    # Probe-based routing (fallback or for diagnostics)
    tag_fps = None
    baseline_fp = None
    if not use_policy or args.diagnose or args.sweep_alpha:
        print(f"\nProbing tags: {tags}")
        tag_fps = probe_tags(model, tokenizer, curve_modules, tags, device,
                             args.context_template)
        baseline_fp = probe_baseline(model, tokenizer, curve_modules, device)
        print(f"  Baseline probed")

    # Diagnostics
    if args.diagnose:
        diagnose_fingerprints(tag_fps, baseline_fp)
        if not args.prompt and not args.sweep_alpha:
            return

    # Alpha sweep
    if args.sweep_alpha:
        from datasets import load_dataset
        try:
            raw_val = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                                   split="validation")
        except Exception:
            raw_val = load_dataset("wikitext", "wikitext-2-raw-v1",
                                   split="validation")
        eval_text = "\n".join(t for t in raw_val["text"] if t.strip())[:10000]
        delta = compute_routing_delta(tag_fps, baseline_fp)
        alpha_sweep(model, tokenizer, curve_modules, delta, eval_text, device)
        if not args.prompt:
            return

    # Generation
    if args.prompt is None:
        if not args.diagnose and not args.sweep_alpha:
            print("\nNo --prompt provided. Use --diagnose or --prompt.")
        return

    # Choose delta source: trained policy (needs prompt) or probe-based
    if use_policy:
        # Policy needs prompt hidden state — compute now
        prompt_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            prompt_out = model(prompt_ids, use_cache=False, output_hidden_states=True)
            prompt_state = prompt_out.hidden_states[-1][0, -1, :].float()
            tag_embs = model.model.embed_tokens(
                torch.tensor(_policy_tag_ids, device=device)).float()
            delta, _ = policy(prompt_state, tag_embs, deterministic=True)
        delta_max = max(d.abs().max().item() for d in delta.values())
        print(f"  Policy edits: |Δc| max={delta_max:.4f}")
    else:
        delta = compute_routing_delta(tag_fps, baseline_fp)

    # Export edits for inspector
    if args.export_edits:
        export_edits(model, tokenizer, curve_modules, args.prompt, tags,
                     delta, args.alpha, device, args.export_edits,
                     model_name=args.model)

    if args.schedule:
        # Scheduled routing
        schedule = parse_schedule(args.schedule)
        print(f"\n  Schedule: {schedule}")
        print(f"  Alpha: {args.alpha}")
        output = generate_scheduled(
            model, tokenizer, curve_modules, args.prompt,
            tag_fps, baseline_fp, schedule,
            alpha=args.alpha, max_tokens=args.max_tokens, device=device)
        print(f"\n{'─' * 60}")
        print(f"  Prompt: \"{args.prompt}\"")
        print(f"  Tags (scheduled): {[t for t, _ in schedule]}")
        print(f"{'─' * 60}")
        print(f"\n  SCHEDULED OUTPUT:")
        _print_wrapped(output, args.prompt)
        print(f"\n{'─' * 60}")

    elif args.compare:
        # Side-by-side comparison
        print(f"\n  Generating comparison (alpha={args.alpha})...")

        base_out = generate(model, tokenizer, args.prompt, device,
                            args.max_tokens, args.temperature)

        # Choose routing method: control point (policy) or tau bias (probe)
        if use_policy:
            routing_ctx = apply_ctrl_routing(curve_modules, delta, args.alpha)
        else:
            routing_ctx = apply_static_routing(curve_modules, delta, args.alpha)

        with routing_ctx:
            routed_out = generate(model, tokenizer, args.prompt, device,
                                  args.max_tokens, args.temperature)

        print(f"\n{'─' * 60}")
        print(f"  Prompt: \"{args.prompt}\"")
        print(f"  Tags: {tags}, Alpha: {args.alpha}")
        print(f"{'─' * 60}")

        print(f"\n  BASELINE (no routing):")
        _print_wrapped(base_out, args.prompt)

        print(f"\n  ROUTED (tags={tags}, alpha={args.alpha}):")
        _print_wrapped(routed_out, args.prompt)

        # Quick keyword analysis
        base_completion = base_out[len(args.prompt):].lower()
        routed_completion = routed_out[len(args.prompt):].lower()
        print(f"\n  Tag keyword presence:")
        for tag in tags:
            b = tag.lower() in base_completion
            r = tag.lower() in routed_completion
            marker = " <<<" if r and not b else ""
            print(f"    '{tag}': baseline={'yes' if b else 'no':>3s}, "
                  f"routed={'yes' if r else 'no':>3s}{marker}")

        print(f"\n{'─' * 60}")

    else:
        # Single routed generation
        print(f"\n  Generating with routing (alpha={args.alpha})...")
        if use_policy:
            routing_ctx = apply_ctrl_routing(curve_modules, delta, args.alpha)
        else:
            routing_ctx = apply_static_routing(curve_modules, delta, args.alpha)
        with routing_ctx:
            output = generate(model, tokenizer, args.prompt, device,
                              args.max_tokens, args.temperature)

        print(f"\n{'─' * 60}")
        print(f"  Prompt: \"{args.prompt}\"")
        print(f"  Tags: {tags}, Alpha: {args.alpha}")
        print(f"{'─' * 60}")
        print(f"\n  ROUTED OUTPUT:")
        _print_wrapped(output, args.prompt)
        print(f"\n{'─' * 60}")


def _print_wrapped(text, prompt, width=60):
    """Print text with wrapping, highlighting the completion part."""
    completion = text[len(prompt):]
    words = completion.split()
    line = "    "
    for w in words:
        if len(line) + len(w) + 1 > width:
            print(line)
            line = "    " + w
        else:
            line += (" " if len(line) > 4 else "") + w
    if line.strip():
        print(line)


if __name__ == "__main__":
    main()
