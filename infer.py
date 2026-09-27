#!/usr/bin/env python3
"""
Inference: compare base model vs curve attention model.

Usage:
  # Compare base vs formation model
  python infer.py --prompt "The theory of relativity" \
    --formation checkpoints/formation/formation.pt

  # Also with calibration checkpoint
  python infer.py --prompt "The theory of relativity" \
    --calibration checkpoints/Llama-3.2-1B-Instruct-curve-attn/curve_attention.pt \
    --formation checkpoints/formation/formation.pt

  # Interactive mode
  python infer.py --formation checkpoints/formation/formation.pt
"""

import argparse
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from prototype_curve_attention import CurveAttention


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
    """Load model and patch with curve attention from checkpoint."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation='eager').to(device)
    model.config.use_cache = False
    model.eval()

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = ckpt['config']
    R = config['R']
    hd = config['head_dim']

    for block_idx, state in ckpt['curve_states'].items():
        block_idx = int(block_idx)
        ca = CurveAttention(hd, R).to(device)
        ca.load_state_dict(state)
        ca.eval()
        patch_block(model, block_idx, ca)

    return model


def generate(model, tokenizer, prompt, device, max_tokens=100, temperature=0.0):
    ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    kwargs = dict(max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id,
                  eos_token_id=tokenizer.eos_token_id, use_cache=False)
    if temperature == 0.0:
        kwargs['do_sample'] = False
    else:
        kwargs.update(do_sample=True, temperature=temperature, top_k=50, top_p=0.95)
    with torch.no_grad():
        out = model.generate(ids, **kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def print_comparison(prompt, results):
    width = 70
    print(f"\n{'─' * width}")
    print(f"  Prompt: \"{prompt}\"")
    print(f"{'─' * width}")
    for label, text in results:
        completion = text[len(prompt):]
        print(f"\n  {label}:")
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
    print(f"\n{'─' * width}\n")


def main():
    parser = argparse.ArgumentParser(description="Inference: base vs curve attention")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--formation", default=None, help="Formation checkpoint")
    parser.add_argument("--calibration", default=None, help="Calibration checkpoint (optional)")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation='eager').to(device)
    base_model.config.use_cache = False
    base_model.eval()

    # Load curve models
    cal_model = None
    form_model = None

    if args.calibration:
        print("Loading calibration model...")
        cal_model = load_curve_model(args.model, args.calibration, device)

    if args.formation:
        print("Loading formation model...")
        form_model = load_curve_model(args.model, args.formation, device)

    mode = "greedy" if args.temperature == 0.0 else f"sampling (t={args.temperature})"
    print(f"Mode: {mode}, max_tokens: {args.max_tokens}\n")

    def run_prompt(prompt):
        results = []

        base_out = generate(base_model, tokenizer, prompt, device,
                            args.max_tokens, args.temperature)
        results.append(("BASE MODEL", base_out))

        if cal_model:
            cal_out = generate(cal_model, tokenizer, prompt, device,
                               args.max_tokens, args.temperature)
            results.append(("CALIBRATED", cal_out))

        if form_model:
            form_out = generate(form_model, tokenizer, prompt, device,
                                args.max_tokens, args.temperature)
            results.append(("FORMATION", form_out))

        print_comparison(prompt, results)

    if args.prompt:
        run_prompt(args.prompt)
    else:
        print("Interactive mode. Type a prompt, 'quit' to exit.\n")
        while True:
            try:
                prompt = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.strip().lower() in ('quit', 'exit', 'q'):
                break
            if not prompt.strip():
                continue
            run_prompt(prompt)


if __name__ == "__main__":
    main()
