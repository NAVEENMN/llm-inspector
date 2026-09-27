#!/usr/bin/env python3
"""
Train a routing policy (differentiable) — τ editing approach.

Policy(prompt_state, tag_embeddings) → Δτ per layer
Applied as bias shift to tau_proj: τ = sigmoid(W_τ·q + b_τ + Δτ)
Changes WHERE queries sample V curves, not the curve content itself.

This is stable during autoregressive generation because τ shifts
don't corrupt information — they redirect attention to different
parts of the same V curve.

Loss = L_anchor + λ * L_route + γ * ||Δτ||²
  L_anchor: CE on first/last quarter (coherence)
  L_route:  -log_prob of tag tokens in middle (steering)
  ||Δτ||²:  keep shifts small

Usage:
  accelerate launch train_router.py \
    --model meta-llama/Llama-3.2-3B \
    --formation checkpoints/Llama-3.2-3B-formation/formation.pt \
    --epochs 20 --lr 5e-4 --lambda-route 0.3 --wandb
"""

import argparse
import math
import time
import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from prototype_curve_attention import CurveAttention
from route import patch_block


# ── Spline Curve Policy ────────────────────────────────────

class SplineCurvePolicy(nn.Module):
    """Policy produces its own B-spline curve per layer.

    The policy's curve is ADDED to the model's blended V curve:
      V_final(t) = V_blended(t) + V_policy(t)

    Adding two B-spline curves = adding their control points.
    The policy outputs [head_dim] control points per layer — these
    define a smooth steering curve that redirects information flow.
    """

    def __init__(self, embed_dim, layer_indices, head_dim, hidden=256):
        super().__init__()
        self.layer_indices = sorted(layer_indices)
        self.head_dim = head_dim

        self.prompt_enc = nn.Linear(embed_dim, hidden)
        self.tag_enc = nn.Linear(embed_dim, hidden)
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Each layer gets its own curve (control points)
        self.curve_layers = nn.ModuleDict({
            str(li): nn.Linear(hidden, head_dim)
            for li in layer_indices
        })
        # Init near zero — no steering at start
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, prompt_state, tag_embeddings):
        h_p = self.prompt_enc(prompt_state)
        h_t = self.tag_enc(tag_embeddings.mean(0))
        h = self.fuse(torch.cat([h_p, h_t], dim=-1))
        curves = {}
        for li_str, layer in self.curve_layers.items():
            curves[int(li_str)] = layer(h)  # [head_dim] control points
        return curves


# ── Curve Addition Hooks ───────────────────────────────────

def install_curve_hooks(curve_modules, policy_curves):
    """Patch eval_curve to add policy's curve to blended V.

    blended coeffs [B, H, S, head_dim] + policy_curve [head_dim]
    = sum of two B-spline curves (adding control points).

    Differentiable — gradients flow through the addition.
    """
    patches = []
    for li, curve in policy_curves.items():
        if li not in curve_modules:
            continue
        ca = curve_modules[li]
        orig_eval = ca.eval_curve

        def make_patched(orig, c):
            def patched(coeffs, tau_01):
                # coeffs: [B, H, S, head_dim]
                # c: [head_dim] — broadcast over all dims
                return orig(coeffs + c, tau_01)
            return patched

        ca.eval_curve = make_patched(orig_eval, curve)
        patches.append((ca, orig_eval))
    return patches


def restore_hooks(patches):
    for ca, orig in patches:
        ca.eval_curve = orig


# ── Tag Vocabulary ─────────────────────────────────────────

def build_tag_vocab(tokenizer, text, min_len=4, min_freq=10,
                    max_freq=2000, max_vocab=2000):
    from collections import Counter
    tokens = tokenizer.encode(text[:2_000_000], add_special_tokens=False)
    counts = Counter(tokens)
    vocab, seen = [], set()
    for tid, freq in counts.most_common():
        if freq < min_freq or freq > max_freq:
            continue
        word = tokenizer.decode([tid]).strip()
        if len(word) < min_len or not word.isalpha():
            continue
        if word.lower() in seen:
            continue
        seen.add(word.lower())
        vocab.append((tid, word, freq))
        if len(vocab) >= max_vocab:
            break
    return vocab


# ── Model Loading ──────────────────────────────────────────

def load_curve_model(model_name, checkpoint_path, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation='eager').to(device)
    model.config.use_cache = False
    model.eval()
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = ckpt['config']
    curve_modules = {}
    for bi, state in ckpt['curve_states'].items():
        bi = int(bi)
        ca = CurveAttention(config['head_dim'], config['R']).to(device)
        ca.load_state_dict(state)
        ca.eval()
        patch_block(model, bi, ca)
        curve_modules[bi] = ca
    return model, curve_modules, config


def eval_ppl(model, eval_ids, device):
    model.eval()
    with torch.no_grad():
        loss = model(eval_ids.to(device), labels=eval_ids.to(device),
                     use_cache=False).loss.float().item()
    return math.exp(loss)


# ── Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train τ routing policy")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--formation", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda-route", type=float, default=0.3)
    parser.add_argument("--lambda-reg", type=float, default=1.0,
                        help="L2 reg on τ shifts")
    parser.add_argument("--n-tags", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=5_000_000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()

    try:
        from accelerate import Accelerator
        accelerator = Accelerator()
        device = accelerator.device
        is_main = accelerator.is_main_process
        use_accel = True
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True
        use_accel = False

    if is_main:
        if args.wandb:
            import wandb
            wandb.init(project="spline-routing",
                       name=f"tau-{args.model.split('/')[-1]}",
                       config=vars(args),
                       tags=["tau-routing", "differentiable"])
        print(f"Train τ Routing Policy")
        print(f"  Model: {args.model}")
        print(f"  Device: {device}")
        if use_accel:
            print(f"  GPUs: {accelerator.num_processes}")

    from transformers import AutoTokenizer
    model, curve_modules, ckpt_config = load_curve_model(args.model, args.formation, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    R = ckpt_config['R']
    embed_dim = model.config.hidden_size
    layer_indices = sorted(curve_modules.keys())

    for p in model.parameters():
        p.requires_grad = False

    # Data
    from datasets import load_dataset
    wiki = "wikitext-2-raw-v1" if args.fast else "wikitext-103-raw-v1"
    if is_main:
        print(f"  Loading {wiki}...")
    try:
        raw_train = load_dataset("Salesforce/wikitext", wiki, split="train")
    except Exception:
        raw_train = load_dataset("wikitext", wiki, split="train")
    train_text = "\n".join(t for t in raw_train["text"] if t.strip())
    max_tok = min(args.max_tokens, 500_000 if args.fast else args.max_tokens)
    train_ids = tokenizer.encode(train_text[:max_tok], add_special_tokens=False)
    sequences = []
    for i in range(0, len(train_ids) - args.block_size, args.block_size):
        sequences.append(train_ids[i:i + args.block_size])
    sequences = torch.tensor(sequences)

    try:
        raw_val = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    except Exception:
        raw_val = load_dataset("wikitext", "wikitext-2-raw-v1")
    val_text = "\n".join(t for t in raw_val["validation"]["text"] if t.strip())
    eval_ids = torch.tensor(
        tokenizer.encode(val_text, add_special_tokens=False)[:512]).unsqueeze(0)

    if is_main:
        print("  Building tag vocabulary...")
    tag_vocab = build_tag_vocab(tokenizer, train_text)
    tag_token_ids = [tid for tid, _, _ in tag_vocab]
    tag_words = [w for _, w, _ in tag_vocab]
    if is_main:
        print(f"  {len(sequences):,} sequences, {len(tag_vocab)} tags, R={R}")

    # Policy — much smaller than control point version
    # Only R values per layer instead of n_heads × head_dim
    policy = SplineCurvePolicy(embed_dim, layer_indices, ckpt_config['head_dim'],
                               args.hidden_dim).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    head_dim = ckpt_config['head_dim']
    if is_main:
        print(f"  Policy: {n_params:,} params (spline curves, {head_dim} ctrl pts per layer)")

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if use_accel:
        dataset = torch.utils.data.TensorDataset(sequences)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
    else:
        dataloader = None

    base_ppl = eval_ppl(model, eval_ids, device)
    if is_main:
        print(f"  Baseline PPL: {base_ppl:.2f}")

    model_name = args.model.split('/')[-1]
    out_dir = os.path.join(args.output_dir, f"{model_name}-curve-router")
    if is_main:
        os.makedirs(out_dir, exist_ok=True)

    best_score = float('-inf')

    if is_main:
        print(f"\n{'=' * 60}")
        print(f"  Spline Curve Addition: L = L_anchor + {args.lambda_route}*L_route + {args.lambda_reg}*||curve||²")
        print(f"  Policy produces curves added to blended V")
        print(f"{'=' * 60}\n")

    for epoch in range(args.epochs):
        policy.train()
        total_anchor = total_route = total_reg = total_loss = 0
        steps = 0
        start_time = time.time()

        if use_accel:
            data_iter = dataloader
        else:
            perm = torch.randperm(len(sequences))
            data_iter = [(sequences[perm[i:i+args.batch_size]],)
                         for i in range(0, len(sequences), args.batch_size)]

        for (batch,) in data_iter:
            batch = batch.to(device)
            if len(batch) == 0:
                continue

            k = random.randint(3, args.n_tags)
            sampled_idx = random.sample(range(len(tag_vocab)), k)
            sampled_tag_ids = [tag_token_ids[j] for j in sampled_idx]

            with torch.no_grad():
                prompt_out = model(batch[:1, :8], output_hidden_states=True, use_cache=False)
                prompt_state = prompt_out.hidden_states[-1][0, -1, :].float()
                tag_embs = model.model.embed_tokens(
                    torch.tensor(sampled_tag_ids, device=device)).float()

            # Policy → steering curves per layer
            curves = policy(prompt_state, tag_embs)

            # Add policy curves to blended V (differentiable)
            patches = install_curve_hooks(curve_modules, curves)

            # Forward
            output = model(input_ids=batch, labels=batch, use_cache=False)
            logits = output.logits.float()

            # Restore
            restore_hooks(patches)

            # Losses
            shift_logits = logits[:, :-1, :]
            shift_labels = batch[:, 1:]
            S = shift_logits.shape[1]
            s, e = S // 4, 3 * S // 4

            # L_anchor: CE on start + end
            anc_logits = torch.cat([shift_logits[:, :s, :], shift_logits[:, e:, :]], 1)
            anc_labels = torch.cat([shift_labels[:, :s], shift_labels[:, e:]], 1)
            L_anchor = F.cross_entropy(
                anc_logits.reshape(-1, anc_logits.shape[-1]),
                anc_labels.reshape(-1))

            # L_route: tag probability in middle
            mid_lp = F.log_softmax(shift_logits[:, s:e, :], dim=-1)
            tag_t = torch.tensor(sampled_tag_ids, device=device)
            L_route = -mid_lp[:, :, tag_t].mean()

            # L_reg: keep policy curves small
            L_reg = sum(c.pow(2).mean() for c in curves.values()) / len(curves)

            loss = L_anchor + args.lambda_route * L_route + args.lambda_reg * L_reg

            optimizer.zero_grad()
            if use_accel:
                accelerator.backward(loss)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_anchor += L_anchor.item()
            total_route += L_route.item()
            total_reg += L_reg.item()
            total_loss += loss.item()
            steps += 1

            if steps % args.log_interval == 0 and is_main:
                elapsed = time.time() - start_time
                c_max = max(c.abs().max().item() for c in curves.values())
                c_mean = sum(c.abs().mean().item() for c in curves.values()) / len(curves)
                tags_str = [tag_words[j] for j in sampled_idx[:3]]
                print(f"  Ep {epoch+1} | Step {steps:>5d} | "
                      f"L={total_loss/steps:.4f} "
                      f"(anc={total_anchor/steps:.4f} "
                      f"rte={total_route/steps:.4f} "
                      f"reg={total_reg/steps:.6f}) | "
                      f"|curve|={c_max:.4f} (mean={c_mean:.4f}) | "
                      f"{tags_str} | {elapsed:.0f}s")
                if args.wandb:
                    import wandb
                    wandb.log({
                        "loss/total": total_loss / steps,
                        "loss/anchor": total_anchor / steps,
                        "loss/route": total_route / steps,
                        "loss/reg": total_reg / steps,
                        "curve/max": c_max,
                        "curve/mean": c_mean,
                    }, step=epoch * 10000 + steps)

        scheduler.step()
        if use_accel:
            accelerator.wait_for_everyone()

        # Eval
        if is_main:
            unwrapped = accelerator.unwrap_model(policy) if use_accel else policy
            unwrapped.eval()
            clean_ppl = eval_ppl(model, eval_ids, device)

            # Rotate through different test prompts and tag sets each epoch
            test_prompts = [
                "The theory of general relativity describes",
                "The history of ancient Rome",
                "In computer science, algorithms",
                "The process of photosynthesis",
                "The principles of economics",
            ]
            test_tag_sets = [
                ["gravity", "Newton", "speed", "light", "force"],
                ["empire", "soldiers", "conquest", "power", "Senate"],
                ["memory", "binary", "sorting", "graph", "search"],
                ["sunlight", "carbon", "oxygen", "chlorophyll", "energy"],
                ["market", "trade", "inflation", "demand", "labor"],
            ]
            test_idx = epoch % len(test_prompts)
            test_prompt = test_prompts[test_idx]
            test_tags = test_tag_sets[test_idx]
            test_tag_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in test_tags]
            test_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                tp_out = model(test_ids, output_hidden_states=True, use_cache=False)
                tp_state = tp_out.hidden_states[-1][0, -1, :].float()
                tp_tags = model.model.embed_tokens(
                    torch.tensor(test_tag_ids, device=device)).float()
                test_curves = unwrapped(tp_state, tp_tags)

                # Baseline
                base_gen = model.generate(test_ids, max_new_tokens=60, do_sample=False,
                                          use_cache=False, pad_token_id=tokenizer.eos_token_id)
                base_text = tokenizer.decode(base_gen[0], skip_special_tokens=True)

                # Routed (add policy curves)
                patches = install_curve_hooks(curve_modules, test_curves)
                routed_gen = model.generate(test_ids, max_new_tokens=60, do_sample=False,
                                            use_cache=False, pad_token_id=tokenizer.eos_token_id)
                routed_text = tokenizer.decode(routed_gen[0], skip_special_tokens=True)
                restore_hooks(patches)

            dt_max = max(c.abs().max().item() for c in test_curves.values())
            hits = [t for t in test_tags if t.lower() in routed_text.lower()]
            elapsed = time.time() - start_time

            print(f"\n  Epoch {epoch+1} done | L={total_loss/steps:.4f} | "
                  f"PPL={clean_ppl:.2f} | |Δτ|={dt_max:.4f} | {elapsed:.0f}s")
            print(f"  BASE:   {base_text[:130]}")
            print(f"  ROUTED: {routed_text[:130]}")
            print(f"  Tag hits: {hits} ({len(hits)}/{len(test_tags)})\n")

            if args.wandb:
                import wandb
                wandb.log({
                    "eval/ppl": clean_ppl,
                    "eval/curve_max": dt_max,
                    "eval/tag_hits": len(hits),
                    "eval/test_prompt": test_prompt,
                    "eval/test_tags": str(test_tags),
                    "eval/comparison": wandb.Html(
                        f"<b>Prompt:</b> {test_prompt}<br>"
                        f"<b>Tags:</b> {test_tags}<br>"
                        f"<b>BASE:</b> {base_text[:300]}<br>"
                        f"<b>ROUTED:</b> {routed_text[:300]}<br>"
                        f"<b>Hits:</b> {hits}"
                    ),
                }, step=epoch * 10000 + steps)

            score = len(hits) - 0.1 * max(0, clean_ppl - base_ppl * 1.5)
            if score > best_score or epoch == 0:
                best_score = score
                save_path = os.path.join(out_dir, "routing_policy.pt")
                torch.save({
                    'policy_state': unwrapped.state_dict(),
                    'policy_type': 'spline_curve',
                    'config': {
                        'embed_dim': embed_dim,
                        'layer_indices': layer_indices,
                        'head_dim': head_dim,
                        'hidden_dim': args.hidden_dim,
                        'model': args.model,
                    },
                    'metrics': {
                        'base_ppl': base_ppl,
                        'clean_ppl': clean_ppl,
                        'tag_hits': len(hits),
                        'epoch': epoch + 1,
                    },
                }, save_path)
                print(f"  Saved (hits={len(hits)}, ppl={clean_ppl:.2f})")

        if use_accel:
            accelerator.wait_for_everyone()

    if is_main:
        print(f"\n{'=' * 60}")
        print(f"  COMPLETE | {n_params:,} params | Best score: {best_score:.2f}")
        print(f"{'=' * 60}")
        if args.wandb:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
