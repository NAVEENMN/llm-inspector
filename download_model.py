#!/usr/bin/env python3
"""
Download a HuggingFace model and optional manifold checkpoint.

Usage:
  python download_model.py meta-llama/Llama-3.2-1B-Instruct
  python download_model.py nmysore/manifold-1.0-1b
  python download_model.py gpt2

When downloading a manifold model (nmysore/manifold-*), this also
downloads the base model and the manifold checkpoint.
"""

import sys
import os


MANIFOLD_MODELS = {
    "nmysore/manifold-1.0-1b": {
        "base": "meta-llama/Llama-3.2-1B-Instruct",
        "checkpoint": "formation.pt",
    },
}


def download_base_model(model_id, local_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"  Tokenizer...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(local_path)
    print("done")

    print("  Model...", end=" ", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.save_pretrained(local_path)
    print("done")

    cfg = model.config
    nh = cfg.num_attention_heads
    nkv = getattr(cfg, 'num_key_value_heads', nh)
    hd = getattr(cfg, 'head_dim', cfg.hidden_size // nh)
    nl = cfg.num_hidden_layers

    print(f"\n  Layers: {nl}, Q heads: {nh}, KV heads: {nkv}")
    print(f"  Hidden: {cfg.hidden_size}, Head dim: {hd}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        sys.exit(0)

    model_id = sys.argv[1]

    # Check if this is a manifold model
    if model_id in MANIFOLD_MODELS:
        info = MANIFOLD_MODELS[model_id]
        base_id = info["base"]
        base_name = base_id.split("/")[-1]
        base_path = os.path.join("models", base_name)
        ckpt_dir = "checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)

        print(f"Manifold model: {model_id}")
        print(f"  Base: {base_id}")

        # Download base model
        if os.path.exists(base_path):
            print(f"\n  Base model exists: {base_path}")
        else:
            print(f"\nDownloading base model: {base_id}")
            os.makedirs(base_path, exist_ok=True)
            download_base_model(base_id, base_path)

        # Download manifold checkpoint
        ckpt_file = info["checkpoint"]
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        if os.path.exists(ckpt_path):
            print(f"\n  Checkpoint exists: {ckpt_path}")
        else:
            print(f"\nDownloading manifold checkpoint...")
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=model_id,
                filename=ckpt_file,
                local_dir=ckpt_dir,
            )
            print(f"  Saved: {ckpt_path}")

        print(f"\nReady! Run:")
        print(f"  python inspector.py --model {base_path} --checkpoint {ckpt_path}")

    else:
        # Standard model download
        local_name = model_id.split("/")[-1]
        local_path = os.path.join("models", local_name)

        if os.path.exists(local_path):
            print(f"Model already exists at {local_path}")
            print(f"Use: python inspector.py --model {local_path}")
            sys.exit(0)

        print(f"Downloading {model_id}...")
        os.makedirs(local_path, exist_ok=True)
        download_base_model(model_id, local_path)

        print(f"\nSaved to: {local_path}/")
        print(f"Run: python inspector.py --model {local_path}")


if __name__ == "__main__":
    main()
