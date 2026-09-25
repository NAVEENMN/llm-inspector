#!/usr/bin/env python3
"""
Download a HuggingFace model to the local models/ directory.

Usage:
  python download_model.py meta-llama/Llama-3.2-1B-Instruct
  python download_model.py Qwen/Qwen2-1.5B
  python download_model.py gpt2

Models are saved to models/<model-name>/ and can be loaded by the inspector:
  python inspector.py --model models/Llama-3.2-1B-Instruct
"""

import sys
import os


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        sys.exit(0)

    model_id = sys.argv[1]
    # Local folder name: strip org prefix, keep model name
    local_name = model_id.split("/")[-1]
    local_path = os.path.join("models", local_name)

    if os.path.exists(local_path):
        print(f"Model already exists at {local_path}")
        print(f"Use: python inspector.py --model {local_path}")
        sys.exit(0)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Downloading {model_id}...")
    print(f"Saving to {local_path}/")

    # Download tokenizer
    print("  Tokenizer...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(local_path)
    print("done")

    # Download model
    print("  Model...", end=" ", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.save_pretrained(local_path)
    print("done")

    # Print config summary
    cfg = model.config
    nh = cfg.num_attention_heads
    nkv = getattr(cfg, 'num_key_value_heads', nh)
    hd = getattr(cfg, 'head_dim', cfg.hidden_size // nh)
    nl = cfg.num_hidden_layers
    ff = cfg.intermediate_size

    print(f"\nModel: {model_id}")
    print(f"  Layers: {nl}")
    print(f"  Q heads: {nh}, KV heads: {nkv}")
    print(f"  Hidden dim: {cfg.hidden_size}, Head dim: {hd}")
    print(f"  Intermediate: {ff}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nSaved to: {local_path}/")
    print(f"Run: python inspector.py --model {local_path}")


if __name__ == "__main__":
    main()
