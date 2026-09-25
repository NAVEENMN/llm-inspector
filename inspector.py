#!/usr/bin/env python3
"""
Inspector: Isometric grid visualization of a transformer block.

Controls:
  W/S        — zoom in/out
  A/D        — pan left/right
  Z/X        — pan up/down
  Q/E        — rotate Y
  R/F        — rotate X
  Tab/`      — cycle selection
  Click      — select component / toggle group
  ESC        — quit (or unfocus text input)
"""

import sys
import os
import math
import json
import random
import threading
import queue
import re
from datetime import datetime
from dataclasses import dataclass, field

try:
    import pygame
except ImportError:
    print("pip install pygame")
    sys.exit(1)

SUB = '₀₁₂₃₄₅₆₇₈₉'

def subscript(n):
    return ''.join(SUB[int(d)] for d in str(n))


# ── Data Model ──────────────────────────────────────────────

@dataclass
class Component:
    name: str
    gx: int
    gy: int
    gw: int
    gh: int
    color: tuple
    dims: str
    typ: str          # "weight", "op", "data", "skip"
    gz: int = 0
    module: str = ""  # "attention", "transform", or ""

    @property
    def is_skip(self):
        return self.typ == "skip"

    @property
    def is_head(self):
        return self.name[:2] in ("Q_", "K_", "V_") and len(self.name) > 2

    @property
    def head_group(self):
        if self.is_head:
            return self.name[0]
        return None


@dataclass
class Connection:
    src: int
    dst: int
    skip: bool = False


# ── Block Builder ───────────────────────────────────────────

class BlockBuilder:
    """Builds component graph for a standard transformer block."""

    def __init__(self):
        self.components: list[Component] = []
        self.connections: list[Connection] = []
        self.idx: dict[str, int] = {}

    def add(self, **kw) -> int:
        i = len(self.components)
        c = Component(**kw)
        self.idx[c.name] = i
        self.components.append(c)
        return i

    def connect(self, src_name, dst_name, skip=False):
        self.connections.append(Connection(self.idx[src_name], self.idx[dst_name], skip))

    def add_heads(self, prefix, count, gy, start_gx, base_color, gz=0):
        names = []
        for h in range(count):
            name = f"{prefix}_{subscript(h)}"
            r, g, b = base_color
            color = (r + h * 2, g + h, b) if count > 10 else (r, g + h * 5, b)
            self.add(name=name, gx=start_gx + h, gy=gy, gw=1, gh=1,
                     color=color, dims="R^{2048->64}", typ="weight",
                     gz=gz, module="attention")
            names.append(name)
        return names

    def add_residual_path(self, tag, src_name, dst_name, gx, gy_top, gy_mid, gy_bot, gz):
        for suffix, gy, desc in [(".out", gy_top, "residual copy"),
                                  (".mid", gy_mid, "↓"),
                                  (".in",  gy_bot, f"-> {dst_name}")]:
            self.add(name=f"{tag}{suffix}", gx=gx, gy=gy, gw=1, gh=1,
                     color=(160, 140, 40), dims=desc, typ="skip", gz=gz)
        self.connect(src_name, f"{tag}.out", skip=True)
        self.connect(f"{tag}.out", f"{tag}.mid", skip=True)
        self.connect(f"{tag}.mid", f"{tag}.in", skip=True)
        self.connect(f"{tag}.in", dst_name, skip=True)

    def tag_module(self, names, module):
        for n in names:
            if n in self.idx:
                self.components[self.idx[n]].module = module

    def build_standard_block(self):
        self.add(name="Input", gx=0, gy=20, gw=5, gh=1,
                 color=(80,80,110), dims="x in R^2048", typ="data")
        self.add(name="RMSNorm₁", gx=0, gy=18, gw=4, gh=1,
                 color=(100,100,140), dims="R^2048 -> R^2048", typ="op", module="attention")
        self.connect("Input", "RMSNorm₁")

        q_names = self.add_heads("Q", 32, gy=16, start_gx=-16, base_color=(40, 80, 210))
        k_names = self.add_heads("K", 8,  gy=14, start_gx=-4,  base_color=(40, 120, 210), gz=-60)
        v_names = self.add_heads("V", 8,  gy=14, start_gx=-4,  base_color=(40, 150, 210), gz=60)
        for n in q_names + k_names + v_names:
            self.connect("RMSNorm₁", n)

        self.add(name="RoPE", gx=-8, gy=12, gw=3, gh=1,
                 color=(130,110,50), dims="rotation on Q,K", typ="op", module="attention")
        for n in q_names + k_names:
            self.connect(n, "RoPE")

        for name, gx, gy, gw, gh, color, dims in [
            ("QKt/d",   -4, 10, 3, 1, (40,150,80), "R^{32 x seq x seq}"),
            ("Softmax", -4, 9,  3, 1, (40,170,80), "[0,1]^{32 x seq x seq}"),
            ("wV",       2, 9,  3, 1, (40,190,80), "32 x R^{seq x 64}"),
        ]:
            self.add(name=name, gx=gx, gy=gy, gw=gw, gh=gh,
                     color=color, dims=dims, typ="op", module="attention")

        self.connect("RoPE", "QKt/d")
        self.connect("QKt/d", "Softmax")
        self.connect("Softmax", "wV")
        for n in v_names:
            self.connect(n, "wV")

        self.add(name="W_o", gx=0, gy=7, gw=4, gh=2,
                 color=(50,90,210), dims="R^{2048x2048} (32h)", typ="weight", module="attention")
        self.add(name="+ res₁", gx=0, gy=5, gw=2, gh=1,
                 color=(190,160,50), dims="x + attn(x)", typ="op", module="attention")
        self.connect("wV", "W_o")
        self.connect("W_o", "+ res₁")

        self.add_residual_path("res₁", "Input", "+ res₁", gx=8,
                               gy_top=20, gy_mid=12, gy_bot=5, gz=40)

        for name, gx, gy, gw, gh, color, dims, typ in [
            ("RMSNorm₂", 0,  3,  4, 1, (100,100,140), "R^2048 -> R^2048", "op"),
            ("W_gate",   -5,  1,  4, 2, (210,100,40),  "R^{2048x8192}",     "weight"),
            ("W_up",      5,  1,  4, 2, (210,120,40),  "R^{2048x8192}",     "weight"),
            ("SiLU",     -5, -1,  3, 1, (210,60,50),   "s(x)*x",            "op"),
            ("gate*up",   0, -2,  3, 1, (210,80,60),   "R^8192 . R^8192",   "op"),
            ("W_down",    0, -4,  4, 2, (210,100,40),  "R^{8192x2048}",     "weight"),
            ("+ res₂",   0, -6,  2, 1, (190,160,50),  "x + mlp(x)",        "op"),
        ]:
            self.add(name=name, gx=gx, gy=gy, gw=gw, gh=gh,
                     color=color, dims=dims, typ=typ, module="transform")

        self.add(name="Output", gx=0, gy=-8, gw=5, gh=1,
                 color=(80,80,110), dims="R^2048", typ="data")

        for src, dst in [("+ res₁", "RMSNorm₂"), ("RMSNorm₂", "W_gate"),
                         ("RMSNorm₂", "W_up"), ("W_gate", "SiLU"),
                         ("SiLU", "gate*up"), ("W_up", "gate*up"),
                         ("gate*up", "W_down"), ("W_down", "+ res₂"),
                         ("+ res₂", "Output")]:
            self.connect(src, dst)

        self.add_residual_path("res₂", "+ res₁", "+ res₂", gx=8,
                               gy_top=5, gy_mid=-1, gy_bot=-6, gz=60)
        self.tag_module(["res₁.out", "res₁.mid", "res₁.in"], "attention")
        self.tag_module(["res₂.out", "res₂.mid", "res₂.in"], "transform")

        return self.components, self.connections, self.idx


# ── Simulation ──────────────────────────────────────────────

class Simulation:
    """Dummy simulation of token data flowing through one transformer block."""

    VOCAB_DUMMY = {
        "the": 278, "cat": 4937, "sat": 3523, "on": 373, "a": 264,
        "mat": 1986, "hello": 9906, "world": 1917, "theory": 10334,
        "of": 315, "relativity": 6370, "is": 338, "beautiful": 5765,
    }

    # Dummy vocabulary for generation
    VOCAB_OUT = ["the", "a", "is", "of", "and", "in", "to", "was", "that", "it",
                 "for", "with", "on", "as", "at", "by", "from", "an", "be", "this",
                 "which", "or", "have", "had", "not", "but", "what", "all", "were", "when",
                 "one", "can", "there", "are", "their", "has", "more", "will", "been", "would",
                 "who", "its", "said", "each", "make", "like", "time", "very", "most", "also"]

    def __init__(self):
        self.text = ""
        self.tokens = []
        self.token_ids = []
        self.seq_len = 0
        self.active = False
        self.data = {}  # comp_name → list of (section_title, [lines])
        self.generated_tokens = []  # tokens produced by stepping/run_all
        self.gen_step = 0  # how many tokens generated so far
        self.attn_mats = {}  # head_idx → matrix (list of lists)
        self.selected_head = 0
        self._attn_stats = {}

    def run(self, text):
        self.text = text.strip()
        if not self.text:
            self.active = False
            self.data = {}
            return

        self.tokens = self.text.split()
        self.seq_len = len(self.tokens)
        self.token_ids = [self.VOCAB_DUMMY.get(t.lower(), hash(t) % 32000) for t in self.tokens]
        self.active = True

        random.seed(abs(hash(self.text)) % (2**31))
        self._generate_all()

    def _rv(self, n=6, scale=0.02):
        """Random vector string."""
        vals = [random.gauss(0, scale) for _ in range(n)]
        inner = ", ".join(f"{v:+.4f}" for v in vals)
        return f"[{inner}, ...]"

    def _attn_row(self, n):
        """Generate one row of attention weights (sums to ~1)."""
        raw = [random.random() for _ in range(n)]
        s = sum(raw)
        return [r / s for r in raw]

    def _generate_all(self):
        S = self.seq_len
        tok_str = ", ".join(f'"{t}"' for t in self.tokens[:6])
        if S > 6:
            tok_str += ", ..."
        id_str = ", ".join(str(i) for i in self.token_ids[:6])
        if S > 6:
            id_str += ", ..."

        d = {}

        # Input
        d["Input"] = [
            ("Tokens", [
                f"Text: \"{self.text}\"",
                f"Tokens: [{tok_str}]",
                f"IDs: [{id_str}]",
                f"Count: {S}",
            ]),
            ("Embedding Lookup", [
                f"Shape: (1, {S}, 2048)",
                f"x[0] = {self._rv(6, 0.02)}",
                f"x[1] = {self._rv(6, 0.02)}",
                f"  (each token -> 2048-dim vector)",
            ]),
            ("Note", [
                "Embeddings are learned during",
                "pretraining. Each of 128k vocab",
                "tokens has a fixed 2048-d vector.",
            ]),
        ]

        # RMSNorm₁
        rms_vals = [round(abs(random.gauss(1.0, 0.15)), 4) for _ in range(min(S, 4))]
        d["RMSNorm₁"] = [
            ("Operation", [
                "x̂ = x / RMS(x) · γ",
                "Normalizes per-token, no centering.",
                f"γ shape: (2048,) — learned scale",
            ]),
            ("Activations", [
                f"Input shape:  (1, {S}, 2048)",
                f"Output shape: (1, {S}, 2048)",
                f"RMS per token: {rms_vals}",
                f"After norm: {self._rv(6, 1.0)}",
            ]),
        ]

        # Q heads
        for h in range(32):
            name = f"Q_{subscript(h)}"
            d[name] = self._head_info("Q", h, 32, S)

        # K heads
        for h in range(8):
            name = f"K_{subscript(h)}"
            d[name] = self._head_info("K", h, 8, S)

        # V heads
        for h in range(8):
            name = f"V_{subscript(h)}"
            d[name] = self._head_info("V", h, 8, S)

        # RoPE
        d["RoPE"] = [
            ("Operation", [
                "Rotary Position Embedding",
                "Encodes position info into Q, K",
                "by rotating pairs of dimensions.",
            ]),
            ("Activations", [
                f"Q shape: (1, 32, {S}, 64)",
                f"K shape: (1,  8, {S}, 64)",
                "Rotation angles: θ_i = 10000^(-2i/d)",
                f"Position 0: cos=1.000, sin=0.000",
                f"Position 1: cos={math.cos(1/100):.3f}, sin={math.sin(1/100):.3f}",
            ]),
            ("Note", [
                "After RoPE, relative position info",
                "is encoded in Q·K^T dot products.",
                "Weights: NONE (deterministic).",
            ]),
        ]

        # Generate attention matrices for all 32 heads
        self.attn_mats = {}
        for h in range(32):
            self.attn_mats[h] = self._gen_attn_matrix(S)
        self._attn_stats = {"max": 0.95, "min_nonzero": 0.001}

        # QK^T/√d
        score_mat = self._gen_score_matrix(S)
        d["QKt/d"] = [
            ("Operation", [
                "scores = Q * Kt / sqrt(64)",
                "sqrt(64) = 8.0 (scaling factor)",
            ]),
            ("Activations", [
                f"Q: (1, 32, {S}, 64)",
                f"K: (1,  8, {S}, 64)  (GQA: repeat 4x)",
                f"Scores: (1, 32, {S}, {S})",
                f"+ causal mask: upper triangle = -inf",
            ]),
            ("Score Heatmap (head 0)", [], {
                "matrix": score_mat, "tokens": self.tokens,
                "label": "Raw scores (pre-softmax)", "cmap": "diverge"}),
        ]

        # Softmax and wV use sentinels — built dynamically in get_display
        d["Softmax"] = "ATTN_HEADS"
        d["wV"] = "ATTN_WV"

        # W_o
        d["W_o"] = [
            ("Weight Matrix (FIXED)", [
                "Shape: (2048, 2048)",
                "Concatenates 32 heads -> project back",
                f"Params: 4,194,304",
                f"Mean: {random.gauss(0, 0.001):.6f}",
                f"Std:  {random.uniform(0.008, 0.015):.6f}",
            ]),
            ("Activations", [
                f"Input:  (1, {S}, 2048)  [32x64 concat]",
                f"Output: (1, {S}, 2048)",
                f"out[0] = {self._rv(6, 0.1)}",
            ]),
        ]

        # + res₁
        d["+ res₁"] = [
            ("Operation", [
                "x' = x + attn(x)",
                "Residual connection preserves",
                "original signal + attention output.",
            ]),
            ("Activations", [
                f"Residual: (1, {S}, 2048)  from Input",
                f"Attn out: (1, {S}, 2048)  from W_o",
                f"Sum:      (1, {S}, 2048)",
                f"out[0] = {self._rv(6, 0.1)}",
            ]),
        ]

        # RMSNorm₂
        rms2 = [round(abs(random.gauss(1.0, 0.15)), 4) for _ in range(min(S, 4))]
        d["RMSNorm₂"] = [
            ("Operation", [
                "x̂ = x / RMS(x) · γ",
                "Same as RMSNorm₁, separate γ.",
            ]),
            ("Activations", [
                f"Input:  (1, {S}, 2048)",
                f"Output: (1, {S}, 2048)",
                f"RMS per token: {rms2}",
            ]),
        ]

        # W_gate
        d["W_gate"] = [
            ("Weight Matrix (FIXED)", [
                "Shape: (2048, 8192)",
                f"Params: 16,777,216",
                f"Mean: {random.gauss(0, 0.001):.6f}",
                f"Std:  {random.uniform(0.008, 0.015):.6f}",
            ]),
            ("Activations", [
                f"Input:  (1, {S}, 2048)",
                f"Output: (1, {S}, 8192)",
                f"gate[0] = {self._rv(6, 0.5)}",
                "-> goes to SiLU activation",
            ]),
        ]

        # W_up
        d["W_up"] = [
            ("Weight Matrix (FIXED)", [
                "Shape: (2048, 8192)",
                f"Params: 16,777,216",
                f"Mean: {random.gauss(0, 0.001):.6f}",
                f"Std:  {random.uniform(0.008, 0.015):.6f}",
            ]),
            ("Activations", [
                f"Input:  (1, {S}, 2048)",
                f"Output: (1, {S}, 8192)",
                f"up[0] = {self._rv(6, 0.5)}",
                "-> element-wise multiply with gate",
            ]),
        ]

        # SiLU
        d["SiLU"] = [
            ("Operation", [
                "SiLU(x) = x * sigmoid(x)",
                "Smooth ReLU variant (Swish).",
                "Also called Swish activation.",
            ]),
            ("Activations", [
                f"Input:  (1, {S}, 8192)  from W_gate",
                f"Output: (1, {S}, 8192)",
                f"silu[0] = {self._rv(6, 0.3)}",
            ]),
        ]

        # gate×up
        d["gate*up"] = [
            ("Operation", [
                "h = SiLU(gate) . up",
                "Element-wise product (SwiGLU).",
                "This is the gating mechanism.",
            ]),
            ("Activations", [
                f"Gate: (1, {S}, 8192)  after SiLU",
                f"Up:   (1, {S}, 8192)  from W_up",
                f"Out:  (1, {S}, 8192)",
                f"h[0] = {self._rv(6, 0.2)}",
            ]),
        ]

        # W_down
        d["W_down"] = [
            ("Weight Matrix (FIXED)", [
                "Shape: (8192, 2048)",
                f"Params: 16,777,216",
                f"Mean: {random.gauss(0, 0.001):.6f}",
                f"Std:  {random.uniform(0.008, 0.015):.6f}",
            ]),
            ("Activations", [
                f"Input:  (1, {S}, 8192)",
                f"Output: (1, {S}, 2048)",
                f"down[0] = {self._rv(6, 0.05)}",
            ]),
        ]

        # + res₂
        d["+ res₂"] = [
            ("Operation", [
                "x'' = x' + mlp(x')",
                "Second residual connection.",
            ]),
            ("Activations", [
                f"Residual: (1, {S}, 2048)  from + res₁",
                f"MLP out:  (1, {S}, 2048)  from W_down",
                f"Sum:      (1, {S}, 2048)",
                f"out[0] = {self._rv(6, 0.1)}",
            ]),
        ]

        # Output
        d["Output"] = [
            ("Block Output", [
                f"Shape: (1, {S}, 2048)",
                f"out[0] = {self._rv(6, 0.1)}",
                f"out[1] = {self._rv(6, 0.1)}",
            ]),
            ("Data Flow Summary", [
                f"Input tokens: {S}",
                "This output feeds into the next",
                "transformer block (or final head).",
                "",
                "Total block params: ~54M",
                "  Attention: ~4x2048x64x(32+8+8)",
                "  + W_o: 2048x2048",
                "  MLP: 3x2048x8192",
            ]),
        ]

        self.data = d

    def _head_info(self, prefix, h, total, S):
        is_gqa = prefix in ("K", "V")
        serves = f"  (GQA: serves 4 Q heads)" if is_gqa else ""
        return [
            (f"Weight Matrix (FIXED)", [
                f"Shape: (2048, 64)",
                f"Params: 131,072",
                f"Mean: {random.gauss(0, 0.001):.6f}",
                f"Std:  {random.uniform(0.008, 0.015):.6f}",
                f"Head {h} of {total}{serves}",
            ]),
            ("Activations (change per input)", [
                f"Input:  (1, {S}, 2048)",
                f"Output: (1, {S}, 64)",
                f"{prefix.lower()}[0] = {self._rv(6, 0.3)}",
                f"{prefix.lower()}[1] = {self._rv(6, 0.3)}",
            ]),
        ]

    def _gen_score_matrix(self, S):
        """Generate S×S raw score matrix (causal: upper triangle is None)."""
        mat = []
        for i in range(S):
            row = []
            for j in range(S):
                if j > i:
                    row.append(None)  # masked
                else:
                    row.append(random.gauss(0, 2.0))
            mat.append(row)
        return mat

    def _gen_attn_matrix(self, S):
        """Generate S×S attention weight matrix (causal, rows sum to 1)."""
        mat = []
        for i in range(S):
            raw = [random.random() * (1.5 if j == i else 1.0) for j in range(i + 1)]
            s = sum(raw)
            row = [r / s for r in raw] + [0.0] * (S - i - 1)
            mat.append(row)
        return mat

    def step(self, text, max_tokens):
        """Generate one next token. Returns the new token or None if done."""
        if self.gen_step >= max_tokens:
            return None
        # Seed based on input + step for deterministic but varying output
        random.seed(abs(hash(text + str(self.gen_step))) % (2**31))
        tok = random.choice(self.VOCAB_OUT)
        self.generated_tokens.append(tok)
        self.gen_step += 1
        # Re-run simulation with extended context
        full = text + " " + " ".join(self.generated_tokens)
        self.run(full)
        return tok

    def run_all(self, text, max_tokens):
        """Generate all tokens at once."""
        self.generated_tokens = []
        self.gen_step = 0
        for _ in range(max_tokens):
            tok = self.step(text, max_tokens)
            if tok is None:
                break

    def reset_generation(self):
        self.generated_tokens = []
        self.gen_step = 0

    def _build_attn_sections(self):
        """Build Softmax display sections dynamically based on selected_head."""
        h = self.selected_head
        sections = [
            ("Operation", [
                "w = softmax(scores, dim=-1)",
                f"32 heads, each {self.seq_len}x{self.seq_len}",
            ]),
            (f"Head {h} Attention (click below to change)", [], {
                "matrix": self.attn_mats[h], "tokens": self.tokens,
                "label": f"Head {h} — real attention weights",
                "cmap": "heat"}),
            ("All 32 Heads (0-7)", [], {
                "multi": [self.attn_mats[i] for i in range(8)],
                "tokens": self.tokens, "label": "Click a head to inspect",
                "head_offset": 0}),
            ("Heads 8-15", [], {
                "multi": [self.attn_mats[i] for i in range(8, 16)],
                "tokens": self.tokens, "label": "",
                "head_offset": 8}),
            ("Heads 16-23", [], {
                "multi": [self.attn_mats[i] for i in range(16, 24)],
                "tokens": self.tokens, "label": "",
                "head_offset": 16}),
            ("Heads 24-31", [], {
                "multi": [self.attn_mats[i] for i in range(24, 32)],
                "tokens": self.tokens, "label": "",
                "head_offset": 24}),
            ("Statistics", [
                f"Selected head: {h}",
                f"Max attention: {self._attn_stats.get('max', 0):.4f}",
                f"Min non-zero:  {self._attn_stats.get('min_nonzero', 0):.6f}",
            ]),
        ]
        return sections

    def get_display(self, comp_name):
        if not self.active:
            return [("No Input", ["Enter text above and press",
                                   "Enter to simulate data flow."])]
        val = self.data.get(comp_name)
        if val == "ATTN_HEADS":
            return self._build_attn_sections()
        if val == "ATTN_WV":
            h = self.selected_head
            return [
                ("Operation", [
                    "out = attn_weights @ V",
                    f"Using head {h} weights",
                ]),
                (f"Head {h} Attention Used", [], {
                    "matrix": self.attn_mats.get(h, []), "tokens": self.tokens,
                    "label": f"These weights multiplied V (head {h})",
                    "cmap": "heat"}),
            ]
        if val is None:
            return [("Info", [f"No simulation data for {comp_name}"])]
        return val


# ── Live Simulation (real model) ────────────────────────────

class LiveSimulation(Simulation):
    """Simulation backed by a real HuggingFace causal LM with hook-captured activations."""

    def __init__(self, model_path=None):
        super().__init__()
        self.model_path = model_path  # local path or HF model ID
        self.model = None
        self.tokenizer = None
        self.device = None
        self._hooks = []
        self._captures = {}
        self._loading = False
        self.model_loaded = False
        # Model config (populated after load)
        self.num_q_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 64
        self.hidden_dim = 2048
        self.intermediate_dim = 8192
        self.num_layers = 16

    def _ensure_model(self):
        if self.model is not None:
            return
        if not self.model_path:
            print("No model specified. Use --model <path>")
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading {self.model_path}...")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, dtype=torch.float32,
            attn_implementation="eager").to(self.device)
        self.model.eval()
        self.model.config.use_cache = False
        self.model_loaded = True

        # Auto-detect model config
        cfg = self.model.config
        self.num_q_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
        self.head_dim = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
        self.hidden_dim = cfg.hidden_size
        self.intermediate_dim = cfg.intermediate_size
        self.num_layers = cfg.num_hidden_layers

        print(f"Loaded on {self.device}: {self.num_layers} layers, "
              f"{self.num_q_heads}Q/{self.num_kv_heads}KV heads, "
              f"dim={self.hidden_dim}, ff={self.intermediate_dim}")

    def _install_hooks(self, block_idx):
        """Install forward hooks on a specific block to capture activations."""
        import torch
        self._remove_hooks()
        self._captures = {}
        layer = self.model.model.layers[block_idx]

        captures = self._captures

        def norm1_hook(module, inp, out):
            captures["norm1_input"] = inp[0].detach().cpu()
            captures["norm1_output"] = out.detach().cpu()

        def norm2_hook(module, inp, out):
            captures["norm2_input"] = inp[0].detach().cpu()
            captures["norm2_output"] = out.detach().cpu()

        def qproj_hook(module, inp, out):
            captures["q_raw"] = out.detach().cpu()
            captures["attn_input"] = inp[0].detach().cpu()

        def kproj_hook(module, inp, out):
            captures["k_raw"] = out.detach().cpu()

        def vproj_hook(module, inp, out):
            captures["v_raw"] = out.detach().cpu()

        def mlp_hook(module, inp, out):
            hidden = inp[0]
            captures["mlp_input"] = hidden.detach().cpu()
            captures["mlp_gate"] = module.gate_proj(hidden).detach().cpu()
            captures["mlp_up"] = module.up_proj(hidden).detach().cpu()
            captures["mlp_output"] = out.detach().cpu()

        self._hooks.append(layer.input_layernorm.register_forward_hook(norm1_hook))
        self._hooks.append(layer.post_attention_layernorm.register_forward_hook(norm2_hook))
        self._hooks.append(layer.self_attn.q_proj.register_forward_hook(qproj_hook))
        self._hooks.append(layer.self_attn.k_proj.register_forward_hook(kproj_hook))
        self._hooks.append(layer.self_attn.v_proj.register_forward_hook(vproj_hook))
        self._hooks.append(layer.mlp.register_forward_hook(mlp_hook))

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def run(self, text, block_idx=0):
        import torch
        self._ensure_model()
        self.text = text.strip()
        if not self.text:
            self.active = False
            self.data = {}
            return

        ids = self.tokenizer.encode(self.text, return_tensors="pt").to(self.device)
        self.token_ids = ids[0].tolist()
        self.tokens = [self.tokenizer.decode([tid]) for tid in self.token_ids]
        self.seq_len = len(self.tokens)
        self.active = True

        self._install_hooks(block_idx)
        with torch.no_grad():
            outputs = self.model(ids)
        self._remove_hooks()

        # Get next token prediction
        logits = outputs.logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        top_ids = torch.topk(probs, 5)
        self._top_tokens = [(self.tokenizer.decode([tid]), p.item())
                            for tid, p in zip(top_ids.indices, top_ids.values)]

        self._build_display(block_idx)

    def _fmt_tensor(self, t, rows=3, cols=6):
        """Format a tensor as display lines."""
        lines = []
        shape = list(t.shape)
        # Get a 2D view
        if t.dim() > 2:
            t = t.view(-1, t.shape[-1])
        for i in range(min(rows, t.shape[0])):
            vals = t[i, :cols].tolist()
            s = ", ".join(f"{v:+.4f}" for v in vals)
            lines.append(f"  [{s}, ...]")
        if t.shape[0] > rows:
            lines.append(f"  ... ({t.shape[0]} rows total)")
        return lines

    def _build_display(self, block_idx):
        import torch
        d = {}
        S = self.seq_len
        cap = self._captures

        tok_str = ", ".join(f'"{t.strip()}"' for t in self.tokens[:6])
        if S > 6: tok_str += ", ..."
        id_str = ", ".join(str(i) for i in self.token_ids[:6])
        if S > 6: id_str += ", ..."

        # RMSNorm1
        if "norm1_output" in cap:
            n1 = cap["norm1_output"][0]
            rms_vals = [round(n1[i].norm().item(), 4) for i in range(min(S, 4))]
            d["RMSNorm₁"] = [
                ("Operation", [
                    "x_hat = x / RMS(x) * gamma",
                    f"Block {block_idx} input layernorm",
                ]),
                ("Activations (REAL)", [
                    f"Input shape:  (1, {S}, 2048)",
                    f"Output shape: (1, {S}, 2048)",
                    f"Norm per token: {rms_vals}",
                ] + self._fmt_tensor(n1)),
            ]

        # Q, K, V heads — reshape raw projections into per-head views
        import torch
        nh = self.model.config.num_attention_heads   # 32
        nkv = self.model.config.num_key_value_heads  # 8
        hd = self.model.config.head_dim              # 64

        for prefix, key, total, n_heads in [("Q", "q_raw", 32, nh), ("K", "k_raw", 8, nkv), ("V", "v_raw", 8, nkv)]:
            if key in cap:
                raw = cap[key]  # (1, seq, total_dim)
                tensor = raw[0].view(S, n_heads, hd)  # (seq, heads, dim)
                for h in range(total):
                    name = f"{prefix}_{subscript(h)}"
                    head_data = tensor[:, h, :]  # (seq, 64)
                    stats_mean = head_data.mean().item()
                    stats_std = head_data.std().item()
                    d[name] = [
                        ("Weight Matrix (FIXED)", [
                            f"Shape: (2048, 64)",
                            f"Head {h} of {total}",
                        ]),
                        ("Activations (REAL)", [
                            f"Output: (1, {S}, 64)",
                            f"Mean: {stats_mean:.6f}",
                            f"Std:  {stats_std:.6f}",
                        ] + self._fmt_tensor(head_data)),
                    ]

        # RoPE
        d["RoPE"] = [
            ("Operation", [
                "Rotary Position Embedding",
                "Applied to Q and K after projection.",
            ]),
            ("Activations (REAL)", [
                f"Q shape: (1, 32, {S}, 64)",
                f"K shape: (1,  8, {S}, 64)",
                "(RoPE applied in-place before scores)",
            ]),
        ]

        # Compute attention scores from raw Q, K
        if "q_raw" in cap and "k_raw" in cap:
            import torch
            q = cap["q_raw"][0].view(S, nh, hd).transpose(0, 1).float()   # (nh, S, hd)
            k = cap["k_raw"][0].view(S, nkv, hd).transpose(0, 1).float()  # (nkv, S, hd)
            ng = nh // nkv
            k_exp = k.repeat_interleave(ng, dim=0)  # (nh, S, hd)
            scores = torch.matmul(q, k_exp.transpose(-2, -1)) / (hd ** 0.5)  # (nh, S, S)
            causal = torch.triu(torch.ones(S, S), diagonal=1).bool()
            scores.masked_fill_(causal.unsqueeze(0), float('-inf'))
            attn_w = torch.softmax(scores, dim=-1)

            score_mat = []
            for i in range(S):
                row = []
                for j in range(S):
                    v = scores[0, i, j].item()
                    row.append(None if v == float('-inf') else v)
                score_mat.append(row)

            d["QKt/d"] = [
                ("Operation", [
                    f"scores = Q * Kt / sqrt(64)",
                    f"Block {block_idx}, causal masked",
                ]),
                ("Activations (REAL)", [
                    f"Shape: (1, 32, {S}, {S})",
                ]),
                ("Score Heatmap (head 0)", [], {
                    "matrix": score_mat, "tokens": self.tokens,
                    "label": "Raw scores (pre-softmax)", "cmap": "diverge"}),
            ]

            # Attention weights — all 32 heads
            self.attn_mats = {}
            for h in range(nh):
                mat = []
                for i in range(S):
                    row = [attn_w[h, i, j].item() for j in range(S)]
                    mat.append(row)
                self.attn_mats[h] = mat

            # Softmax and wV use _selected_head (set by RightPanel click)
            d["Softmax"] = "ATTN_HEADS"  # sentinel — built dynamically
            d["wV"] = "ATTN_WV"

            self._attn_stats = {
                "max": attn_w.max().item(),
                "min_nonzero": attn_w[attn_w > 0].min().item(),
            }

        # W_o
        wo = self.model.model.layers[block_idx].self_attn.o_proj.weight
        d["W_o"] = [
            ("Weight Matrix (FIXED, REAL)", [
                f"Shape: {list(wo.shape)}",
                f"Params: {wo.numel():,}",
                f"Mean: {wo.mean().item():.6f}",
                f"Std:  {wo.std().item():.6f}",
            ]),
        ]

        # + res1
        d["+ res₁"] = [
            ("Operation", ["x' = x + attn(x)"]),
        ]

        # RMSNorm2
        if "norm2_output" in cap:
            n2 = cap["norm2_output"][0]
            d["RMSNorm₂"] = [
                ("Activations (REAL)", [
                    f"Shape: (1, {S}, 2048)",
                ] + self._fmt_tensor(n2)),
            ]

        # MLP components
        layer = self.model.model.layers[block_idx]
        for wname, attr in [("W_gate", "gate_proj"), ("W_up", "up_proj"), ("W_down", "down_proj")]:
            w = getattr(layer.mlp, attr).weight
            d[wname] = [
                ("Weight Matrix (FIXED, REAL)", [
                    f"Shape: {list(w.shape)}",
                    f"Params: {w.numel():,}",
                    f"Mean: {w.mean().item():.6f}",
                    f"Std:  {w.std().item():.6f}",
                ]),
            ]
            if wname == "W_gate" and "mlp_gate" in cap:
                g = cap["mlp_gate"][0]
                d[wname].append(("Activations (REAL)", [
                    f"Shape: (1, {S}, {g.shape[-1]})",
                ] + self._fmt_tensor(g)))
            elif wname == "W_up" and "mlp_up" in cap:
                u = cap["mlp_up"][0]
                d[wname].append(("Activations (REAL)", [
                    f"Shape: (1, {S}, {u.shape[-1]})",
                ] + self._fmt_tensor(u)))

        # SiLU
        if "mlp_gate" in cap:
            import torch
            g = cap["mlp_gate"][0]
            silu_out = torch.nn.functional.silu(g)
            d["SiLU"] = [
                ("Activations (REAL)", [
                    f"Shape: (1, {S}, {g.shape[-1]})",
                ] + self._fmt_tensor(silu_out)),
            ]

        # gate*up
        if "mlp_gate" in cap and "mlp_up" in cap:
            import torch
            g = torch.nn.functional.silu(cap["mlp_gate"][0])
            u = cap["mlp_up"][0]
            prod = g * u
            d["gate*up"] = [
                ("Activations (REAL)", [
                    f"Shape: (1, {S}, {prod.shape[-1]})",
                ] + self._fmt_tensor(prod)),
            ]

        # + res2
        d["+ res₂"] = [("Operation", ["x'' = x' + mlp(x')"])]

        # Next token prediction
        d["_next_token"] = self._top_tokens

        self.data = d

    def step(self, text, max_tokens):
        if self.gen_step >= max_tokens:
            return None
        import torch
        self._ensure_model()
        full = text
        if self.generated_tokens:
            full = text + " " + " ".join(self.generated_tokens)
        ids = self.tokenizer.encode(full, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(ids).logits[0, -1]
        # Greedy
        next_id = logits.argmax().item()
        tok = self.tokenizer.decode([next_id])
        self.generated_tokens.append(tok.strip())
        self.gen_step += 1
        # Re-run with full context to update captures
        full_new = text + " " + " ".join(self.generated_tokens)
        self.run(full_new, block_idx=0)
        return tok.strip()

    def run_all(self, text, max_tokens):
        import torch
        self._ensure_model()
        self.generated_tokens = []
        self.gen_step = 0
        ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=max_tokens,
                do_sample=False, use_cache=False,
                pad_token_id=self.tokenizer.eos_token_id)
        gen_ids = out[0, ids.shape[1]:]
        self.generated_tokens = [self.tokenizer.decode([tid]).strip() for tid in gen_ids]
        self.gen_step = len(self.generated_tokens)
        # Run final state through hooks
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        self.run(full, block_idx=0)


# ── Camera ──────────────────────────────────────────────────

class IsoCamera:
    GRID = 20

    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.ax = 7
        self.ay = 0
        self.zoom = 1.21
        self.px = 290
        self.py = 170

    def project(self, x, y, z=0):
        ry = math.radians(self.ay)
        rx = math.radians(self.ax)
        x2 = x * math.cos(ry) - z * math.sin(ry)
        z2 = x * math.sin(ry) + z * math.cos(ry)
        y2 = y * math.cos(rx) - z2 * math.sin(rx)
        sx = int(x2 * self.zoom + self.w // 2 + self.px)
        sy = int(-y2 * self.zoom + self.h // 2 + self.py)
        return sx, sy

    def project_grid(self, gx, gy, gz=0):
        return self.project(gx * self.GRID, gy * self.GRID, gz)

    def world_size(self, gw, gh):
        sw = max(int(gw * self.GRID * self.zoom * 0.5), 15)
        sh = max(int(gh * self.GRID * self.zoom * 0.5), 8)
        return sw, sh


# ── Renderer ────────────────────────────────────────────────

class Renderer:
    """Draws diamonds, arrows, and grid on a pygame surface."""

    def __init__(self, screen, cam, fonts):
        self.screen = screen
        self.cam = cam
        self.fonts = fonts

    def draw_grid(self, extent=12):
        G = self.cam.GRID
        color = (32, 34, 40)
        for i in range(-extent, extent + 1):
            p1 = self.cam.project(i * G, -extent * G)
            p2 = self.cam.project(i * G,  extent * G)
            pygame.draw.line(self.screen, color, p1, p2, 1)
            p1 = self.cam.project(-extent * G, i * G)
            p2 = self.cam.project( extent * G, i * G)
            pygame.draw.line(self.screen, color, p1, p2, 1)

    def draw_diamond(self, sx, sy, sw, sh, color, selected=False):
        pts = [(sx, sy - sh), (sx + sw, sy), (sx, sy + sh), (sx - sw, sy)]
        pygame.draw.polygon(self.screen, color, pts)
        bc = (255, 220, 50) if selected else tuple(min(255, c + 40) for c in color)
        bw = 2 if selected else 1
        pygame.draw.polygon(self.screen, bc, pts, bw)
        s = 0.3
        ipts = [(sx, sy - int(sh * s)), (sx + int(sw * s), sy),
                (sx, sy + int(sh * s)), (sx - int(sw * s), sy)]
        ic = tuple(min(255, c + 60) for c in color)
        pygame.draw.polygon(self.screen, ic, ipts)
        pygame.draw.polygon(self.screen, bc, ipts, 1)
        return pygame.Rect(sx - sw, sy - sh, sw * 2, sh * 2)

    def draw_arrow(self, p1, p2, color, width=2, dashed=False):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 5:
            return
        if dashed:
            segs = max(int(dist / 10), 1)
            for s in range(0, segs, 2):
                t1, t2 = s / segs, min((s + 1) / segs, 1.0)
                a = (int(p1[0] + dx * t1), int(p1[1] + dy * t1))
                b = (int(p1[0] + dx * t2), int(p1[1] + dy * t2))
                pygame.draw.line(self.screen, color, a, b, width)
        else:
            pygame.draw.line(self.screen, color, p1, p2, width)
        nx, ny = dx / dist, dy / dist
        ax, ay = p2[0] - nx * 10, p2[1] - ny * 10
        px, py = -ny * 5, nx * 5
        pygame.draw.polygon(self.screen, color, [
            p2, (int(ax + px), int(ay + py)), (int(ax - px), int(ay - py))])

    def draw_component(self, comp, sx, sy, sw, sh, selected=False):
        rect = self.draw_diamond(sx, sy, sw, sh, comp.color, selected)
        if not comp.is_skip:
            col = (255, 255, 255) if selected else (160, 160, 170)
            lbl = self.fonts["m"].render(comp.name, True, col)
            self.screen.blit(lbl, (sx - lbl.get_width() // 2, sy - sh - 16))
        return rect

    def draw_connection(self, conn, proj):
        fp, tp = proj[conn.src], proj[conn.dst]
        color = (160, 140, 50) if conn.skip else (80, 150, 100)
        self.draw_arrow((fp[0], fp[1]), (tp[0], tp[1]),
                        color, width=1 if conn.skip else 2, dashed=conn.skip)

    def draw_section_label(self, text, gx, gy, color):
        pos = self.cam.project_grid(gx, gy)
        self.screen.blit(self.fonts["l"].render(text, True, color), pos)


# ── Top Bar ─────────────────────────────────────────────────

class TopBar:
    """Text input + toolbar (Run All / Step / max tokens)."""

    HEIGHT = 72  # two rows

    def __init__(self, fonts):
        self.fonts = fonts
        self.text = ""
        self.focused = False
        self.cursor_blink = 0
        self.max_tokens = 20
        self.max_tokens_focused = False
        self.max_tokens_text = "20"
        # Rects set during draw
        self.input_rect = pygame.Rect(0, 0, 0, 0)
        self.enter_rect = pygame.Rect(0, 0, 0, 0)
        self.run_all_rect = pygame.Rect(0, 0, 0, 0)
        self.step_rect = pygame.Rect(0, 0, 0, 0)
        self.max_tok_rect = pygame.Rect(0, 0, 0, 0)
        self.reset_rect = pygame.Rect(0, 0, 0, 0)

    def _draw_button(self, screen, rect, label, color, enabled=True):
        c = color if enabled else (45, 47, 55)
        pygame.draw.rect(screen, c, rect, border_radius=4)
        pygame.draw.rect(screen, tuple(min(255, x + 30) for x in c), rect, 1, border_radius=4)
        tc = (220, 220, 230) if enabled else (100, 100, 110)
        lbl = self.fonts["m"].render(label, True, tc)
        screen.blit(lbl, (rect.x + (rect.w - lbl.get_width()) // 2, rect.y + 4))

    def draw(self, screen, W):
        # Background
        pygame.draw.rect(screen, (30, 32, 42), (0, 0, W, self.HEIGHT))
        pygame.draw.line(screen, (50, 52, 60), (0, self.HEIGHT), (W, self.HEIGHT), 1)

        # ── Row 1: text input ──
        screen.blit(self.fonts["m"].render("Input:", True, (150, 150, 160)), (12, 10))
        ix = 70
        iw = W - ix - 90
        self.input_rect = pygame.Rect(ix, 6, iw, 26)
        bc = (100, 140, 220) if self.focused else (60, 62, 72)
        pygame.draw.rect(screen, (38, 40, 52), self.input_rect, border_radius=4)
        pygame.draw.rect(screen, bc, self.input_rect, 1, border_radius=4)

        txt_surf = self.fonts["m"].render(self.text, True, (220, 220, 230))
        clip = self.input_rect.inflate(-8, -4)
        screen.set_clip(clip)
        tx = clip.x
        if txt_surf.get_width() > clip.width:
            tx = clip.right - txt_surf.get_width()
        screen.blit(txt_surf, (tx, clip.y + 1))
        screen.set_clip(None)

        if self.focused:
            self.cursor_blink = (self.cursor_blink + 1) % 60
            if self.cursor_blink < 35:
                cx = min(tx + txt_surf.get_width() + 1, clip.right - 1)
                pygame.draw.line(screen, (200, 200, 220), (cx, clip.y + 1), (cx, clip.bottom - 2))
        elif not self.text:
            screen.blit(self.fonts["m"].render("Type tokens here...", True, (80, 80, 90)),
                        (clip.x, clip.y + 1))

        self.enter_rect = pygame.Rect(ix + iw + 8, 6, 70, 26)
        self._draw_button(screen, self.enter_rect, "Enter", (60, 120, 200), bool(self.text))

        # ── Row 2: toolbar ──
        ty = 38
        # Max tokens
        screen.blit(self.fonts["s"].render("Max tokens:", True, (130, 130, 140)), (12, ty + 5))
        self.max_tok_rect = pygame.Rect(100, ty + 2, 44, 22)
        mtc = (100, 140, 220) if self.max_tokens_focused else (60, 62, 72)
        pygame.draw.rect(screen, (38, 40, 52), self.max_tok_rect, border_radius=3)
        pygame.draw.rect(screen, mtc, self.max_tok_rect, 1, border_radius=3)
        mt_text = self.max_tokens_text if self.max_tokens_focused else str(self.max_tokens)
        screen.blit(self.fonts["m"].render(mt_text, True, (220, 220, 230)),
                    (self.max_tok_rect.x + 5, ty + 4))

        # Run All button
        self.run_all_rect = pygame.Rect(160, ty + 2, 80, 22)
        self._draw_button(screen, self.run_all_rect, "▶ Run All", (40, 120, 60), bool(self.text))

        # Step button
        self.step_rect = pygame.Rect(248, ty + 2, 70, 22)
        self._draw_button(screen, self.step_rect, "▶| Step", (120, 100, 40), bool(self.text))

        # Reset button
        self.reset_rect = pygame.Rect(326, ty + 2, 62, 22)
        self._draw_button(screen, self.reset_rect, "Reset", (140, 50, 50), True)

    def handle_event(self, ev):
        """Returns dict action or None.
        Actions: {"type": "submit", "text": ...}
                 {"type": "run_all", "text": ..., "max_tokens": ...}
                 {"type": "step",    "text": ..., "max_tokens": ...}
        """
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            pos = ev.pos
            # Input field
            if self.input_rect.collidepoint(pos):
                self.focused = True
                self.max_tokens_focused = False
                return None
            # Max tokens field
            if self.max_tok_rect.collidepoint(pos):
                self.max_tokens_focused = True
                self.max_tokens_text = str(self.max_tokens)
                self.focused = False
                return None
            # Enter
            if self.enter_rect.collidepoint(pos) and self.text:
                self._commit_max_tokens()
                return {"type": "submit", "text": self.text}
            # Run All
            if self.run_all_rect.collidepoint(pos) and self.text:
                self._commit_max_tokens()
                return {"type": "run_all", "text": self.text, "max_tokens": self.max_tokens}
            # Step
            if self.step_rect.collidepoint(pos) and self.text:
                self._commit_max_tokens()
                return {"type": "step", "text": self.text, "max_tokens": self.max_tokens}
            # Reset
            if self.reset_rect.collidepoint(pos):
                return {"type": "reset"}
            # Click elsewhere
            self.focused = False
            if self.max_tokens_focused:
                self._commit_max_tokens()
                self.max_tokens_focused = False
            return None

        if ev.type == pygame.KEYDOWN:
            if self.max_tokens_focused:
                if ev.key == pygame.K_RETURN:
                    self._commit_max_tokens()
                    self.max_tokens_focused = False
                elif ev.key == pygame.K_ESCAPE:
                    self.max_tokens_focused = False
                elif ev.key == pygame.K_BACKSPACE:
                    self.max_tokens_text = self.max_tokens_text[:-1]
                elif ev.unicode and ev.unicode.isdigit():
                    self.max_tokens_text += ev.unicode
                return None
            if self.focused:
                if ev.key == pygame.K_RETURN and self.text:
                    return {"type": "submit", "text": self.text}
                elif ev.key == pygame.K_ESCAPE:
                    self.focused = False
                elif ev.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                elif ev.unicode and ev.unicode.isprintable():
                    self.text += ev.unicode
                return None
        return None

    def _commit_max_tokens(self):
        try:
            v = int(self.max_tokens_text)
            self.max_tokens = max(1, min(200, v))
        except ValueError:
            pass
        self.max_tokens_text = str(self.max_tokens)


# ── Left Panel ──────────────────────────────────────────────

class LeftPanel:
    """Collapsible component tree on the left side with block selector."""

    NUM_BLOCKS = 16
    MODULE_COLORS = {"attention": (50, 130, 70), "transform": (190, 100, 40)}
    MODULE_LABELS = {"attention": "Attention Module", "transform": "Transform Module"}
    HEAD_COUNTS = {"Q": 32, "K": 8, "V": 8}
    TYPE_COLORS = {"weight": (100,150,255), "op": (100,210,100), "data": (210,210,100)}

    def __init__(self, width, fonts):
        self.width = width
        self.fonts = fonts
        self.module_expanded = {"attention": False, "transform": False}
        self.head_expanded = {"Q": False, "K": False, "V": False}
        self.click_zones = []
        self.current_block = 0
        self.block_dropdown_open = False
        self.output_scroll = 0

    def _draw_row(self, screen, y, indent, text, color_dot, dot_r, selected, dims=None):
        w = self.width - indent - 8
        if selected:
            pygame.draw.rect(screen, (50, 60, 80), (indent, y, w, 18), border_radius=3)
        pygame.draw.circle(screen, color_dot, (indent + dot_r + 5, y + 9), dot_r)
        col = (255, 255, 255) if selected else (140, 140, 150)
        screen.blit(self.fonts["s"].render(text, True, col), (indent + dot_r * 2 + 10, y + 2))
        if dims:
            screen.blit(self.fonts["s"].render(dims, True, (90, 100, 110)),
                        (indent + 100, y + 2))
        return y + 19

    def _draw_group_header(self, screen, y, indent, label, expanded, color_dot, selected):
        arrow = "v" if expanded else ">"
        w = self.width - indent - 8
        if selected and not expanded:
            pygame.draw.rect(screen, (40, 50, 65), (indent, y, w, 20), border_radius=3)
        col = (255, 255, 255) if selected else (190, 190, 200)
        if isinstance(color_dot, tuple) and len(color_dot) == 3:
            pygame.draw.rect(screen, color_dot, (indent + 4, y + 4, 10, 10), border_radius=2)
            screen.blit(self.fonts["m"].render(f"{arrow} {label}", True, col), (indent + 18, y + 2))
        else:
            pygame.draw.circle(screen, color_dot, (indent + 10, y + 9), 4)
            screen.blit(self.fonts["s"].render(f"{arrow} {label}", True, col), (indent + 18, y + 2))
        return y + 22

    def draw(self, screen, components, sel_idx, top, H, simulation=None):
        pygame.draw.rect(screen, (28, 30, 38), (0, top, self.width, H - top))
        pygame.draw.line(screen, (50, 52, 60), (self.width, top), (self.width, H), 1)

        y = top + 10
        screen.blit(self.fonts["t"].render("LLaMA 3.2-1B", True, (220, 220, 230)), (12, y))
        y += 26

        # Block selector — grid of numbered buttons
        screen.blit(self.fonts["m"].render("Block:", True, (150, 150, 160)), (12, y + 2))
        self.click_zones = []
        bx = 60
        btn_w = 24
        btn_h = 20
        cols = 8
        for b in range(self.NUM_BLOCKS):
            col_i = b % cols
            row_i = b // cols
            rx = bx + col_i * (btn_w + 2)
            ry = y + row_i * (btn_h + 2)
            is_cur = b == self.current_block
            bg = (60, 100, 180) if is_cur else (42, 44, 54)
            pygame.draw.rect(screen, bg, (rx, ry, btn_w, btn_h), border_radius=3)
            if is_cur:
                pygame.draw.rect(screen, (100, 160, 255), (rx, ry, btn_w, btn_h), 1, border_radius=3)
            tc = (255, 255, 255) if is_cur else (130, 130, 140)
            lbl = self.fonts["s"].render(str(b), True, tc)
            screen.blit(lbl, (rx + (btn_w - lbl.get_width()) // 2, ry + 3))
            self.click_zones.append((pygame.Rect(rx, ry, btn_w, btn_h), ("select_block", b)))
        rows = (self.NUM_BLOCKS + cols - 1) // cols
        y += rows * (btn_h + 2) + 6

        # Selected info box (skip data types)
        sc = components[sel_idx]
        if sc.typ not in ("data", "skip"):
            pygame.draw.rect(screen, (40, 42, 52), (8, y, self.width - 16, 70), border_radius=5)
            screen.blit(self.fonts["l"].render(sc.name, True, (255, 255, 255)), (15, y + 5))
            screen.blit(self.fonts["m"].render(sc.dims, True, (180, 210, 240)), (15, y + 26))
            tc = self.TYPE_COLORS.get(sc.typ, (200, 200, 200))
            screen.blit(self.fonts["m"].render(f"Type: {sc.typ}", True, tc), (15, y + 48))
            y += 80

        screen.blit(self.fonts["m"].render("Components:", True, (160, 160, 170)), (12, y))
        y += 22

        # Keep block button zones, add component zones below
        drawn_modules = set()
        drawn_heads = set()
        sel_comp = components[sel_idx]

        for i, c in enumerate(components):
            if c.is_skip or c.typ == "data":
                continue
            mod = c.module

            if mod and mod not in drawn_modules:
                drawn_modules.add(mod)
                exp = self.module_expanded[mod]
                label = self.MODULE_LABELS[mod]
                mc = self.MODULE_COLORS[mod]
                mod_selected = sel_comp.module == mod
                y = self._draw_group_header(screen, y, 8, label, exp, mc, mod_selected)
                self.click_zones.append((pygame.Rect(8, y - 22, self.width - 16, 22),
                                         ("toggle_mod", mod)))
                if not exp:
                    continue

            if mod and not self.module_expanded[mod]:
                continue

            grp = c.head_group
            if grp and grp not in drawn_heads:
                drawn_heads.add(grp)
                exp = self.head_expanded[grp]
                n = self.HEAD_COUNTS[grp]
                grp_selected = sel_comp.head_group == grp
                y = self._draw_group_header(screen, y, 22, f"{grp} Heads ({n})",
                                            exp, c.color, grp_selected)
                self.click_zones.append((pygame.Rect(22, y - 22, self.width - 30, 22),
                                         ("toggle_head", grp)))
                if not exp:
                    continue

            if grp and not self.head_expanded[grp]:
                continue

            if grp:
                y = self._draw_row(screen, y, 36, c.name, c.color, 3, i == sel_idx)
            elif mod:
                y = self._draw_row(screen, y, 22, c.name, c.color, 5, i == sel_idx, c.dims)
            else:
                y = self._draw_row(screen, y, 8, c.name, c.color, 5, i == sel_idx, c.dims)
            self.click_zones.append((pygame.Rect(8, y - 19, self.width - 16, 19),
                                     ("select", i)))

        # ── Output Box ──
        y += 10
        out_top = y
        out_h = H - y - 4
        if out_h > 40 and simulation is not None:
            pygame.draw.rect(screen, (22, 24, 30), (6, out_top, self.width - 12, out_h), border_radius=4)
            pygame.draw.rect(screen, (50, 55, 65), (6, out_top, self.width - 12, out_h), 1, border_radius=4)

            # Header
            gen_count = len(simulation.generated_tokens)
            hdr = f"Output ({gen_count} tokens)" if gen_count else "Output"
            screen.blit(self.fonts["m"].render(hdr, True, (160, 170, 190)), (12, out_top + 4))
            oy = out_top + 22

            clip = pygame.Rect(8, oy, self.width - 16, out_h - 24)
            screen.set_clip(clip)

            if simulation.generated_tokens:
                # Word-wrap generated text
                words = simulation.generated_tokens
                line = ""
                line_y = oy - self.output_scroll
                for w in words:
                    test = line + (" " if line else "") + w
                    tw = self.fonts["m"].size(test)[0]
                    if tw > self.width - 32 and line:
                        if line_y >= oy - 2:
                            screen.blit(self.fonts["m"].render(line, True, (100, 220, 140)),
                                        (14, line_y))
                        line_y += 16
                        line = w
                    else:
                        line = test
                if line and line_y >= oy - 2:
                    # Blinking cursor at end
                    cursor = "▌" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
                    screen.blit(self.fonts["m"].render(line + cursor, True, (100, 220, 140)),
                                (14, line_y))
            else:
                screen.blit(self.fonts["s"].render("Press Run All or Step", True, (70, 70, 80)),
                            (14, oy + 2))
                screen.blit(self.fonts["s"].render("to generate tokens.", True, (70, 70, 80)),
                            (14, oy + 16))
            screen.set_clip(None)

    def handle_click(self, mx, my):
        for rect, action in self.click_zones:
            if rect.collidepoint(mx, my):
                if action[0] == "select_block":
                    old = self.current_block
                    self.current_block = action[1]
                    if old != self.current_block:
                        return ("switch_block", self.current_block)
                    return None
                elif action[0] == "toggle_mod":
                    self.module_expanded[action[1]] = not self.module_expanded[action[1]]
                elif action[0] == "toggle_head":
                    self.head_expanded[action[1]] = not self.head_expanded[action[1]]
                elif action[0] == "select":
                    return action
                return None
        return None


# ── Right Panel ─────────────────────────────────────────────

class RightPanel:
    """Context-dependent data display on the right side."""

    def __init__(self, width, fonts):
        self.width = width
        self.fonts = fonts
        self.scroll = 0
        self.max_scroll = 0
        self.save_rect = pygame.Rect(0, 0, 0, 0)
        self.save_flash = 0
        self.head_click_zones = []  # (rect, head_idx)

    def _heat_color(self, val, cmap="heat"):
        """Map 0..1 value to an RGB color."""
        if cmap == "heat":
            # Black → Blue → Cyan → Yellow → White
            if val < 0.25:
                t = val / 0.25
                return (0, 0, int(80 + 175 * t))
            elif val < 0.5:
                t = (val - 0.25) / 0.25
                return (0, int(200 * t), 255)
            elif val < 0.75:
                t = (val - 0.5) / 0.25
                return (int(255 * t), 200 + int(55 * t), int(255 * (1 - t)))
            else:
                t = (val - 0.75) / 0.25
                return (255, 255, int(180 + 75 * t))
        else:  # diverge: blue (neg) → black (0) → red (pos)
            if val < 0.5:
                t = val / 0.5
                return (0, 0, int(220 * (1 - t)))
            else:
                t = (val - 0.5) / 0.5
                return (int(220 * t), 0, 0)

    def _draw_heatmap(self, screen, x_offset, y, hm_data):
        """Draw a heatmap. Returns new y."""
        tokens = hm_data["tokens"]
        cmap = hm_data.get("cmap", "heat")
        label = hm_data.get("label", "")

        # Multi-head small multiples
        if "multi" in hm_data:
            matrices = hm_data["multi"]
            n = len(tokens)
            n_show = min(n, 10)
            per_head = min(40, (self.width - 40) // len(matrices) - 4)
            cell = max(3, per_head // n_show)
            map_sz = cell * n_show

            head_offset = hm_data.get("head_offset", 0)
            selected = getattr(self, '_sel_head', 0)

            for hi, mat in enumerate(matrices):
                hx = x_offset + 16 + hi * (map_sz + 8)
                abs_head = head_offset + hi
                # Head label — highlight selected
                is_sel = abs_head == selected
                lc = (255, 220, 50) if is_sel else (120, 120, 140)
                screen.blit(self.fonts["s"].render(f"h{abs_head}", True, lc),
                             (hx + map_sz // 2 - 7, y))

            y += 14

            for hi, mat in enumerate(matrices):
                hx = x_offset + 16 + hi * (map_sz + 8)
                abs_head = head_offset + hi
                is_sel = abs_head == selected
                # Draw border for selected head
                if is_sel:
                    pygame.draw.rect(screen, (255, 220, 50),
                                     (hx - 1, y - 1, map_sz + 2, map_sz + 2), 1)
                for i in range(n_show):
                    for j in range(n_show):
                        v = mat[i][j] if mat[i][j] is not None else 0
                        c = self._heat_color(min(1.0, max(0.0, v)), cmap)
                        pygame.draw.rect(screen, c, (hx + j * cell, y + i * cell, cell - 1, cell - 1))
                # Click zone for this head
                self.head_click_zones.append(
                    (pygame.Rect(hx, y, map_sz, map_sz), abs_head))
            y += map_sz + 4

            if label:
                screen.blit(self.fonts["s"].render(label, True, (100, 100, 115)),
                             (x_offset + 16, y))
                y += 14
            return y + 6

        # Single heatmap
        matrix = hm_data["matrix"]
        n = len(matrix)
        n_show = min(n, 12)

        # Layout
        margin_left = 50  # space for row labels
        margin_top = 16   # space for column labels
        avail_w = self.width - 32 - margin_left
        cell = max(6, min(22, avail_w // n_show))
        map_w = cell * n_show
        map_h = cell * n_show
        hx = x_offset + 16 + margin_left
        hy = y + margin_top

        # Normalize values for color mapping
        if cmap == "diverge":
            all_vals = [v for row in matrix[:n_show] for v in row[:n_show] if v is not None]
            max_abs = max(abs(v) for v in all_vals) if all_vals else 1.0
            norm = lambda v: 0.5 + 0.5 * v / max_abs if v is not None else 0.5
        else:
            norm = lambda v: min(1.0, max(0.0, v)) if v is not None else 0.0

        # Column labels (top)
        for j in range(n_show):
            tok = tokens[j][:4] if j < len(tokens) else ""
            lbl = self.fonts["s"].render(tok, True, (100, 140, 100))
            # Rotate-ish: just draw at angle via positioning
            screen.blit(lbl, (hx + j * cell, y))

        # Draw cells + row labels
        for i in range(n_show):
            # Row label (left)
            tok = tokens[i][:5] if i < len(tokens) else ""
            lbl = self.fonts["s"].render(tok, True, (140, 140, 100))
            screen.blit(lbl, (x_offset + 14, hy + i * cell + 1))

            for j in range(n_show):
                v = matrix[i][j]
                if v is None:
                    # Masked cell — dark
                    c = (30, 30, 35)
                else:
                    c = self._heat_color(norm(v), cmap)
                rect = pygame.Rect(hx + j * cell, hy + i * cell, cell - 1, cell - 1)
                pygame.draw.rect(screen, c, rect)

                # Show value on hover (if cell is big enough)
                if cell >= 16 and v is not None:
                    vt = f"{v:.1f}" if cmap == "diverge" else f"{v:.2f}"
                    vs = self.fonts["s"].render(vt, True, (0, 0, 0))
                    screen.blit(vs, (rect.x + 1, rect.y + 1))

        y = hy + map_h + 6

        # Label
        if label:
            screen.blit(self.fonts["s"].render(label, True, (100, 100, 115)),
                         (x_offset + 16, y))
            y += 14

        # Colorbar
        bar_w = min(map_w, 120)
        bar_h = 8
        bx = hx
        for px in range(bar_w):
            t = px / bar_w
            c = self._heat_color(t, cmap)
            pygame.draw.line(screen, c, (bx + px, y + 2), (bx + px, y + 2 + bar_h))
        lo = "neg" if cmap == "diverge" else "0"
        hi = "pos" if cmap == "diverge" else "1"
        screen.blit(self.fonts["s"].render(lo, True, (90, 90, 100)), (bx - 4, y))
        screen.blit(self.fonts["s"].render(hi, True, (90, 90, 100)), (bx + bar_w + 4, y))
        y += bar_h + 14

        return y

    def draw(self, screen, simulation, comp, x_offset, top, H):
        panel_rect = pygame.Rect(x_offset, top, self.width, H - top)
        pygame.draw.rect(screen, (26, 28, 36), panel_rect)
        pygame.draw.line(screen, (50, 52, 60), (x_offset, top), (x_offset, H), 1)

        # Only show content for actual components (not data/skip)
        if comp.typ in ("data", "skip") or not simulation.active:
            hy = top + 40
            msg = "Select a component to inspect." if simulation.active else \
                  "Enter text and press Enter."
            screen.blit(self.fonts["m"].render(msg, True, (80, 80, 95)),
                         (x_offset + 16, hy))
            return

        y = top + 10

        # Component name + Save button
        pygame.draw.rect(screen, (40, 42, 52),
                         (x_offset + 8, y, self.width - 16, 28), border_radius=4)
        screen.blit(self.fonts["l"].render(comp.name, True, (255, 220, 100)),
                     (x_offset + 15, y + 5))

        # Save button
        self.save_rect = pygame.Rect(x_offset + self.width - 70, y + 3, 54, 22)
        if self.save_flash > 0:
            pygame.draw.rect(screen, (40, 150, 60), self.save_rect, border_radius=3)
            lbl = self.fonts["s"].render("Saved!", True, (220, 255, 220))
            self.save_flash -= 1
        else:
            pygame.draw.rect(screen, (60, 80, 120), self.save_rect, border_radius=3)
            pygame.draw.rect(screen, (80, 100, 150), self.save_rect, 1, border_radius=3)
            lbl = self.fonts["s"].render("Save", True, (200, 200, 220))
        screen.blit(lbl, (self.save_rect.x + (self.save_rect.w - lbl.get_width()) // 2,
                          self.save_rect.y + 4))
        y += 38

        # Pass selected head to heatmap drawing
        self._sel_head = simulation.selected_head

        # Sections with scroll
        self.head_click_zones = []
        sections = simulation.get_display(comp.name)
        content_top = y
        y -= self.scroll
        clip_rect = pygame.Rect(x_offset, content_top, self.width, H - content_top)
        screen.set_clip(clip_rect)

        for section in sections:
            if y > H + 200:  # render a bit beyond viewport for smooth scroll
                break

            title = section[0]
            lines = section[1]
            hm_data = section[2] if len(section) > 2 else None

            # Section title
            pygame.draw.rect(screen, (45, 48, 58),
                             (x_offset + 8, y, self.width - 16, 20), border_radius=3)
            screen.blit(self.fonts["m"].render(title, True, (140, 180, 220)),
                         (x_offset + 14, y + 3))
            y += 24

            # Text lines
            for line in lines:
                if y > H:
                    break
                if not line:
                    y += 6
                    continue
                color = (160, 160, 170)
                if line.startswith("Shape:") or "shape:" in line.lower():
                    color = (120, 200, 140)
                elif line.startswith("Params:"):
                    color = (200, 160, 100)
                elif "FIXED" in line:
                    color = (180, 130, 130)
                elif line.startswith("[") or line.startswith("  ["):
                    color = (130, 150, 180)
                screen.blit(self.fonts["s"].render(line, True, color),
                             (x_offset + 16, y))
                y += 15

            # Heatmap
            if hm_data and y < H:
                y += 4
                y = self._draw_heatmap(screen, x_offset, y, hm_data)

            y += 8

        # Track content height for scroll limits
        content_h = (y + self.scroll) - content_top
        self.max_scroll = max(0, content_h - (H - content_top) + 20)
        self.scroll = max(0, min(self.scroll, self.max_scroll))

        screen.set_clip(None)

    def handle_scroll(self, dy):
        self.scroll = max(0, min(self.scroll - dy * 20, self.max_scroll))

    def handle_click(self, mx, my, simulation=None):
        """Returns 'save' if save clicked, else handles head selection."""
        if self.save_rect.collidepoint(mx, my):
            return "save"
        # Check head click zones
        for rect, head_idx in self.head_click_zones:
            if rect.collidepoint(mx, my):
                if simulation:
                    simulation.selected_head = head_idx
                    self.scroll = 0  # scroll back to top to see updated heatmap
                return None
        return None

    def save_data(self, run_dir, comp, simulation, block_idx):
        """Save current component's data to the run directory."""
        import numpy as np

        safe_name = comp.name.replace("/", "_").replace("*", "x").replace(" ", "_")
        comp_dir = os.path.join(run_dir, f"block_{block_idx}", safe_name)
        os.makedirs(comp_dir, exist_ok=True)

        sections = simulation.get_display(comp.name)
        stats = {
            "component": comp.name,
            "block": block_idx,
            "type": comp.typ,
            "dims": comp.dims,
            "input_text": simulation.text,
            "tokens": simulation.tokens,
            "token_ids": simulation.token_ids,
        }

        for section in sections:
            title = section[0]
            lines = section[1]
            hm_data = section[2] if len(section) > 2 else None

            # Save text info
            if lines:
                stats[title] = lines

            # Save matrix data as numpy + compute stats
            if hm_data:
                tokens = hm_data.get("tokens", simulation.tokens)

                if "matrix" in hm_data:
                    mat = hm_data["matrix"]
                    arr = []
                    for row in mat:
                        arr.append([float('nan') if v is None else v for v in row])
                    arr = np.array(arr)
                    fname = f"{title.replace(' ', '_').replace('/', '_')}.npy"
                    np.save(os.path.join(comp_dir, fname), arr)
                    stats[f"{title}_file"] = fname

                    # Compute summary stats on the matrix
                    valid = arr[~np.isnan(arr)]
                    ms = {
                        "mean": float(valid.mean()),
                        "variance": float(valid.var()),
                        "std": float(valid.std()),
                        "min": float(valid.min()),
                        "max": float(valid.max()),
                        "global_argmax": int(np.nanargmax(arr)),
                    }
                    # Per-row: which token gets max attention
                    max_per_row = []
                    for ri in range(arr.shape[0]):
                        row = arr[ri]
                        row_valid = np.where(np.isnan(row), -np.inf, row)
                        mj = int(np.argmax(row_valid))
                        tok = tokens[mj] if mj < len(tokens) else f"[{mj}]"
                        max_per_row.append({
                            "token": tokens[ri].strip() if ri < len(tokens) else f"[{ri}]",
                            "attends_to": tok.strip(),
                            "weight": float(row_valid[mj]),
                        })
                    ms["per_token_max"] = max_per_row
                    stats[f"{title}_stats"] = ms

                if "multi" in hm_data:
                    multi_stats = []
                    for hi, mat in enumerate(hm_data["multi"]):
                        arr = np.array(mat)
                        fname = f"{title.replace(' ', '_')}_head{hi}.npy"
                        np.save(os.path.join(comp_dir, fname), arr)
                        valid = arr[arr > 0]
                        hs = {
                            "head": hi,
                            "mean": float(valid.mean()) if len(valid) else 0,
                            "variance": float(valid.var()) if len(valid) else 0,
                            "max": float(arr.max()),
                            "diagonal_avg": float(np.diag(arr).mean()),
                            "bos_attn_avg": float(arr[:, 0].mean()),
                        }
                        multi_stats.append(hs)
                    stats[f"{title}_heads"] = multi_stats

        # Save raw captured tensors if available (LiveSimulation)
        if hasattr(simulation, '_captures') and simulation._captures:
            cap = simulation._captures
            tensor_keys = {
                "Q": "q_raw", "K": "k_raw", "V": "v_raw",
                "mlp_gate": "mlp_gate", "mlp_up": "mlp_up",
                "norm1_output": "norm1_output", "norm2_output": "norm2_output",
            }
            for label, key in tensor_keys.items():
                if key in cap and comp.name.startswith(label[0]):
                    np.save(os.path.join(comp_dir, f"{label}_tensor.npy"),
                            cap[key].numpy())

        # Write stats JSON
        with open(os.path.join(comp_dir, "info.json"), "w") as f:
            json.dump(stats, f, indent=2, default=str)

        print(f"Saved: {comp_dir}")
        self.save_flash = 60  # show feedback for ~1 second


# ── Command Processor ───────────────────────────────────────

class CommandProcessor:
    """Background thread reading stdin commands while pygame runs."""

    HELP = """
Commands (type in terminal while inspector runs):
  save block <N> softmax [head <H>]  — save softmax attention for block N
  save block <N> <component>         — save any component (e.g. Q_0, W_o, SiLU)
  save all heads block <N>           — save all 32 head matrices for block N
  select block <N>                   — switch to block N
  select head <H>                    — switch displayed head
  run <text>                         — run simulation with text
  status                             — show current state
  list                               — list components
  help                               — show this help
  quit                               — exit
""".strip()

    def __init__(self):
        self.cmd_queue = queue.Queue()
        self._thread = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print(self.HELP)
        print("\ncmd> ", end="", flush=True)

    def stop(self):
        self._running = False

    def _read_loop(self):
        while self._running:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    self.cmd_queue.put(line)
            except (EOFError, KeyboardInterrupt):
                break

    def poll(self):
        """Return next command string or None."""
        try:
            return self.cmd_queue.get_nowait()
        except queue.Empty:
            return None

    @staticmethod
    def parse(cmd):
        """Parse a command string into an action dict."""
        cmd = cmd.strip().lower()

        if cmd in ("help", "?"):
            return {"action": "help"}
        if cmd in ("quit", "exit", "q"):
            return {"action": "quit"}
        if cmd == "status":
            return {"action": "status"}
        if cmd in ("list", "ls"):
            return {"action": "list"}

        # select block <N>
        m = re.match(r"select\s+block\s+(\d+)", cmd)
        if m:
            return {"action": "select_block", "block": int(m.group(1))}

        # select head <H>
        m = re.match(r"select\s+head\s+(\d+)", cmd)
        if m:
            return {"action": "select_head", "head": int(m.group(1))}

        # save all heads block <N>
        m = re.match(r"save\s+all\s+heads?\s+block\s+(\d+)", cmd)
        if m:
            return {"action": "save_all_heads", "block": int(m.group(1))}

        # save block <N> softmax [head <H>]
        m = re.match(r"save\s+block\s+(\d+)\s+(?:attention\s+)?softmax(?:\s+head\s+(\d+))?", cmd)
        if m:
            d = {"action": "save", "block": int(m.group(1)), "component": "Softmax"}
            if m.group(2):
                d["head"] = int(m.group(2))
            return d

        # save block <N> <component>
        m = re.match(r"save\s+block\s+(\d+)\s+(.+)", cmd)
        if m:
            return {"action": "save", "block": int(m.group(1)), "component": m.group(2).strip()}

        # run <text>
        m = re.match(r"run\s+(.+)", cmd)
        if m:
            return {"action": "run", "text": m.group(1)}

        return {"action": "unknown", "cmd": cmd}


# ── Inspector ───────────────────────────────────────────────

class Inspector:
    def __init__(self, model_path=None):
        pygame.init()
        info = pygame.display.Info()
        self.W = info.current_w
        self.H = info.current_h - 50
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.fonts = {
            "s": pygame.font.SysFont("Monaco", 11),
            "m": pygame.font.SysFont("Monaco", 13),
            "l": pygame.font.SysFont("Monaco", 16),
            "t": pygame.font.SysFont("Monaco", 20),
        }
        self.cam = IsoCamera(self.W, self.H)
        self.renderer = Renderer(self.screen, self.cam, self.fonts)
        self.top_bar = TopBar(self.fonts)
        self.left_panel = LeftPanel(260, self.fonts)
        self.right_panel = RightPanel(320, self.fonts)
        self.simulation = LiveSimulation(model_path)
        self.model_path = model_path
        self.sel = 0

        model_label = os.path.basename(model_path) if model_path else "No model"
        pygame.display.set_caption(f"Transformer Inspector — {model_label}")

        # Run ID and directory
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join("runs", self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"Run ID: {self.run_id}  ->  {self.run_dir}/")

        # Pre-build all 16 blocks (identical structure, different block index)
        self.blocks = []
        for b in range(LeftPanel.NUM_BLOCKS):
            builder = BlockBuilder()
            builder.build_standard_block()
            self.blocks.append((builder.components, builder.connections))

        self._load_block(0)

        # Command processor
        self.cmd = CommandProcessor()
        self.cmd.start()

    def _load_block(self, idx):
        self.comps, self.conns = self.blocks[idx]
        self.comp_rects = [None] * len(self.comps)
        self.sel = 0
        pygame.display.set_caption(f"LLaMA 3.2-1B — Block {idx} — Isometric Inspector")

    def handle_input(self):
        # Only process camera keys when text input is NOT focused
        if not self.top_bar.focused:
            keys = pygame.key.get_pressed()
            cam = self.cam
            if keys[pygame.K_w]:     cam.zoom *= 1.02
            if keys[pygame.K_s]:     cam.zoom = max(0.5, cam.zoom / 1.02)
            if keys[pygame.K_a]:     cam.px -= 5
            if keys[pygame.K_d]:     cam.px += 5
            if keys[pygame.K_z] or keys[pygame.K_UP]:    cam.py -= 5
            if keys[pygame.K_x] or keys[pygame.K_DOWN]:  cam.py += 5
            if keys[pygame.K_LEFT]:  cam.px -= 5
            if keys[pygame.K_RIGHT]: cam.px += 5
            if keys[pygame.K_q]:     cam.ay -= 1
            if keys[pygame.K_e]:     cam.ay += 1
            if keys[pygame.K_r]:     cam.ax = max(-89, min(89, cam.ax - 1))
            if keys[pygame.K_f]:     cam.ax = max(-89, min(89, cam.ax + 1))

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False

            # Top bar gets first crack at events
            action = self.top_bar.handle_event(ev)
            if action is not None:
                bi = self.left_panel.current_block
                if action["type"] == "submit":
                    self.simulation.reset_generation()
                    self.simulation.run(action["text"], block_idx=bi)
                elif action["type"] == "run_all":
                    self.simulation.reset_generation()
                    self.simulation.run_all(action["text"], action["max_tokens"])
                elif action["type"] == "step":
                    self.simulation.step(action["text"], action["max_tokens"])
                elif action["type"] == "reset":
                    self.top_bar.text = ""
                    self.top_bar.focused = False
                    self.simulation.reset_generation()
                    self.simulation.active = False
                    self.simulation.data = {}
                    self.simulation.text = ""
                    self.simulation.tokens = []
                    self.simulation.token_ids = []
                    self.sel = 0
                continue

            # Skip other key handling if text input focused
            if self.top_bar.focused or self.top_bar.max_tokens_focused:
                continue

            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return False
                elif ev.key == pygame.K_TAB:
                    self.sel = (self.sel + 1) % len(self.comps)
                elif ev.key == pygame.K_BACKQUOTE:
                    self.sel = (self.sel - 1) % len(self.comps)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    self._handle_click(*ev.pos)
                elif ev.button in (4, 5):
                    dy = 1 if ev.button == 4 else -1
                    # Scroll right panel if mouse is over it
                    if ev.pos[0] > self.W - self.right_panel.width:
                        self.right_panel.handle_scroll(dy)
                    else:
                        self.cam.zoom *= 1.1 if dy > 0 else 1 / 1.1
                        self.cam.zoom = max(0.5, self.cam.zoom)
            elif ev.type == pygame.VIDEORESIZE:
                self.W, self.H = ev.w, ev.h
                self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
                self.renderer.screen = self.screen
                self.cam.w, self.cam.h = self.W, self.H
        return True

    def _handle_click(self, mx, my):
        lw = self.left_panel.width
        rw = self.right_panel.width
        top = TopBar.HEIGHT

        if my < top:
            return  # handled by top_bar
        if mx < lw:
            result = self.left_panel.handle_click(mx, my)
            if result:
                if result[0] == "select":
                    self.sel = result[1]
                elif result[0] == "switch_block":
                    self._load_block(result[1])
                    # Re-run simulation on new block if active
                    if self.simulation.active and self.simulation.text:
                        self.simulation.run(self.simulation.text, block_idx=result[1])
            return
        if mx > self.W - rw:
            result = self.right_panel.handle_click(mx, my, self.simulation)
            if result == "save":
                comp = self.comps[self.sel]
                if comp.typ not in ("data", "skip") and self.simulation.active:
                    self.right_panel.save_data(
                        self.run_dir, comp, self.simulation,
                        self.left_panel.current_block)
            return

        for i, rect in enumerate(self.comp_rects):
            if rect and rect.collidepoint(mx, my):
                self.sel = i
                return

    def _project_components(self):
        proj = []
        for c in self.comps:
            sx, sy = self.cam.project_grid(c.gx, c.gy, c.gz)
            sw, sh = self.cam.world_size(c.gw, c.gh)
            proj.append((sx, sy, sw, sh))
        return proj

    def draw(self):
        lw = self.left_panel.width
        rw = self.right_panel.width
        top = TopBar.HEIGHT
        viewport_w = self.W - lw - rw

        # Offset camera for viewport area
        self.cam.w = viewport_w
        save_px = self.cam.px
        self.cam.px += lw // 2 - rw // 2

        self.screen.fill((22, 24, 30))

        # Clip to viewport for grid + components
        vp_rect = pygame.Rect(lw, top, viewport_w, self.H - top)
        self.screen.set_clip(vp_rect)

        self.renderer.draw_grid()
        proj = self._project_components()

        for conn in self.conns:
            self.renderer.draw_connection(conn, proj)

        for i, c in enumerate(self.comps):
            sx, sy, sw, sh = proj[i]
            self.comp_rects[i] = self.renderer.draw_component(c, sx, sy, sw, sh, i == self.sel)

        self.renderer.draw_section_label("ATTENTION", -7, 10, (50, 130, 70))
        self.renderer.draw_section_label("TRANSFORM", -7, -3, (190, 90, 30))

        self.screen.set_clip(None)

        self.cam.px = save_px
        self.cam.w = self.W

        # Panels
        self.top_bar.draw(self.screen, self.W)
        self.left_panel.draw(self.screen, self.comps, self.sel, top, self.H, self.simulation)
        self.right_panel.draw(self.screen, self.simulation, self.comps[self.sel],
                              self.W - rw, top, self.H)

        # Help bar
        self.screen.blit(self.fonts["s"].render(
            "ADZX: pan  W/S: zoom  Q/E: rot Y  R/F: rot X  Tab: next  Click: select",
            True, (55, 55, 65)), (lw + 10, self.H - 16))

    def _process_commands(self):
        """Process any pending CLI commands."""
        while True:
            raw = self.cmd.poll()
            if raw is None:
                break
            parsed = CommandProcessor.parse(raw)
            act = parsed["action"]

            if act == "help":
                print(CommandProcessor.HELP)

            elif act == "quit":
                return False

            elif act == "status":
                bi = self.left_panel.current_block
                sh = self.simulation.selected_head
                active = self.simulation.active
                text = self.simulation.text[:50] if self.simulation.text else "(none)"
                gen = len(self.simulation.generated_tokens)
                print(f"  Block: {bi}  Head: {sh}  Active: {active}")
                print(f"  Text: {text}")
                print(f"  Generated: {gen} tokens")
                print(f"  Run dir: {self.run_dir}")

            elif act == "list":
                print("  Components:")
                for c in self.comps:
                    if c.typ not in ("data", "skip"):
                        print(f"    {c.name:15s}  {c.typ:8s}  {c.dims}")

            elif act == "select_block":
                bi = parsed["block"]
                if 0 <= bi < LeftPanel.NUM_BLOCKS:
                    self.left_panel.current_block = bi
                    self._load_block(bi)
                    if self.simulation.active and self.simulation.text:
                        self.simulation.run(self.simulation.text, block_idx=bi)
                    print(f"  Switched to block {bi}")
                else:
                    print(f"  Invalid block: {bi} (0-{LeftPanel.NUM_BLOCKS-1})")

            elif act == "select_head":
                h = parsed["head"]
                if 0 <= h < 32:
                    self.simulation.selected_head = h
                    print(f"  Selected head {h}")
                else:
                    print(f"  Invalid head: {h} (0-31)")

            elif act == "run":
                text = parsed["text"]
                bi = self.left_panel.current_block
                self.simulation.reset_generation()
                self.simulation.run(text, block_idx=bi)
                print(f"  Running: \"{text}\" on block {bi}")
                print(f"  Tokens: {len(self.simulation.tokens)}")

            elif act == "save":
                bi = parsed["block"]
                comp_name = parsed["component"]
                if "head" in parsed:
                    self.simulation.selected_head = parsed["head"]
                # Switch block and re-run if needed
                if bi != self.left_panel.current_block:
                    self.left_panel.current_block = bi
                    self._load_block(bi)
                    if self.simulation.active and self.simulation.text:
                        self.simulation.run(self.simulation.text, block_idx=bi)

                if not self.simulation.active:
                    print("  No simulation active. Run text first.")
                else:
                    self._save_component(bi, comp_name)

            elif act == "save_all_heads":
                bi = parsed["block"]
                if bi != self.left_panel.current_block:
                    self.left_panel.current_block = bi
                    self._load_block(bi)
                    if self.simulation.active and self.simulation.text:
                        self.simulation.run(self.simulation.text, block_idx=bi)
                if not self.simulation.active:
                    print("  No simulation active. Run text first.")
                else:
                    # Save softmax with all head data
                    self._save_component(bi, "Softmax")
                    print(f"  All 32 heads saved for block {bi}")

            elif act == "unknown":
                print(f"  Unknown command: {parsed['cmd']}")
                print("  Type 'help' for commands.")

            print("cmd> ", end="", flush=True)
        return True

    def _save_component(self, block_idx, comp_name):
        """Save a component by name."""
        # Find the component
        comp = None
        # Try exact match first
        for c in self.comps:
            if c.name == comp_name:
                comp = c
                break
        # Try case-insensitive / partial match
        if comp is None:
            cn = comp_name.lower().replace(" ", "")
            for c in self.comps:
                if c.name.lower().replace(" ", "") == cn:
                    comp = c
                    break
        # Try matching just the start
        if comp is None:
            cn = comp_name.lower()
            for c in self.comps:
                if c.name.lower().startswith(cn):
                    comp = c
                    break

        if comp is None:
            print(f"  Component not found: {comp_name}")
            print(f"  Use 'list' to see available components.")
            return

        self.right_panel.save_data(
            self.run_dir, comp, self.simulation, block_idx)

    def run(self):
        while self.handle_input():
            if not self._process_commands():
                break
            self.draw()

            c = self.cam
            pygame.display.set_caption(
                f"Block {self.left_panel.current_block} | "
                f"ax={c.ax:.0f} ay={c.ay:.0f} zoom={c.zoom:.2f} px={c.px:.0f} py={c.py:.0f}")
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()


CLI_HELP = """
inspector.py — Interactive Transformer Block Inspector

USAGE:
  python inspector.py --model <path_or_hf_id>
  python inspector.py --model models/Llama-3.2-1B-Instruct
  python inspector.py --model Qwen/Qwen2-1.5B

OPTIONS:
  --model, -m <path>  Model path (local dir or HuggingFace ID). Required.
  --help, -h          Show this help and exit

SETUP:
  # Download a model first:
  python download_model.py meta-llama/Llama-3.2-1B-Instruct
  # Then inspect it:
  python inspector.py --model models/Llama-3.2-1B-Instruct

DESCRIPTION:
  Launches a pygame-based isometric visualizer for inspecting transformer
  block internals. Works with any HuggingFace causal LM. Captures real
  activations, attention patterns, and weight statistics per block/head.

  Model config (heads, layers, dims) is auto-detected from the loaded model.
  Each session gets a timestamped run ID. Saved data goes to runs/<run_id>/.

GUI CONTROLS:
  A/D, Left/Right    Pan left/right
  Z/X, Up/Down       Pan up/down
  W/S                 Zoom in/out
  Q/E                 Rotate Y axis
  R/F                 Rotate X axis
  Tab / `             Cycle component selection
  Click               Select component / toggle group / select head
  Mouse wheel         Zoom (viewport) / Scroll (right panel)
  ESC                 Quit

GUI ELEMENTS:
  Top bar             Text input + Enter, Max tokens, Run All, Step, Reset
  Left panel          Block selector, component tree, output box
  Center              Isometric 3D view of one transformer block
  Right panel         Context-dependent data: activations, heatmaps, Save

TERMINAL COMMANDS (type while inspector is running):
  save block <N> softmax [head <H>]   Save attention data for block N
  save block <N> <component>          Save component (e.g. Q_0, W_o, SiLU)
  save all heads block <N>            Save all 32 head matrices
  select block <N>                    Switch to block N
  select head <H>                     Switch displayed attention head
  run <text>                          Run forward pass with input text
  status                              Show current state
  list                                List available components
  help                                Show command help
  quit                                Exit

SAVED DATA (runs/<run_id>/block_<N>/<component>/):
  info.json           Metadata, tokens, stats, per-token max attention
  *.npy               Attention matrices, activations as numpy arrays
""".strip()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(CLI_HELP)
        sys.exit(0)

    model_path = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ("--model", "-m") and i < len(sys.argv) - 1:
            model_path = sys.argv[i + 1]

    if not model_path:
        print("Usage: python inspector.py --model <path_or_hf_id>")
        print("       python inspector.py --help")
        sys.exit(1)

    Inspector(model_path=model_path).run()
