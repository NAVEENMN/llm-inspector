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

    def build_manifold_block(self):
        """Build attention + manifold + transform in correct order."""
        # ── Attention (same as standard up to wV) ──
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

        # ── Manifold (CurveAttention components) ──
        self.add(name="AV blend", gx=2, gy=8, gw=3, gh=1,
                 color=(180, 60, 180), dims="weights @ V", typ="op", module="manifold")
        self.add(name="tau_proj", gx=-4, gy=7, gw=3, gh=1,
                 color=(200, 80, 200), dims="Q -> tau (64->64)", typ="weight", module="manifold")
        self.add(name="sigmoid", gx=-4, gy=6, gw=2, gh=1,
                 color=(220, 100, 180), dims="tau in [0,1]^R", typ="op", module="manifold")
        self.add(name="B-spline", gx=0, gy=6, gw=3, gh=1,
                 color=(160, 40, 200), dims="eval curve at tau", typ="op", module="manifold")
        self.add(name="sample_proj", gx=0, gy=5, gw=3, gh=1,
                 color=(200, 80, 200), dims="R -> head_dim (64->64)", typ="weight", module="manifold")

        self.connect("wV", "AV blend")
        self.connect("RoPE", "tau_proj")
        self.connect("tau_proj", "sigmoid")
        self.connect("sigmoid", "B-spline")
        self.connect("AV blend", "B-spline")
        self.connect("B-spline", "sample_proj")

        # ── W_o + residual ──
        self.add(name="W_o", gx=0, gy=4, gw=4, gh=2,
                 color=(50,90,210), dims="R^{2048x2048} (32h)", typ="weight", module="attention")
        self.add(name="+ res₁", gx=0, gy=2, gw=2, gh=1,
                 color=(190,160,50), dims="x + attn(x)", typ="op", module="attention")
        self.connect("sample_proj", "W_o")
        self.connect("W_o", "+ res₁")

        self.add_residual_path("res₁", "Input", "+ res₁", gx=8,
                               gy_top=20, gy_mid=10, gy_bot=2, gz=40)

        # ── Transform Module ──
        for name, gx, gy, gw, gh, color, dims, typ in [
            ("RMSNorm₂", 0,  0,  4, 1, (100,100,140), "R^2048 -> R^2048", "op"),
            ("W_gate",   -5, -2,  4, 2, (210,100,40),  "R^{2048x8192}",     "weight"),
            ("W_up",      5, -2,  4, 2, (210,120,40),  "R^{2048x8192}",     "weight"),
            ("SiLU",     -5, -4,  3, 1, (210,60,50),   "s(x)*x",            "op"),
            ("gate*up",   0, -5,  3, 1, (210,80,60),   "R^8192 . R^8192",   "op"),
            ("W_down",    0, -7,  4, 2, (210,100,40),  "R^{8192x2048}",     "weight"),
            ("+ res₂",   0, -9,  2, 1, (190,160,50),  "x + mlp(x)",        "op"),
        ]:
            self.add(name=name, gx=gx, gy=gy, gw=gw, gh=gh,
                     color=color, dims=dims, typ=typ, module="transform")

        self.add(name="Output", gx=0, gy=-11, gw=5, gh=1,
                 color=(80,80,110), dims="R^2048", typ="data")

        for src, dst in [("+ res₁", "RMSNorm₂"), ("RMSNorm₂", "W_gate"),
                         ("RMSNorm₂", "W_up"), ("W_gate", "SiLU"),
                         ("SiLU", "gate*up"), ("W_up", "gate*up"),
                         ("gate*up", "W_down"), ("W_down", "+ res₂"),
                         ("+ res₂", "Output")]:
            self.connect(src, dst)

        self.add_residual_path("res₂", "+ res₁", "+ res₂", gx=8,
                               gy_top=2, gy_mid=-4, gy_bot=-9, gz=60)
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
        self.selected_token = -1  # -1 = last token
        self._attn_stats = {}
        self.curve_data = {}  # head_idx → [per_token_curve_dict]

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
        nh = len(self.attn_mats)
        disp_tokens = getattr(self, 'display_tokens', None) or self.tokens
        sections = [
            ("Operation", [
                "w = softmax(scores, dim=-1)",
                f"{nh} heads, each {len(disp_tokens)}x{len(disp_tokens)}",
            ]),
            (f"Head {h} Attention (click below to change)", [], {
                "matrix": self.attn_mats[h], "tokens": disp_tokens,
                "label": f"Head {h} — real attention weights",
                "cmap": "heat"}),
        ]
        # Show heads in groups of 8
        for group_start in range(0, nh, 8):
            group_end = min(group_start + 8, nh)
            label = f"Heads {group_start}-{group_end-1}"
            if group_start == 0:
                label += " (click to inspect)"
            sections.append((label, [], {
                "multi": [self.attn_mats[i] for i in range(group_start, group_end)],
                "tokens": disp_tokens,
                "label": "Click a head to inspect" if group_start == 0 else "",
                "head_offset": group_start}))
        sections.append(("Statistics", [
                f"Selected head: {h}",
                f"Max attention: {self._attn_stats.get('max', 0):.4f}",
                f"Min non-zero:  {self._attn_stats.get('min_nonzero', 0):.6f}",
            ]))

        return sections

    def _build_curve_sections(self):
        """Build B-spline curve display sections dynamically."""
        h = self.selected_head
        tok_i = self.selected_token if self.selected_token >= 0 else self.seq_len - 1
        tok_i = min(tok_i, self.seq_len - 1)
        tok_label = self.tokens[tok_i].strip() if tok_i < len(self.tokens) else f"[{tok_i}]"

        sections = [
            ("B-spline Curves", [
                f"Head {h}, Token: \"{tok_label}\" (pos {tok_i})",
                f"{self.seq_len} tokens, {len(self.curve_data)} heads",
            ]),
        ]

        # Main curve plot for selected head + token
        if h in self.curve_data and tok_i < len(self.curve_data[h]):
            cd = self.curve_data[h][tok_i]
            sections.append((f"Head {h} Curve (click below to change)", [], {
                "curve": cd, "label": f"Head {h}, token \"{tok_label}\""}))

        # Small multiples: one curve per head (same token), 8 per row
        for row_start in range(0, min(32, len(self.curve_data)), 8):
            row_end = min(row_start + 8, len(self.curve_data))
            row_curves = []
            for hi in range(row_start, row_end):
                if hi in self.curve_data and tok_i < len(self.curve_data[hi]):
                    row_curves.append(self.curve_data[hi][tok_i])
                else:
                    row_curves.append(None)
            label = f"Heads {row_start}-{row_end-1}" if row_start > 0 else f"All Heads ({row_start}-{row_end-1})"
            sections.append((label, [], {
                "multi_curves": row_curves, "head_offset": row_start,
                "label": "Click a head to inspect"}))

        # Token selector hint
        tok_list = [f"{i}:{t.strip()[:6]}" for i, t in enumerate(self.tokens[:10])]
        sections.append(("Tokens", [
            "Select token: " + ", ".join(tok_list),
            f"Current: pos {tok_i} \"{tok_label}\"",
            "(Use 'select token N' in terminal)",
        ]))

        # Intervention comparison (if available)
        if self.original_predictions and self.modified_predictions:
            lines = ["Drag control points, click Apply",
                      "to see how predictions change.", ""]
            lines.append(f"{'Original':>14s} {'prob':>7s} | {'Modified':>14s} {'prob':>7s}")
            lines.append("-" * 48)
            for i in range(min(5, len(self.original_predictions))):
                ot, op = self.original_predictions[i]
                mt, mp = self.modified_predictions[i]
                marker = " <<" if ot.strip() != mt.strip() else ""
                lines.append(f"{ot.strip():>14s} {op:6.3%} | {mt.strip():>14s} {mp:6.3%}{marker}")

            # Show rank change for original top-1
            orig_top = self.original_predictions[0][0].strip()
            mod_rank = next((i for i, (t, _) in enumerate(self.modified_predictions)
                            if t.strip() == orig_top), -1)
            if mod_rank > 0:
                lines.append(f"")
                lines.append(f"Original top \"{orig_top}\" moved to rank {mod_rank+1}")
            elif mod_rank == 0:
                lines.append(f"")
                lines.append(f"Top prediction unchanged: \"{orig_top}\"")

            sections.append(("Intervention Result", lines))

        return sections

    def get_display(self, comp_name):
        if not self.active:
            return [("No Input", ["Enter text above and press",
                                   "Enter to simulate data flow."])]
        val = self.data.get(comp_name)
        if val == "ATTN_HEADS":
            return self._build_attn_sections()
        if val == "CURVE_VIEW":
            return self._build_curve_sections()
        if val == "ATTN_WV":
            h = self.selected_head
            disp_tok = getattr(self, 'display_tokens', None) or self.tokens
            return [
                ("Operation", [
                    "out = attn_weights @ V",
                    f"Using head {h} weights",
                ]),
                (f"Head {h} Attention Used", [], {
                    "matrix": self.attn_mats.get(h, []), "tokens": disp_tok,
                    "label": f"These weights multiplied V (head {h})",
                    "cmap": "heat"}),
            ]
        if val is None:
            return [("Info", [f"No simulation data for {comp_name}"])]
        return val


# ── Live Simulation (real model) ────────────────────────────

class LiveSimulation(Simulation):
    """Simulation backed by a real HuggingFace causal LM with hook-captured activations."""

    def __init__(self, model_path=None, checkpoint_path=None):
        super().__init__()
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path  # curve attention checkpoint
        self.model = None
        self.tokenizer = None
        self.device = None
        self._hooks = []
        self._captures = {}
        self._loading = False
        self.model_loaded = False
        self.is_manifold = False  # True if checkpoint loaded
        self.patched_blocks = set()
        self.interventions = {}  # (block, head, token) -> edited_ctrl_pts list
        self.original_predictions = None  # top-5 before intervention
        self.modified_predictions = None  # top-5 after intervention
        self.sensitivity = {}  # {block: {head: score}} — computed on demand
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

        # Detect if instruct model (has chat template)
        self.is_instruct = hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None
        mode_str = "instruct" if self.is_instruct else "base"

        print(f"Loaded on {self.device}: {self.num_layers} layers, "
              f"{self.num_q_heads}Q/{self.num_kv_heads}KV heads, "
              f"dim={self.hidden_dim}, ff={self.intermediate_dim} ({mode_str})")

        # Load curve attention checkpoint if provided
        if self.checkpoint_path:
            self._load_checkpoint()

    def _load_checkpoint(self):
        """Load CurveAttention checkpoint and patch the model."""
        import torch
        from prototype_curve_attention import CurveAttention

        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        ckpt = torch.load(self.checkpoint_path, weights_only=False, map_location=self.device)
        config = ckpt['config']
        R = config['R']
        hd = config['head_dim']

        nh = self.num_q_heads
        nkv = self.num_kv_heads
        ng = nh // nkv

        for block_idx_str, state in ckpt['curve_states'].items():
            block_idx = int(block_idx_str)
            ca = CurveAttention(hd, R).to(self.device)
            ca.load_state_dict(state)
            ca.eval()

            # Patch the attention forward
            attn = self.model.model.layers[block_idx].self_attn
            attn.curve_attention = ca

            def make_fwd(am, ca_mod, _nh, _nkv, _ng, _hd):
                def fwd(hidden_states, position_embeddings=None, attention_mask=None,
                        past_key_values=None, cache_position=None, **kw):
                    bsz, ql, _ = hidden_states.size()
                    q = am.q_proj(hidden_states).view(bsz, ql, _nh, _hd).transpose(1, 2)
                    k = am.k_proj(hidden_states).view(bsz, ql, _nkv, _hd).transpose(1, 2)
                    v = am.v_proj(hidden_states).view(bsz, ql, _nkv, _hd).transpose(1, 2)
                    cos, sin = position_embeddings
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                    if _ng > 1:
                        k = k.repeat_interleave(_ng, dim=1)
                        v = v.repeat_interleave(_ng, dim=1)
                    mask = attention_mask[:, :, :, :k.shape[-2]] if attention_mask is not None else None
                    out = ca_mod(q, k, v, mask=mask)
                    out = out.transpose(1, 2).reshape(bsz, ql, -1)
                    return am.o_proj(out), None
                return fwd

            attn.forward = make_fwd(attn, ca, nh, nkv, ng, hd)
            self.patched_blocks.add(block_idx)

        self.is_manifold = True
        print(f"Manifold: loaded CurveAttention for {len(self.patched_blocks)} blocks "
              f"(R={R}, head_dim={hd})")

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

        # Manifold hooks: capture tau and sample_proj from CurveAttention
        if self.is_manifold and block_idx in self.patched_blocks:
            ca = layer.self_attn.curve_attention

            def tau_proj_hook(module, inp, out):
                captures["tau_logits"] = out.detach().cpu()
                captures["tau"] = torch.sigmoid(out).detach().cpu()

            def sample_proj_hook(module, inp, out):
                captures["curve_samples"] = inp[0].detach().cpu()
                captures["sample_proj_out"] = out.detach().cpu()

            self._hooks.append(ca.tau_proj.register_forward_hook(tau_proj_hook))
            self._hooks.append(ca.sample_proj.register_forward_hook(sample_proj_hook))

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _encode(self, text):
        """Encode text, applying chat template for instruct models.
        Sets self.prompt_start and self.prompt_end for display slicing."""
        if self.is_instruct:
            messages = [{"role": "user", "content": text}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            full_ids = self.tokenizer.encode(formatted, return_tensors="pt")
            # Find where user text starts and ends
            user_ids = self.tokenizer.encode(text, add_special_tokens=False)
            full_list = full_ids[0].tolist()
            self.prompt_start = 0
            self.prompt_end = len(full_list)
            for start in range(len(full_list) - len(user_ids) + 1):
                if full_list[start:start + len(user_ids)] == user_ids:
                    self.prompt_start = start
                    self.prompt_end = start + len(user_ids)
                    break
            return full_ids
        self.prompt_start = 0
        self.prompt_end = None  # means use all
        return self.tokenizer.encode(text, return_tensors="pt")

    def _run_with_ids(self, ids, original_text, block_idx=0):
        """Run with pre-encoded IDs (avoids double-encoding for run_all)."""
        import torch
        self.text = original_text.strip()
        ids = ids.to(self.device)
        self.token_ids = ids[0].tolist()
        self.tokens = [self.tokenizer.decode([tid]) for tid in self.token_ids]

        # Use the prompt_start/end from the original _encode call
        self.display_start = getattr(self, 'prompt_start', 0)
        self.display_end = len(self.tokens)  # include generated tokens
        self.display_tokens = self.tokens[self.display_start:]
        if self.display_start > 0:
            print(f"  [instruct] Showing tokens {self.display_start}+ "
                  f"of {len(self.tokens)} (prompt: {self.display_tokens[:5]}...)")
        self.seq_len = len(self.tokens)
        self.active = True

        self._install_hooks(block_idx)
        with torch.no_grad():
            outputs = self.model(ids)
        self._remove_hooks()

        logits = outputs.logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        top_ids = torch.topk(probs, 5)
        self._top_tokens = [(self.tokenizer.decode([tid]), p.item())
                            for tid, p in zip(top_ids.indices, top_ids.values)]
        self._build_display(block_idx)

    def run(self, text, block_idx=0):
        import torch
        self._ensure_model()
        self.text = text.strip()
        if not self.text:
            self.active = False
            self.data = {}
            return

        ids = self._encode(self.text).to(self.device)
        self.token_ids = ids[0].tolist()
        self.tokens = [self.tokenizer.decode([tid]) for tid in self.token_ids]

        # For display: only show user prompt tokens (skip system template + suffix)
        self.display_start = getattr(self, 'prompt_start', 0)
        self.display_end = getattr(self, 'prompt_end', None) or len(self.tokens)
        self.display_tokens = self.tokens[self.display_start:self.display_end]
        if self.display_start > 0:
            print(f"  [instruct] Showing tokens {self.display_start}:{self.display_end} "
                  f"of {len(self.tokens)} ({self.display_tokens})")
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

        hidden_dim = self.model.config.hidden_size
        for prefix, key, n_heads in [("Q", "q_raw", nh), ("K", "k_raw", nkv), ("V", "v_raw", nkv)]:
            if key in cap:
                raw = cap[key]  # (1, seq, total_dim)
                tensor = raw[0].view(S, n_heads, hd)  # (seq, heads, dim)
                for h in range(n_heads):
                    name = f"{prefix}_{subscript(h)}"
                    head_data = tensor[:, h, :]  # (seq, head_dim)
                    stats_mean = head_data.mean().item()
                    stats_std = head_data.std().item()
                    d[name] = [
                        ("Weight Matrix (FIXED)", [
                            f"Shape: ({hidden_dim}, {hd})",
                            f"Head {h} of {n_heads}",
                        ]),
                        ("Activations (REAL)", [
                            f"Output: (1, {S}, {hd})",
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

            # Slice attention to show only user prompt tokens
            ds = getattr(self, 'display_start', 0)
            de = getattr(self, 'display_end', S)
            disp_tokens = self.tokens[ds:de]

            score_mat = []
            for i in range(ds, de):
                row = []
                for j in range(ds, de):
                    v = scores[0, i, j].item()
                    row.append(None if v == float('-inf') else v)
                score_mat.append(row)

            d["QKt/d"] = [
                ("Operation", [
                    f"scores = Q * Kt / sqrt({hd})",
                    f"Block {block_idx}, causal masked",
                ]),
                ("Activations (REAL)", [
                    f"Shape: (1, {nh}, {S}, {S})",
                ]),
                ("Score Heatmap (head 0)", [], {
                    "matrix": score_mat, "tokens": disp_tokens,
                    "label": "Raw scores (pre-softmax)", "cmap": "diverge"}),
            ]

            # Attention weights — show only user prompt region
            self.attn_mats = {}
            for h in range(nh):
                mat = []
                for i in range(ds, de):
                    row = [attn_w[h, i, j].item() for j in range(ds, de)]
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

        # ── Manifold components ──
        if self.is_manifold and block_idx in self.patched_blocks:
            import torch
            ca = self.model.model.layers[block_idx].self_attn.curve_attention

            # Compute blended V (control points after attention weighting)
            if hasattr(self, 'attn_mats') and self.attn_mats and "v_raw" in cap:
                v_tensor = cap["v_raw"][0].view(S, nkv, hd).transpose(0, 1).float()
                v_exp = v_tensor.repeat_interleave(ng, dim=0)  # (nh, S, hd)
                # Use real attention weights
                aw_tensor = attn_w if 'attn_w' in dir() else None
                if aw_tensor is not None:
                    blended = torch.matmul(aw_tensor, v_exp)  # (nh, S, hd)
                else:
                    blended = v_exp  # fallback

                # Evaluate curves at dense points for visualization
                n_plot = 100
                dev = self.device  # model device (mps/cuda/cpu)
                t_dense = torch.linspace(0.01, 0.99, n_plot).to(dev)
                self.curve_data = {}

                for h in range(nh):
                    head_curves = []
                    for tok_i in range(S):
                        coeffs = blended[h, tok_i].to(dev)  # move to model device
                        curve_vals = ca.eval_curve(
                            coeffs.unsqueeze(0).float(),
                            t_dense.unsqueeze(0).float()
                        ).squeeze(0)
                        head_curves.append({
                            "t": t_dense.cpu().tolist(),
                            "y": curve_vals.cpu().tolist(),
                            "ctrl_pts": coeffs.cpu().tolist(),
                        })
                    self.curve_data[h] = head_curves

                # Get tau per head per token
                if "tau" in cap:
                    tau_all = cap["tau"]  # (1, nh, S, R) or similar
                    tau_flat = tau_all.view(nh, S, -1) if tau_all.dim() >= 3 else tau_all
                    for h in range(min(nh, len(self.curve_data))):
                        for tok_i in range(min(S, len(self.curve_data[h]))):
                            if tau_flat.dim() == 3 and h < tau_flat.shape[0] and tok_i < tau_flat.shape[1]:
                                self.curve_data[h][tok_i]["tau"] = tau_flat[h, tok_i].cpu().tolist()

            d["AV blend"] = [
                ("Operation", ["blended = attn_weights @ V",
                               "V coefficients as B-spline control points"]),
            ]

            if "tau" in cap:
                tau = cap["tau"]
                tau_logits = cap["tau_logits"]
                d["tau_proj"] = [
                    ("Weight Matrix (LEARNED)", [
                        f"Shape: ({self.head_dim}, {self.head_dim})",
                        f"Params: {ca.tau_proj.weight.numel() + ca.tau_proj.bias.numel():,}",
                        f"W mean: {ca.tau_proj.weight.mean().item():.6f}",
                        f"W std:  {ca.tau_proj.weight.std().item():.6f}",
                        f"W max:  {ca.tau_proj.weight.abs().max().item():.6f}",
                    ]),
                    ("Tau Logits (REAL)", [
                        f"Shape: {list(tau_logits.shape)}",
                    ] + self._fmt_tensor(tau_logits.view(-1, tau_logits.shape[-1]))),
                ]

                d["sigmoid"] = [
                    ("Operation", ["tau = sigmoid(logits)", "Maps to [0,1] sampling range"]),
                    ("Tau Values (REAL)", [
                        f"Shape: {list(tau.shape)}",
                        f"Min: {tau.min().item():.4f}",
                        f"Max: {tau.max().item():.4f}",
                        f"Mean: {tau.mean().item():.4f}",
                        f"Std:  {tau.std().item():.4f}",
                    ] + self._fmt_tensor(tau.view(-1, tau.shape[-1]))),
                ]

            # B-spline uses sentinel for dynamic head/token selection
            d["B-spline"] = "CURVE_VIEW"

            if "sample_proj_out" in cap:
                sp = cap["sample_proj_out"]
                d["sample_proj"] = [
                    ("Weight Matrix (LEARNED)", [
                        f"Shape: ({self.head_dim}, {ca.R})",
                        f"Params: {ca.sample_proj.weight.numel() + ca.sample_proj.bias.numel():,}",
                        f"Init: pinv(B) for identity at init",
                    ]),
                    ("Output (REAL)", [
                        f"Shape: {list(sp.shape)}",
                    ] + self._fmt_tensor(sp.view(-1, sp.shape[-1]))),
                ]

        # Next token prediction
        d["_next_token"] = self._top_tokens

        self.data = d

    def compute_sensitivity(self):
        """Compute which (layer, head) spline curves are most sensitive.

        For each block, perturbs each head's control points slightly
        and measures the change in next-token prediction. Fast — one
        forward pass per block (not per head).

        Stores results in self.sensitivity and prints ranked list.
        """
        import torch
        if not self.is_manifold or not self.text:
            print("  Need manifold model + text first.")
            return

        self._ensure_model()
        ids = self._encode(self.text).to(self.device)
        nh = self.model.config.num_attention_heads
        hd = self.model.config.head_dim
        eps = 0.1  # perturbation magnitude

        # Baseline: clean forward pass
        with torch.no_grad():
            base_logits = self.model(ids, use_cache=False).logits[0, -1, :]
            base_probs = torch.softmax(base_logits.float(), dim=-1)

        self.sensitivity = {}
        all_scores = []

        print(f"\n  Computing sensitivity ({len(self.patched_blocks)} blocks, {nh} heads)...")

        for bi in sorted(self.patched_blocks):
            ca = self.model.model.layers[bi].self_attn.curve_attention
            orig_eval = ca.eval_curve
            self.sensitivity[bi] = {}

            for h in range(nh):
                # Perturb head h's control points
                def make_perturbed(orig, head, epsilon):
                    def perturbed(coeffs, tau_01):
                        if coeffs.dim() == 4 and coeffs.shape[1] > head:
                            c = coeffs.clone()
                            c[:, head, :, :] += epsilon
                            return orig(c, tau_01)
                        return orig(coeffs, tau_01)
                    return perturbed

                ca.eval_curve = make_perturbed(orig_eval, h, eps)

                with torch.no_grad():
                    pert_logits = self.model(ids, use_cache=False).logits[0, -1, :]
                    pert_probs = torch.softmax(pert_logits.float(), dim=-1)

                # KL divergence as sensitivity measure
                kl = (base_probs * (base_probs.log() - pert_probs.log())).sum().item()
                score = abs(kl) / eps
                self.sensitivity[bi][h] = score
                all_scores.append((bi, h, score))

            ca.eval_curve = orig_eval

        # Rank and print
        all_scores.sort(key=lambda x: x[2], reverse=True)

        print(f"\n  {'Rank':>4s}  {'Layer':>5s}  {'Head':>4s}  {'Sensitivity':>11s}  {'Bar'}")
        print(f"  {'─'*4}  {'─'*5}  {'─'*4}  {'─'*11}  {'─'*20}")

        max_score = all_scores[0][2] if all_scores else 1.0
        for rank, (bi, h, score) in enumerate(all_scores[:20]):
            bar_len = int(20 * score / max_score)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            print(f"  {rank+1:>4d}  {bi:>5d}  {h:>4d}  {score:>11.6f}  {bar}")

        if all_scores:
            top_bi, top_h, _ = all_scores[0]
            print(f"\n  Most sensitive: Layer {top_bi}, Head {top_h}")
            print(f"  Select it: click block {top_bi}, then head {top_h}")

        return all_scores

    def _apply_all_interventions(self):
        """Patch all blocks that have interventions. Returns list of (ca, orig_eval) to restore."""
        import torch
        if not self.interventions:
            return []

        # Group interventions by block
        by_block = {}
        for (bi, h, ti), coeffs in self.interventions.items():
            by_block.setdefault(bi, []).append((h, ti, coeffs))

        patches = []
        for bi, edits in by_block.items():
            if bi >= len(self.model.model.layers):
                continue
            ca = self.model.model.layers[bi].self_attn.curve_attention
            orig_eval = ca.eval_curve
            dev = self.device

            # Build lookup: (head, token) -> edited tensor
            edit_map = {}
            for h, ti, coeffs in edits:
                edit_map[(h, ti)] = torch.tensor(coeffs, dtype=torch.float32).to(dev)

            def make_patched(_orig, _edit_map):
                def patched_eval(coeffs, tau_01):
                    if coeffs.dim() < 3:
                        return _orig(coeffs, tau_01)
                    c = coeffs.clone()
                    for (h, ti), ed in _edit_map.items():
                        if c.shape[-3] > h and c.shape[-2] > ti:
                            c[..., h, ti, :] = ed
                    return _orig(c, tau_01)
                return patched_eval

            ca.eval_curve = make_patched(orig_eval, edit_map)
            patches.append((ca, orig_eval))

        n = sum(len(v) for v in by_block.values())
        print(f"  [intervention] {n} edits across {len(by_block)} blocks applied")
        return patches

    @staticmethod
    def _restore_patches(patches):
        for ca, orig in patches:
            ca.eval_curve = orig

    def step(self, text, max_tokens):
        if self.gen_step >= max_tokens:
            return None
        import torch
        self._ensure_model()

        # Build IDs: template + user text + generated so far
        ids = self._encode(text).to(self.device)
        if self.generated_tokens:
            gen_ids = self.tokenizer.encode(
                " ".join(self.generated_tokens), add_special_tokens=False,
                return_tensors="pt").to(self.device)
            ids = torch.cat([ids, gen_ids], dim=-1)

        patches = self._apply_all_interventions()

        with torch.no_grad():
            logits = self.model(ids).logits[0, -1]

        self._restore_patches(patches)

        next_id = logits.argmax().item()
        tok = self.tokenizer.decode([next_id])
        self.generated_tokens.append(tok.strip())
        self.gen_step += 1

        # Append new token to IDs and re-run display (no double-encoding)
        new_ids = torch.cat([ids, torch.tensor([[next_id]], device=self.device)], dim=-1)
        self._run_with_ids(new_ids, text, block_idx=0)
        return tok.strip()

    def run_all(self, text, max_tokens):
        import torch
        self._ensure_model()
        self.generated_tokens = []
        self.gen_step = 0
        ids = self._encode(text).to(self.device)
        prompt_len = ids.shape[1]

        patches = self._apply_all_interventions()

        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=max_tokens,
                do_sample=False, use_cache=False,
                pad_token_id=self.tokenizer.eos_token_id)

        self._restore_patches(patches)

        gen_ids = out[0, prompt_len:]
        self.generated_tokens = [self.tokenizer.decode([tid]).strip() for tid in gen_ids]
        self.gen_step = len(self.generated_tokens)
        # Run full output through hooks — pass raw IDs to avoid double-encoding
        self._run_with_ids(out, text, block_idx=0)

    def run_with_intervention(self, text, block_idx, head, token_idx, edited_coeffs):
        """Re-run forward pass with and without intervention. Returns (orig, mod) top predictions."""
        import torch
        self._ensure_model()

        ids = self._encode(text).to(self.device)

        # 1. Original forward (clean)
        with torch.no_grad():
            orig_out = self.model(ids)
        orig_logits = orig_out.logits[0, -1]
        orig_probs = torch.softmax(orig_logits, dim=-1)
        orig_top = torch.topk(orig_probs, 10)
        self.original_predictions = [
            (self.tokenizer.decode([tid]), p.item())
            for tid, p in zip(orig_top.indices, orig_top.values)]

        # 2. Modified forward (all interventions)
        patches = self._apply_all_interventions()
        with torch.no_grad():
            mod_out = self.model(ids)
        self._restore_patches(patches)

        mod_logits = mod_out.logits[0, -1]
        mod_probs = torch.softmax(mod_logits, dim=-1)
        mod_top = torch.topk(mod_probs, 10)
        self.modified_predictions = [
            (self.tokenizer.decode([tid]), p.item())
            for tid, p in zip(mod_top.indices, mod_top.values)]

        # Print comparison
        print(f"\n  === Intervention: block {block_idx}, head {head}, token {token_idx} ===")
        print(f"  {'Original':>20s}  {'prob':>7s}  |  {'Modified':>20s}  {'prob':>7s}")
        print(f"  {'-'*20}  {'-'*7}  |  {'-'*20}  {'-'*7}")
        for i in range(min(5, len(self.original_predictions))):
            ot, op = self.original_predictions[i]
            mt, mp = self.modified_predictions[i]
            marker = " *" if ot.strip() != mt.strip() else ""
            print(f"  {ot:>20s}  {op:6.3%}  |  {mt:>20s}  {mp:6.3%}{marker}")
        print()

        return self.original_predictions, self.modified_predictions


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
    MODULE_COLORS = {"attention": (50, 130, 70), "transform": (190, 100, 40), "manifold": (160, 50, 200)}
    MODULE_LABELS = {"attention": "Attention Module", "transform": "Transform Module", "manifold": "Manifold Module"}
    HEAD_COUNTS = {"Q": 32, "K": 8, "V": 8}
    TYPE_COLORS = {"weight": (100,150,255), "op": (100,210,100), "data": (210,210,100)}

    def __init__(self, width, fonts):
        self.width = width
        self.fonts = fonts
        self.module_expanded = {"attention": False, "transform": False, "manifold": False}
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
        model_name = self.model_label if hasattr(self, 'model_label') else "Model"
        screen.blit(self.fonts["t"].render(model_name, True, (220, 220, 230)), (12, y))
        y += 26

        # Block selector — grid of numbered buttons
        screen.blit(self.fonts["m"].render("Block:", True, (150, 150, 160)), (12, y + 2))
        self.click_zones = []
        bx = 60
        btn_w = 38
        btn_h = 22
        cols = 4
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
        # Curve editor state
        self.editing = False
        self.drag_ctrl_idx = -1  # which control point is being dragged
        self.edited_ctrl_pts = None  # list of floats, or None
        self.original_ctrl_pts = None
        self.plot_bounds = None  # (plot_x, plot_y, plot_w, plot_h, y_min, y_max)
        self.apply_rect = pygame.Rect(0, 0, 0, 0)
        self.reset_edit_rect = pygame.Rect(0, 0, 0, 0)
        self.run_edit_rect = pygame.Rect(0, 0, 0, 0)
        self.save_edits_rect = pygame.Rect(0, 0, 0, 0)
        self.load_edits_rect = pygame.Rect(0, 0, 0, 0)
        self.edit_dirty = False
        self.edits_flash = 0
        self.current_block = 0
        # Edit file picker
        self.file_picker_open = False
        self.file_picker_items = []  # list of (path, label)
        self.file_picker_rects = []  # click zones
        self.active_edits_label = ""  # shows loaded file name

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

    def _draw_curve_plot(self, screen, x_offset, y, curve_data, width=None, height=140):
        """Draw a single B-spline curve plot with editable control points. Returns new y."""
        w = (width or self.width - 32)
        plot_x = x_offset + 16
        plot_y = y
        plot_w = w
        plot_h = height

        # Background
        pygame.draw.rect(screen, (32, 34, 42), (plot_x, plot_y, plot_w, plot_h))
        pygame.draw.rect(screen, (50, 52, 62), (plot_x, plot_y, plot_w, plot_h), 1)

        t_vals = curve_data.get("t", [])
        y_vals = curve_data.get("y", [])
        orig_ctrl = curve_data.get("ctrl_pts", [])
        tau_vals = curve_data.get("tau", [])

        # Use edited control points if available, otherwise original
        ctrl_pts = self.edited_ctrl_pts if self.edited_ctrl_pts is not None else list(orig_ctrl)
        if self.original_ctrl_pts is None and orig_ctrl:
            self.original_ctrl_pts = list(orig_ctrl)

        if not t_vals or not y_vals:
            return y + plot_h + 8

        # Compute y range from both original and edited
        all_y = y_vals + ctrl_pts + (self.original_ctrl_pts or [])
        y_min = min(all_y) if all_y else -1
        y_max = max(all_y) if all_y else 1
        y_range = y_max - y_min if y_max > y_min else 1.0
        margin = y_range * 0.15
        y_min -= margin
        y_max += margin
        y_range = y_max - y_min

        # Store bounds for mouse mapping
        self.plot_bounds = (plot_x, plot_y, plot_w, plot_h, y_min, y_max)

        def map_x(t):
            return int(plot_x + t * plot_w)

        def map_y(v):
            return int(plot_y + plot_h - (v - y_min) / y_range * plot_h)

        # Zero line
        if y_min < 0 < y_max:
            zy = map_y(0)
            pygame.draw.line(screen, (45, 47, 55), (plot_x, zy), (plot_x + plot_w, zy), 1)

        # Draw ORIGINAL curve as ghost if edited
        if self.edited_ctrl_pts is not None and self.original_ctrl_pts:
            pts = [(map_x(t), map_y(v)) for t, v in zip(t_vals, y_vals)]
            if len(pts) > 1:
                pygame.draw.lines(screen, (60, 80, 100), False, pts, 1)

        # Draw CURRENT curve (edited or original)
        if self.edited_ctrl_pts is not None and hasattr(self, '_edited_curve_y') and self._edited_curve_y:
            pts = [(map_x(t), map_y(v)) for t, v in zip(t_vals, self._edited_curve_y)]
        else:
            pts = [(map_x(t), map_y(v)) for t, v in zip(t_vals, y_vals)]
        if len(pts) > 1:
            color = (50, 255, 120) if self.edited_ctrl_pts is not None else (100, 200, 255)
            pygame.draw.lines(screen, color, False, pts, 2)

        # Draw control points (draggable)
        n_ctrl = len(ctrl_pts)
        self._ctrl_screen_pts = []
        if n_ctrl > 1:
            for ci, cv in enumerate(ctrl_pts):
                cx = map_x(ci / (n_ctrl - 1))
                cy = map_y(cv)
                self._ctrl_screen_pts.append((cx, cy))
                is_dragging = ci == self.drag_ctrl_idx
                is_edited = self.edited_ctrl_pts is not None
                r = 6 if is_dragging else 4
                color = (255, 255, 100) if is_dragging else (
                    (50, 255, 120) if is_edited else (120, 140, 200))
                pygame.draw.circle(screen, color, (cx, cy), r)
                pygame.draw.circle(screen, (200, 200, 220), (cx, cy), r, 1)

        # Draw original control points as ghosts if edited
        if self.edited_ctrl_pts is not None and self.original_ctrl_pts:
            n_orig = len(self.original_ctrl_pts)
            for ci, cv in enumerate(self.original_ctrl_pts):
                cx = map_x(ci / (n_orig - 1))
                cy = map_y(cv)
                pygame.draw.circle(screen, (60, 60, 80), (cx, cy), 3)

        # Draw tau positions
        if tau_vals:
            for ti, tv in enumerate(tau_vals):
                tx = map_x(tv)
                pygame.draw.line(screen, (200, 100, 50), (tx, plot_y + 2), (tx, plot_y + plot_h - 2), 1)
                if t_vals:
                    # Use edited curve if available
                    use_y = self._edited_curve_y if (self.edited_ctrl_pts and hasattr(self, '_edited_curve_y') and self._edited_curve_y) else y_vals
                    idx = min(range(len(t_vals)), key=lambda i: abs(t_vals[i] - tv))
                    sy = map_y(use_y[idx])
                    pygame.draw.circle(screen, (255, 150, 50), (tx, sy), 4)

        # Axes labels
        screen.blit(self.fonts["s"].render(f"{y_min:.2f}", True, (80, 80, 90)),
                     (plot_x + 2, plot_y + plot_h - 12))
        screen.blit(self.fonts["s"].render(f"{y_max:.2f}", True, (80, 80, 90)),
                     (plot_x + 2, plot_y + 1))
        screen.blit(self.fonts["s"].render("0", True, (70, 70, 80)), (plot_x + 2, plot_y + plot_h + 2))
        screen.blit(self.fonts["s"].render("1", True, (70, 70, 80)),
                     (plot_x + plot_w - 8, plot_y + plot_h + 2))

        # "Drag control points to edit" hint
        hint = "EDITED — drag points, Apply to re-run" if self.edited_ctrl_pts else "Drag control points to edit"
        screen.blit(self.fonts["s"].render(hint, True, (100, 100, 120)),
                     (plot_x + 2, plot_y + plot_h + 14))
        y = plot_y + plot_h + 28

        # ── Toolbar: always visible ──
        bx = x_offset + 16

        # Row 1: Load / Apply All / Run / Reset
        self.load_edits_rect = pygame.Rect(bx, y, 75, 22)
        pygame.draw.rect(screen, (80, 80, 120), self.load_edits_rect, border_radius=4)
        pygame.draw.rect(screen, (100, 100, 150), self.load_edits_rect, 1, border_radius=4)
        lbl = self.fonts["m"].render("Load", True, (200, 200, 220))
        screen.blit(lbl, (self.load_edits_rect.x + 18, self.load_edits_rect.y + 3))

        has_interventions = hasattr(self, '_sim_ref') and self._sim_ref and self._sim_ref.interventions
        has_edits = self.edited_ctrl_pts is not None or has_interventions

        self.apply_rect = pygame.Rect(bx + 80, y, 55, 22)
        ac = (40, 140, 60) if has_edits else (45, 47, 55)
        pygame.draw.rect(screen, ac, self.apply_rect, border_radius=4)
        lbl = self.fonts["m"].render("Apply", True, (220, 255, 220) if has_edits else (80, 80, 90))
        screen.blit(lbl, (self.apply_rect.x + 6, self.apply_rect.y + 3))

        self.run_edit_rect = pygame.Rect(bx + 140, y, 55, 22)
        rc = (60, 100, 200) if has_edits else (45, 47, 55)
        pygame.draw.rect(screen, rc, self.run_edit_rect, border_radius=4)
        lbl = self.fonts["m"].render("Run", True, (220, 230, 255) if has_edits else (80, 80, 90))
        screen.blit(lbl, (self.run_edit_rect.x + 12, self.run_edit_rect.y + 3))

        self.reset_edit_rect = pygame.Rect(bx + 200, y, 55, 22)
        pygame.draw.rect(screen, (140, 50, 40), self.reset_edit_rect, border_radius=4)
        lbl = self.fonts["m"].render("Reset", True, (255, 220, 220))
        screen.blit(lbl, (self.reset_edit_rect.x + 6, self.reset_edit_rect.y + 3))
        y += 26

        # Row 2: Save / status
        self.save_edits_rect = pygame.Rect(bx, y, 75, 22)
        if self.edits_flash > 0:
            pygame.draw.rect(screen, (40, 150, 60), self.save_edits_rect, border_radius=4)
            lbl = self.fonts["m"].render("Saved!", True, (220, 255, 220))
            self.edits_flash -= 1
        else:
            sc = (80, 80, 120) if has_edits else (45, 47, 55)
            pygame.draw.rect(screen, sc, self.save_edits_rect, border_radius=4)
            lbl = self.fonts["m"].render("Save", True, (200, 200, 220) if has_edits else (80, 80, 90))
        screen.blit(lbl, (self.save_edits_rect.x + 18, self.save_edits_rect.y + 3))

        if self.active_edits_label:
            screen.blit(self.fonts["s"].render(self.active_edits_label, True, (100, 200, 140)),
                         (bx + 82, y + 4))
        y += 26

        # File picker popup
        if self.file_picker_open:
            self.file_picker_rects = []
            pw = self.width - 32
            ph = len(self.file_picker_items) * 22 + 8
            pygame.draw.rect(screen, (35, 38, 50), (bx, y, pw, ph), border_radius=4)
            pygame.draw.rect(screen, (70, 80, 110), (bx, y, pw, ph), 1, border_radius=4)
            fy = y + 4
            for path, label in self.file_picker_items:
                r = pygame.Rect(bx + 4, fy, pw - 8, 20)
                mx_now, my_now = pygame.mouse.get_pos()
                hover = r.collidepoint(mx_now, my_now)
                if hover:
                    pygame.draw.rect(screen, (50, 60, 80), r, border_radius=3)
                screen.blit(self.fonts["s"].render(label, True, (220, 220, 230) if hover else (160, 160, 170)),
                             (bx + 8, fy + 3))
                self.file_picker_rects.append((r, path))
                fy += 22
            y += ph + 4

        return y + 4

    def _draw_multi_curves(self, screen, x_offset, y, hm_data):
        """Draw small multiples of curves. Returns new y."""
        curves = hm_data.get("multi_curves", [])
        head_offset = hm_data.get("head_offset", 0)
        label = hm_data.get("label", "")
        selected = getattr(self, '_sel_head', 0)

        n = len(curves)
        if n == 0:
            return y

        per_w = min(60, (self.width - 40) // n - 4)
        per_h = 40

        # Head labels
        for hi, cd in enumerate(curves):
            hx = x_offset + 16 + hi * (per_w + 4)
            abs_head = head_offset + hi
            is_sel = abs_head == selected
            lc = (255, 220, 50) if is_sel else (100, 100, 120)
            screen.blit(self.fonts["s"].render(f"h{abs_head}", True, lc),
                         (hx + per_w // 2 - 7, y))
        y += 13

        # Draw mini curves
        for hi, cd in enumerate(curves):
            hx = x_offset + 16 + hi * (per_w + 4)
            abs_head = head_offset + hi
            is_sel = abs_head == selected

            # Background
            bg = (40, 42, 55) if not is_sel else (50, 55, 70)
            pygame.draw.rect(screen, bg, (hx, y, per_w, per_h))
            if is_sel:
                pygame.draw.rect(screen, (255, 220, 50), (hx, y, per_w, per_h), 1)

            if cd is None:
                continue

            t_vals = cd.get("t", [])
            y_vals = cd.get("y", [])
            if len(t_vals) < 2:
                continue

            y_min, y_max = min(y_vals), max(y_vals)
            yr = y_max - y_min if y_max > y_min else 1.0

            pts = []
            for t, v in zip(t_vals, y_vals):
                px = int(hx + t * per_w)
                py = int(y + per_h - (v - y_min) / yr * per_h)
                pts.append((px, py))
            if len(pts) > 1:
                color = (100, 200, 255) if is_sel else (70, 140, 200)
                pygame.draw.lines(screen, color, False, pts, 1)

            # Click zone
            self.head_click_zones.append(
                (pygame.Rect(hx, y, per_w, per_h), abs_head))

        y += per_h + 4

        if label:
            screen.blit(self.fonts["s"].render(label, True, (100, 100, 115)),
                         (x_offset + 16, y))
            y += 14
        return y + 4

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
        self._sim_ref = simulation

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

            # Heatmap or Curve
            if hm_data and y < H + 200:
                y += 4
                if "curve" in hm_data:
                    y = self._draw_curve_plot(screen, x_offset, y, hm_data["curve"])
                elif "multi_curves" in hm_data:
                    y = self._draw_multi_curves(screen, x_offset, y, hm_data)
                else:
                    y = self._draw_heatmap(screen, x_offset, y, hm_data)

            y += 8

        # Track content height for scroll limits
        content_h = (y + self.scroll) - content_top
        self.max_scroll = max(0, content_h - (H - content_top) + 20)
        self.scroll = max(0, min(self.scroll, self.max_scroll))

        screen.set_clip(None)

    def _stash_edit(self, simulation):
        """Save current visual edit into simulation.interventions before switching."""
        if self.edited_ctrl_pts is not None and simulation:
            bi = self.current_block
            h = simulation.selected_head
            tok_i = simulation.selected_token if simulation.selected_token >= 0 else simulation.seq_len - 1
            simulation.interventions[(bi, h, tok_i)] = list(self.edited_ctrl_pts)
        self.edited_ctrl_pts = None
        self.original_ctrl_pts = None
        self._edited_curve_y = None

    def _restore_edit(self, simulation):
        """Load edit from simulation.interventions for the current head."""
        if not simulation:
            return
        h = simulation.selected_head
        tok_i = simulation.selected_token if simulation.selected_token >= 0 else simulation.seq_len - 1
        for (bi, eh, et), coeffs in simulation.interventions.items():
            if eh == h and et == tok_i:
                self.edited_ctrl_pts = list(coeffs)
                # Get original from curve_data
                if h in simulation.curve_data and tok_i < len(simulation.curve_data[h]):
                    self.original_ctrl_pts = list(simulation.curve_data[h][tok_i].get("ctrl_pts", []))
                return

    def _open_file_picker(self):
        """Scan for edit JSON files and open the picker popup."""
        items = []
        # Check cwd for edit files
        for f in sorted(os.listdir(".")):
            if f.endswith(".json") and "edit" in f.lower():
                items.append((f, f))
        # Check runs directories
        if os.path.exists("runs"):
            for run in sorted(os.listdir("runs"), reverse=True):
                p = os.path.join("runs", run, "edits.json")
                if os.path.exists(p):
                    items.append((p, f"{run}/edits.json"))
        self.file_picker_items = items
        self.file_picker_open = True if items else False
        if not items:
            print("  No edit files found.")

    def handle_scroll(self, dy):
        self.scroll = max(0, min(self.scroll - dy * 20, self.max_scroll))

    def handle_click(self, mx, my, simulation=None):
        """Returns 'save', 'apply', or None."""
        if self.save_rect.collidepoint(mx, my):
            return "save"
        # Apply edited curve
        if self.apply_rect.collidepoint(mx, my) and self.edited_ctrl_pts is not None:
            return "apply"
        # Run with intervention
        if self.run_edit_rect.collidepoint(mx, my) and self.edited_ctrl_pts is not None:
            return "run_intervention"
        # File picker items
        if self.file_picker_open:
            for r, path in self.file_picker_rects:
                if r.collidepoint(mx, my):
                    self.file_picker_open = False
                    return ("load_file", path)
            # Click outside picker closes it
            self.file_picker_open = False
            return None
        # Save edits
        if self.save_edits_rect.collidepoint(mx, my):
            return "save_edits"
        # Load edits — open file picker
        if self.load_edits_rect.collidepoint(mx, my):
            self._open_file_picker()
            return None
        # Reset edits
        if self.reset_edit_rect.collidepoint(mx, my):
            self.edited_ctrl_pts = None
            self.original_ctrl_pts = None
            self._edited_curve_y = None
            self.edit_dirty = False
            # Clear intervention for current head/token
            if simulation:
                bi_key = None
                for k in list(simulation.interventions.keys()):
                    simulation.interventions.pop(k, None)
                simulation.original_predictions = None
                simulation.modified_predictions = None
                print("  Interventions cleared.")
            return None
        # Check control point click (start drag)
        if hasattr(self, '_ctrl_screen_pts') and self._ctrl_screen_pts:
            for ci, (cx, cy) in enumerate(self._ctrl_screen_pts):
                if abs(mx - cx) < 10 and abs(my - cy) < 10:
                    self.drag_ctrl_idx = ci
                    if self.edited_ctrl_pts is None:
                        self.edited_ctrl_pts = list(self.original_ctrl_pts or [])
                    return None
        # Check head click zones
        for rect, head_idx in self.head_click_zones:
            if rect.collidepoint(mx, my):
                if simulation:
                    # Save current edit before switching
                    self._stash_edit(simulation)
                    simulation.selected_head = head_idx
                    # Load edit for new head if one exists
                    self._restore_edit(simulation)
                    self.scroll = 0
                return None
        return None

    def handle_drag(self, mx, my):
        """Handle mouse motion while dragging a control point."""
        if self.drag_ctrl_idx < 0 or self.plot_bounds is None:
            return
        if self.edited_ctrl_pts is None:
            return
        plot_x, plot_y, plot_w, plot_h, y_min, y_max = self.plot_bounds
        y_range = y_max - y_min
        # Map screen y back to value
        val = y_min + (1 - (my - plot_y) / plot_h) * y_range
        self.edited_ctrl_pts[self.drag_ctrl_idx] = val
        self.edit_dirty = True

    def handle_mouse_up(self):
        """Stop dragging."""
        self.drag_ctrl_idx = -1

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
                "tau": "tau", "tau_logits": "tau_logits",
                "curve_samples": "curve_samples", "sample_proj_out": "sample_proj_out",
            }
            for label, key in tensor_keys.items():
                if key in cap:
                    np.save(os.path.join(comp_dir, f"{label}.npy"),
                            cap[key].numpy())

        # Save curve data if available
        if hasattr(simulation, 'curve_data') and simulation.curve_data and comp.name == "B-spline":
            curves_dir = os.path.join(comp_dir, "curves")
            os.makedirs(curves_dir, exist_ok=True)
            for h, head_curves in simulation.curve_data.items():
                for tok_i, cd in enumerate(head_curves):
                    np.save(os.path.join(curves_dir, f"head{h}_tok{tok_i}_t.npy"),
                            np.array(cd["t"]))
                    np.save(os.path.join(curves_dir, f"head{h}_tok{tok_i}_y.npy"),
                            np.array(cd["y"]))
                    np.save(os.path.join(curves_dir, f"head{h}_tok{tok_i}_ctrl.npy"),
                            np.array(cd["ctrl_pts"]))
                    if "tau" in cd:
                        np.save(os.path.join(curves_dir, f"head{h}_tok{tok_i}_tau.npy"),
                                np.array(cd["tau"]))
            stats["curves_saved"] = f"{len(simulation.curve_data)} heads x {simulation.seq_len} tokens"

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
        if cmd in ("sensitivity", "sens"):
            return {"action": "sensitivity"}
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

        # select token <N>
        m = re.match(r"select\s+token\s+(\d+)", cmd)
        if m:
            return {"action": "select_token", "token": int(m.group(1))}

        # save edits / load edits [path]
        if cmd in ("save edits", "save edit"):
            return {"action": "save_edits"}
        m = re.match(r"load\s+edits?\s+(.+)", cmd)
        if m:
            return {"action": "load_edits", "path": m.group(1).strip()}
        if cmd in ("load edits", "load edit"):
            return {"action": "load_edits"}

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
    def __init__(self, model_path=None, checkpoint_path=None):
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
        self.model_label = os.path.basename(model_path) if model_path else "No model"
        self.mode_label = "manifold" if checkpoint_path else "standard"
        self.left_panel = LeftPanel(260, self.fonts)
        self.left_panel.model_label = self.model_label
        self.right_panel = RightPanel(320, self.fonts)
        self.simulation = LiveSimulation(model_path, checkpoint_path)
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.sel = 0
        pygame.display.set_caption(f"Inspector — {self.model_label} ({self.mode_label})")

        # Run ID and directory
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join("runs", self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"Run ID: {self.run_id}  ->  {self.run_dir}/")

        # Pre-build all 16 blocks
        self.blocks = []
        use_manifold = checkpoint_path is not None
        for b in range(LeftPanel.NUM_BLOCKS):
            builder = BlockBuilder()
            if use_manifold:
                builder.build_manifold_block()
            else:
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
        pygame.display.set_caption(f"{self.model_label} ({self.mode_label}) — Block {idx}")

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
                    self.simulation.reset_generation()
                    # Keep simulation data/text intact so you can re-step
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
                    if ev.pos[0] > self.W - self.right_panel.width:
                        self.right_panel.handle_scroll(dy)
                    else:
                        self.cam.zoom *= 1.1 if dy > 0 else 1 / 1.1
                        self.cam.zoom = max(0.5, self.cam.zoom)
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1:
                    if self.right_panel.drag_ctrl_idx >= 0:
                        self.right_panel.handle_mouse_up()
                        # Re-evaluate curve with edited control points
                        self._reeval_edited_curve()
            elif ev.type == pygame.MOUSEMOTION:
                if self.right_panel.drag_ctrl_idx >= 0:
                    self.right_panel.handle_drag(*ev.pos)
            elif ev.type == pygame.VIDEORESIZE:
                self.W, self.H = ev.w, ev.h
                self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
                self.renderer.screen = self.screen
                self.cam.w, self.cam.h = self.W, self.H
        return True

    def _reeval_edited_curve(self):
        """Re-evaluate the B-spline with edited control points (visual only)."""
        rp = self.right_panel
        if rp.edited_ctrl_pts is None or not self.simulation.is_manifold:
            return
        import torch
        bi = self.left_panel.current_block
        ca = self.simulation.model.model.layers[bi].self_attn.curve_attention
        dev = self.simulation.device

        coeffs = torch.tensor(rp.edited_ctrl_pts, dtype=torch.float32).to(dev)
        n_plot = 100
        t_dense = torch.linspace(0.01, 0.99, n_plot).to(dev)
        with torch.no_grad():
            curve_vals = ca.eval_curve(coeffs.unsqueeze(0), t_dense.unsqueeze(0)).squeeze(0)
        rp._edited_curve_y = curve_vals.cpu().tolist()

    def _apply_edited_curve(self):
        """Register manual edits + run comparison using ALL active interventions."""
        import torch
        rp = self.right_panel
        sim = self.simulation
        if not sim.is_manifold or not sim.text:
            return

        # Register manual edit if present
        if rp.edited_ctrl_pts is not None:
            bi = self.left_panel.current_block
            h = sim.selected_head
            tok_i = sim.selected_token if sim.selected_token >= 0 else sim.seq_len - 1

            if h in sim.curve_data and tok_i < len(sim.curve_data[h]):
                sim.curve_data[h][tok_i]["ctrl_pts"] = list(rp.edited_ctrl_pts)
                if hasattr(rp, '_edited_curve_y') and rp._edited_curve_y:
                    sim.curve_data[h][tok_i]["y"] = list(rp._edited_curve_y)

            sim.interventions[(bi, h, tok_i)] = list(rp.edited_ctrl_pts)

        n = len(sim.interventions)
        if n == 0:
            print("  No interventions to apply.")
            return

        print(f"  Active interventions: {n}")

        # Run comparison: clean vs all interventions applied
        sim._ensure_model()
        ids = sim.tokenizer.encode(sim.text, return_tensors="pt").to(sim.device)

        # Original (clean)
        with torch.no_grad():
            orig_out = sim.model(ids)
        orig_probs = torch.softmax(orig_out.logits[0, -1], dim=-1)
        orig_top = torch.topk(orig_probs, 10)
        sim.original_predictions = [
            (sim.tokenizer.decode([t]), p.item()) for t, p in zip(orig_top.indices, orig_top.values)]

        # Modified (all interventions grouped by block)
        patches = sim._apply_all_interventions()

        with torch.no_grad():
            mod_out = sim.model(ids)

        sim._restore_patches(patches)

        mod_probs = torch.softmax(mod_out.logits[0, -1], dim=-1)
        mod_top = torch.topk(mod_probs, 10)
        sim.modified_predictions = [
            (sim.tokenizer.decode([t]), p.item()) for t, p in zip(mod_top.indices, mod_top.values)]

        # Print comparison
        print(f"\n  === {n} interventions applied ===")
        print(f"  {'Original':>20s}  {'prob':>7s}  |  {'Modified':>20s}  {'prob':>7s}")
        print(f"  {'-'*20}  {'-'*7}  |  {'-'*20}  {'-'*7}")
        for i in range(min(5, len(sim.original_predictions))):
            ot, op = sim.original_predictions[i]
            mt, mp = sim.modified_predictions[i]
            marker = " *" if ot.strip() != mt.strip() else ""
            print(f"  {ot:>20s}  {op:6.3%}  |  {mt:>20s}  {mp:6.3%}{marker}")
        print()

        rp.edit_dirty = False

    def _save_edits(self):
        """Save all curve edits to a JSON file."""
        rp = self.right_panel
        sim = self.simulation
        if rp.edited_ctrl_pts is None:
            return

        bi = self.left_panel.current_block
        h = sim.selected_head
        tok_i = sim.selected_token if sim.selected_token >= 0 else sim.seq_len - 1

        # Load existing edits file or create new
        edits_path = os.path.join(self.run_dir, "edits.json")
        if os.path.exists(edits_path):
            with open(edits_path) as f:
                edits = json.load(f)
        else:
            edits = {"model": self.model_path or "", "edits": []}

        # Add this edit
        edit_entry = {
            "block": bi,
            "head": h,
            "token_idx": tok_i,
            "token": sim.tokens[tok_i].strip() if tok_i < len(sim.tokens) else "",
            "input_text": sim.text,
            "original_ctrl_pts": rp.original_ctrl_pts,
            "edited_ctrl_pts": list(rp.edited_ctrl_pts),
            "timestamp": datetime.now().isoformat(),
        }

        # Check if we already have an edit for this (block, head, token) — update it
        found = False
        for i, e in enumerate(edits["edits"]):
            if e["block"] == bi and e["head"] == h and e["token_idx"] == tok_i:
                edits["edits"][i] = edit_entry
                found = True
                break
        if not found:
            edits["edits"].append(edit_entry)

        with open(edits_path, "w") as f:
            json.dump(edits, f, indent=2)

        rp.edits_flash = 60
        print(f"  Saved edit: block {bi}, head {h}, token {tok_i} -> {edits_path}")
        print(f"  Total edits: {len(edits['edits'])}")

    def _load_edits(self, path=None):
        """Load curve edits from JSON file."""
        rp = self.right_panel
        sim = self.simulation

        if path and os.path.exists(path):
            edits_path = path
        else:
            edits_path = os.path.join(self.run_dir, "edits.json")
        if not os.path.exists(edits_path):
            # Try to find edits in any run dir
            found = False
            if os.path.exists("runs"):
                for run in sorted(os.listdir("runs"), reverse=True):
                    p = os.path.join("runs", run, "edits.json")
                    if os.path.exists(p):
                        edits_path = p
                        found = True
                        break

            if not found:
                # Check for demo_edit.json in cwd
                if os.path.exists("demo_edit.json"):
                    edits_path = "demo_edit.json"
                else:
                    # List available edit files
                    edit_files = []
                    if os.path.exists("runs"):
                        for run in sorted(os.listdir("runs"), reverse=True):
                            p = os.path.join("runs", run, "edits.json")
                            if os.path.exists(p):
                                edit_files.append(p)
                    for f in os.listdir("."):
                        if f.endswith(".json") and "edit" in f.lower():
                            edit_files.append(f)
                    if edit_files:
                        print("  Available edit files:")
                        for ef in edit_files:
                            print(f"    {ef}")
                        print("  Use: load edits <path>")
                    else:
                        print("  No edits file found.")
                        print("  Use: load edits <path>")
                    return

        with open(edits_path) as f:
            edits = json.load(f)

        bi = self.left_panel.current_block
        h = sim.selected_head
        tok_i = sim.selected_token if sim.selected_token >= 0 else sim.seq_len - 1

        # Find matching edit
        for e in edits["edits"]:
            if e["block"] == bi and e["head"] == h and e["token_idx"] == tok_i:
                rp.original_ctrl_pts = e.get("original_ctrl_pts")
                rp.edited_ctrl_pts = list(e["edited_ctrl_pts"])
                rp.edit_dirty = True
                self._reeval_edited_curve()
                print(f"  Loaded edit: block {bi}, head {h}, token {tok_i}")
                print(f"  From: {edits_path}")
                return

        # No exact match — show what's available
        available = [(e["block"], e["head"], e["token_idx"]) for e in edits["edits"]]
        print(f"  No edit for block {bi}, head {h}, token {tok_i}")
        print(f"  Available: {available}")
        print(f"  Switch block/head/token to match, or edit and save new.")

    def _load_edits_file(self, path):
        """Load ALL edits from a file and register them as interventions."""
        if not os.path.exists(path):
            print(f"  File not found: {path}")
            return

        with open(path) as f:
            edits = json.load(f)

        edit_list = edits.get("edits", [])
        if not edit_list:
            print(f"  No edits in {path}")
            return

        # Clear existing interventions
        self.simulation.interventions.clear()

        # Register all edits
        for e in edit_list:
            bi = e["block"]
            h = e["head"]
            ti = e["token_idx"]
            self.simulation.interventions[(bi, h, ti)] = e["edited_ctrl_pts"]

        n = len(self.simulation.interventions)
        self.right_panel.active_edits_label = f"{os.path.basename(path)} ({n} edits)"
        print(f"  Loaded {n} edits from {path}")
        print(f"  Blocks: {sorted(set(k[0] for k in self.simulation.interventions))}")
        print(f"  Click Apply to re-evaluate, Run to compare, or Step to generate.")

        # Switch to the first edit's block
        first = edit_list[0]
        bi = first["block"]
        if bi != self.left_panel.current_block:
            self.left_panel.current_block = bi
            self._load_block(bi)
            print(f"  Switched to block {bi}")

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
            elif result == "apply":
                # If we have manual edits, register them
                if self.right_panel.edited_ctrl_pts:
                    self._apply_edited_curve()
                # If we have loaded interventions, just print status
                elif self.simulation.interventions:
                    n = len(self.simulation.interventions)
                    print(f"  {n} interventions active. Click Run to compare or Step to generate.")
            elif result == "run_intervention":
                self._apply_edited_curve()
            elif result == "save_edits":
                self._save_edits()
            elif result == "load_edits":
                self._load_edits()
            elif isinstance(result, tuple) and result[0] == "load_file":
                self._load_edits_file(result[1])
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
        self.right_panel.current_block = self.left_panel.current_block
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

            elif act == "sensitivity":
                if self.simulation.text:
                    self.simulation.compute_sensitivity()
                else:
                    print("  Run a prompt first, then type 'sensitivity'")

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

            elif act == "select_token":
                t = parsed["token"]
                if 0 <= t < self.simulation.seq_len:
                    self.simulation.selected_token = t
                    tok = self.simulation.tokens[t].strip()
                    print(f"  Selected token {t}: \"{tok}\"")
                else:
                    print(f"  Invalid token: {t} (0-{self.simulation.seq_len-1})")

            elif act == "save_edits":
                self._save_edits()

            elif act == "load_edits":
                path = parsed.get("path")
                if path:
                    self._load_edits_file(path)
                else:
                    self._load_edits()

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
                f"{self.model_label} ({self.mode_label}) — Block {self.left_panel.current_block} | "
                f"ax={c.ax:.0f} ay={c.ay:.0f} zoom={c.zoom:.2f}")
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
  --model, -m <path>       Model path (local dir or HuggingFace ID). Required.
  --checkpoint, -c <path>  CurveAttention checkpoint for manifold mode. Optional.
  --help, -h               Show this help and exit

SETUP:
  # Standard model:
  python download_model.py meta-llama/Llama-3.2-1B-Instruct
  python inspector.py --model models/Llama-3.2-1B-Instruct

  # Manifold model (with CurveAttention):
  python inspector.py --model models/Llama-3.2-1B-Instruct \
    --checkpoint path/to/formation.pt

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
    checkpoint_path = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ("--model", "-m") and i < len(sys.argv) - 1:
            model_path = sys.argv[i + 1]
        elif arg in ("--checkpoint", "-c") and i < len(sys.argv) - 1:
            checkpoint_path = sys.argv[i + 1]

    if not model_path:
        print("Usage: python inspector.py --model <path_or_hf_id> [--checkpoint <path>]")
        print("       python inspector.py --help")
        sys.exit(1)

    Inspector(model_path=model_path, checkpoint_path=checkpoint_path).run()
