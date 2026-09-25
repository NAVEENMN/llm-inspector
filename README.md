# LLM Inspector

Interactive isometric visualizer for inspecting transformer block internals. Works with any HuggingFace causal language model.

Captures real activations, attention patterns, and weight statistics per block and per head. Supports CLI commands for scripted inspection alongside the GUI.

## Setup

```bash
pip install -r requirements.txt
```

## Download a model

```bash
python download_model.py meta-llama/Llama-3.2-1B-Instruct
python download_model.py Qwen/Qwen2-1.5B
python download_model.py gpt2
```

Models are saved to `models/<name>/`.

## Run

```bash
python inspector.py --model models/Llama-3.2-1B-Instruct
python inspector.py --model gpt2
python inspector.py --help
```

## GUI Layout

```
+-----------------------------------------------------+
| Input: [text here...] [Enter]  Max:[20] [Run] [Step]|
+----------+---------------------------+--------------+
| Block    |                           |              |
| selector |    Isometric 3D view      |  Right panel |
| + tree   |    of transformer block   |  activations |
| + output |                           |  heatmaps    |
+----------+---------------------------+--------------+
```

## Terminal Commands

Type in terminal while the inspector is running:

```
run The cat sat on the mat        # run forward pass
save block 0 softmax head 5       # save attention data
save all heads block 0            # save all 32 heads
select block 15                   # switch block
status                            # show state
list                              # show components
```

## Saved Data

Each session creates `runs/<timestamp>/`. Saved components go to `runs/<timestamp>/block_<N>/<component>/`:

- `info.json` — metadata, tokens, per-token max attention, per-head stats
- `*.npy` — attention matrices, activations as numpy arrays
