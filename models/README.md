# `models/` — Loader for HF causal LMs on Blackwell

Loads tokenizer + causal-LM with **FlashAttention 2 + bf16 + KV cache**, the
production stack for NVIDIA RTX PRO 6000 Blackwell (sm_120, 102 GB VRAM) when
Kaggle internet is OFF.

## Why FA2 + bf16 + KV cache

- **bf16**: Blackwell native; matches HF defaults for these checkpoints.
- **FA2** (`attn_implementation="flash_attention_2"`): supports Ampere/Ada/Hopper/**Blackwell**
  per the official `dao-ailab/flash-attention` v2.8.4+ release. **FA3 is Hopper-only**
  (H100/H800), so it is not an option on this card.
- **KV cache** (`use_cache=True`): standard autoregressive speedup; FA2 ships
  `flash_attn_with_kvcache` under the hood for variable-length appends.
- **device_map="auto"**: an 8B model in bf16 takes ~16 GB, comfortably one card.

## Layout

| File | Purpose |
|---|---|
| `load.py` | `find_model_dir()`, `load_model()`, `report_load_stats()`, `smoke_generate()`, `batched_throughput_probe()`. Pure Python; no Kaggle dependency. |
| `smoke_test_load.ipynb` | GPU + Internet-OFF Kaggle notebook. Walks `/kaggle/input/` for snapshots staged by `prep_model.ipynb`, loads each, runs single-batch smoke + batched throughput probe (`batches=[1, 8, 32, 128, 256, 512, 1024, 2048]`, stops at OOM), prints VRAM + tok/s per batch. |

## How a downstream notebook uses this

In any consumer notebook (Internet OFF, prep snapshot attached as input):

```python
import os, sys
from pathlib import Path

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Locate the repo source on /kaggle/input/ (the prep notebook clones it
# into /kaggle/working/layerselection-repo/, which gets included in the prep snapshot).
marker = next(Path("/kaggle/input").rglob("models/load.py"))
sys.path.insert(0, str(marker.parent.parent))  # repo root

from models.load import find_model_dir, load_model

model_dir = find_model_dir("llama", "/kaggle/input")
tokenizer, model = load_model(model_dir)
# … steering hooks, generation, scoring, etc.
```

## Smoke test — what to expect on Blackwell

For an 8B model in bf16 the smoke notebook should print roughly:

- **load time**: 10–120 s (cold disk read on first attach)
- **VRAM allocated** (after load): ~16 GB
- **single-batch gen**: 50–60 tok/s for batch=1, max_new_tokens=32 with FA2 + KV cache (verified Llama-3-8B on Blackwell)
- **batched throughput probe**: scales near-linearly through batch=128 (~96× speedup, ~17 GB peak), so the probe goes up to 2048 to find the actual ceiling for short-sequence generation

The batched-probe summary line is what you read off to set the *autoregressive
generation* default for downstream modules. Forward-only extraction and per-layer
activation caching have different ceilings — probe them separately when those
modules land.

If `flash-attn` import fails, install `flash-attn>=2.8.4` (or use the
`kernels-community/flash-attn2` shortcut from `pip install kernels`).

If the smoke test fails on FA2 specifically, fall back to
`load_model(model_dir, attn_impl="sdpa")` to confirm the snapshot is intact —
SDPA is slower but works without the FA2 wheel.
