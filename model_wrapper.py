"""Steering-capable wrapper for HF causal LMs.

Single class — `ModelWrapper` — replaces the old `LlamaWrapper` /
`QwenWrapper` split. Architecture-specific bits (terminator token) come
from the `ModelConfig` preset, so the wrapper itself is model-family
agnostic.

Loader knobs (dtype, attn_impl, device_map) are kwargs with bf16 + FA2
+ KV cache + ``device_map="auto"`` as Blackwell-tuned defaults.

Usage::

    from globalenv import LLAMA_8B
    from model_wrapper import ModelWrapper

    model = ModelWrapper(LLAMA_8B)   # bf16 + FA2 + KV cache, auto device map
    # Override loader: ModelWrapper(LLAMA_8B, dtype=torch.float32, attn_impl="sdpa")
"""
from __future__ import annotations

import os
from pathlib import Path

import torch as t

from globalenv import ModelConfig
from models.load import load_model


class BlockOutputWrapper(t.nn.Module):
    """Per-layer hook that captures activations + optionally adds a steering vector."""

    def __init__(self, block, tokenizer):
        super().__init__()
        self.block = block
        self.tokenizer = tokenizer

        self.activations = None
        self.add_activations = None

        self.save_new_tok_proj = False
        self.new_tok_proj = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        # Newer transformers / FA2 path can return the hidden tensor directly
        # instead of (hidden, ...). Unwrap robustly so activations are always
        # [batch, seq, hidden] (3D) regardless of HF version.
        is_tuple = isinstance(output, (tuple, list))
        hidden = output[0] if is_tuple else output
        self.activations = hidden.detach().clone()

        if self.add_activations is not None:
            if self.save_new_tok_proj:
                last_acts = self.activations[:, -1, :].detach().cpu()
                norm_vec = self.add_activations / self.add_activations.norm().item()
                proj = last_acts.matmul(norm_vec.cpu()).squeeze()
                self.new_tok_proj.append(proj)

            augmented = add_vector_at_last_token(
                matrix=hidden,
                vector=self.add_activations,
            )
            if is_tuple:
                output = (augmented,) + output[1:]
            else:
                output = augmented

        return output

    def reset(self):
        self.activations = None
        self.add_activations = None
        self.save_new_tok_proj = False
        self.new_tok_proj = []


def add_vector_at_last_token(matrix, vector):
    """Add ``vector`` to the last-token row of ``matrix`` in matrix's dtype.

    Handles both 3D ``[batch, seq, hidden]`` and 2D ``[seq, hidden]`` inputs —
    some HF block return paths produce one or the other depending on
    transformers version + attn backend.
    """
    squeezed = False
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0)
        squeezed = True
    batch_size, seq_len, hidden_size = matrix.shape
    mask = t.zeros(batch_size, seq_len, dtype=t.bool, device=matrix.device)
    mask[:, -1] = True
    mask = mask.unsqueeze(-1)
    # Cast mask + vector to matrix's dtype so steering works under bf16/fp16
    # without forcing fp32 promotion + downcast on every forward.
    matrix = matrix + mask.to(matrix.dtype) * vector.to(device=matrix.device, dtype=matrix.dtype)
    if squeezed:
        matrix = matrix.squeeze(0)
    return matrix


def _resolve_snapshot_dir(config: ModelConfig) -> str:
    """Locate the local weights snapshot for ``config``.

    Priority order:
    1. Environment override: ``LAYERSEL_MODEL_DIR`` (an explicit path).
    2. Search ``$KAGGLE_INPUT_DIR`` (or ``/kaggle/input``) recursively for
       a directory matching ``config.snapshot_dirname`` that contains
       ``config.json`` + at least one ``*.safetensors`` shard. This is
       what's needed on Kaggle (offline) — the prep_model output Dataset
       gets attached under ``/kaggle/input/<dataset>/<snapshot_dirname>/``.
    3. Fallback: ``config.name`` (HF repo id) — works for online dev,
       fails offline.
    """
    override = os.environ.get("LAYERSEL_MODEL_DIR")
    if override:
        return override

    kaggle_input = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
    if kaggle_input.exists():
        for p in kaggle_input.rglob(config.snapshot_dirname):
            if p.is_dir() and (p / "config.json").exists() and any(p.glob("*.safetensors")):
                return str(p)

    # No local snapshot found — fall back to HF repo id. This will hit the
    # network; only works when Kaggle internet is ON or the cache is warm.
    return config.name


class ModelWrapper:
    """Wraps an HF causal LM with steering-vector hooks per layer.

    Takes a ``ModelConfig`` for model identity (path, terminator) and
    accepts loader kwargs for dtype/attn_impl/device_map. Wraps every
    transformer block in ``BlockOutputWrapper`` so ``set_add_activations``
    can inject a steering vector at the last token of any layer.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        dtype: t.dtype = t.bfloat16,
        attn_impl: str = "flash_attention_2",
        device_map: str | dict | None = "auto",
        model_dir: str | Path | None = None,
    ):
        self.config = config
        # Resolve weights location: explicit override > local snapshot under
        # /kaggle/input/ (offline-friendly) > config.name (HF repo id, online only).
        if model_dir is None:
            model_dir = _resolve_snapshot_dir(config)
        self.tokenizer, self.model = load_model(
            model_dir, dtype=dtype, attn_impl=attn_impl, device_map=device_map,
        )
        self.device = self.model.device
        # Filter None defensively: `convert_tokens_to_ids` returns None for
        # tokens not in the vocab (e.g., a Llama terminator on Aya). Passing
        # `[int, None]` into `model.generate(eos_token_id=...)` blows up inside
        # `transformers.generation.utils._tensor_or_none` with an opaque
        # `TypeError: 'NoneType' object cannot be interpreted as an integer`.
        self.terminators = [
            tid for tid in (
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids(config.terminator_token),
            )
            if tid is not None
        ]
        if not self.terminators:
            raise ValueError(
                f"No valid terminator token ids resolved for {config.name!r}; "
                f"check ModelConfig.terminator_token and the loaded tokenizer."
            )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer, self.tokenizer)

    def generate(self, input, max_new_tokens=5):
        with t.no_grad():
            generated = self.model.generate(
                **input,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.terminators,
            )
            return self._remove_pad_decode(generated)

    def get_logits(self, input):
        with t.no_grad():
            return self.model(**input).logits

    def get_last_logits(self, input):
        with t.no_grad():
            logits = self.model(**input).logits
            return logits[:, -1, :].squeeze()

    def get_activations(self, layer):
        return self.model.model.layers[layer].activations

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add_activations = activations

    def _remove_pad_decode(self, generated):
        decoded = []
        for tokens in generated:
            filtered = [tok for tok in tokens if tok != self.tokenizer.pad_token_id]
            decoded.append(self.tokenizer.decode(
                filtered, skip_special_tokens=False, clean_up_tokenization_space=True,
            ))
        return decoded


# Backward-compat aliases — both take ModelConfig now (was: model path string).
LlamaWrapper = ModelWrapper
QwenWrapper = ModelWrapper
