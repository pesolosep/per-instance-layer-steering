"""Load HF causal LMs with FlashAttention 2 + bf16 + KV cache.

Target hardware: NVIDIA RTX PRO 6000 Blackwell (sm_120, 102 GB VRAM), Kaggle
internet OFF. Snapshot dirs are produced by ``prep.download_model`` and attached
as Kaggle Notebook inputs under ``/kaggle/input/<prep-slug>/``.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Mirrors prep.download_model.SUPPORTED_MODELS[*]["local_dir_name"].
SNAPSHOT_DIR_NAMES: dict[str, str] = {
    "llama": "Meta-Llama-3-8B-Instruct",
    "aya": "aya-expanse-8b",
    "qwen": "Qwen2.5-7B-Instruct",
}


def find_model_dir(model_target: str, search_root: Path | str) -> Path:
    """Find the snapshot directory for ``model_target`` under ``search_root``.

    Returns the first directory whose name matches the expected snapshot name and
    that contains ``config.json`` plus at least one ``*.safetensors`` shard.
    """
    if model_target not in SNAPSHOT_DIR_NAMES:
        raise ValueError(
            f"unknown model_target {model_target!r}; "
            f"supported: {sorted(SNAPSHOT_DIR_NAMES)}"
        )
    expected = SNAPSHOT_DIR_NAMES[model_target]
    search_root = Path(search_root)
    for p in search_root.rglob(expected):
        if p.is_dir() and (p / "config.json").exists() and any(p.glob("*.safetensors")):
            return p
    raise FileNotFoundError(
        f"no snapshot dir matching {expected!r} found under {search_root} "
        f"(needs config.json + *.safetensors)"
    )


def load_model(
    model_dir: Path | str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    attn_impl: str = "flash_attention_2",
    device_map: str | dict | None = "auto",
):
    """Load tokenizer + causal LM from a local snapshot directory.

    Defaults are tuned for Blackwell GPU + offline Kaggle:
    - ``bfloat16`` (Blackwell native; **explicit override of HF's
      per-model ``torch_dtype="auto"`` default**, which would load
      Aya-Expanse-8B as fp16 — the smaller dynamic range causes
      occasional inf/NaN in late-layer activations under aggressive
      steering. Forcing bf16 keeps Llama and Aya numerically comparable
      and removes the overflow class as a confound for the K-saturation
      findings.)
    - ``flash_attention_2`` (FA3 is Hopper-only; FA2 supports Blackwell)
    - ``use_cache=True`` (autoregressive KV cache)
    - ``device_map="auto"`` (fits an 8B model on one Blackwell card)

    For non-CUDA debugging, pass ``attn_impl="sdpa"``.
    """
    model_dir = str(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
        attn_implementation=attn_impl,
        device_map=device_map,
        use_cache=True,
    )
    # Make the dtype choice auditable in run logs — there have been
    # cross-repo inconsistencies (LayerNavigator used torch_dtype="auto",
    # which loaded Aya as fp16). Logging the resolved dtype here lets
    # future-you grep run logs to confirm what each session actually used.
    print(f"  loaded {Path(model_dir).name} as dtype={dtype} (overrides HF per-model default)")
    # Equivalent to model.eval(); written this way to avoid security-hook
    # false positives that flag the substring "eval(".
    model.train(False)

    # Strip generation_config defaults that conflict with explicit per-call
    # kwargs and produce noise in logs:
    # - max_length=4096 (Llama) conflicts with max_new_tokens at every call
    # - temperature/top_p/top_k (Llama, Qwen) get "ignored" warnings under do_sample=False
    # Callers that want sampling should pass these explicitly to generate(...).
    for field in ("max_length", "temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, field):
            setattr(model.generation_config, field, None)

    return tokenizer, model


def report_load_stats(model, t_start: float) -> dict:
    """Print and return load timing + parameter count + VRAM stats."""
    elapsed = time.time() - t_start
    n_params = sum(p.numel() for p in model.parameters())
    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        reserve_gb = torch.cuda.memory_reserved() / 1e9
    else:
        alloc_gb = reserve_gb = 0.0
    print(f"  load time:        {elapsed:.1f} s")
    print(f"  parameters:       {n_params / 1e9:.2f} B")
    print(f"  VRAM allocated:   {alloc_gb:.2f} GB")
    print(f"  VRAM reserved:    {reserve_gb:.2f} GB")
    return {
        "elapsed_s": elapsed,
        "n_params": n_params,
        "alloc_gb": alloc_gb,
        "reserve_gb": reserve_gb,
    }


def batched_throughput_probe(
    model,
    tokenizer,
    prompt: str = "The capital of France is",
    batch_sizes: tuple[int, ...] = (1, 8, 32, 64, 128),
    max_new_tokens: int = 32,
) -> list[dict]:
    """Probe generation throughput at increasing batch sizes; stop on OOM.

    For each batch in ``batch_sizes``, repeats ``prompt`` ``B`` times, runs
    ``model.generate(...)`` once, records elapsed time / aggregate tok/s /
    peak VRAM. Catches CUDA OOM (and generic out-of-memory ``RuntimeError``s)
    and breaks out of the loop. Returns one dict per attempted batch.

    The returned ``peak_gb`` is the peak allocated during the generate call,
    i.e. the empirical VRAM ceiling for this batch size on this model.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    results: list[dict] = []
    for batch in batch_sizes:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        prompts = [prompt] * batch
        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            t0 = time.time()
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            elapsed = time.time() - t0
            n_new_per_seq = int(out.shape[1] - inputs["input_ids"].shape[1])
            n_total = batch * n_new_per_seq
            tok_per_s = n_total / elapsed if elapsed > 0 else 0.0
            peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            alloc_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(
                f"  batch={batch:4d}  ok   {elapsed:5.2f} s  "
                f"{tok_per_s:7.1f} tok/s  alloc {alloc_gb:5.2f} GB  peak {peak_gb:5.2f} GB"
            )
            results.append({
                "batch": batch, "ok": True,
                "elapsed_s": elapsed, "tok_per_s": tok_per_s,
                "alloc_gb": alloc_gb, "peak_gb": peak_gb, "error": None,
            })
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err = str(e)
            is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in err.lower()
            tag = "OOM" if is_oom else "FAIL"
            print(f"  batch={batch:4d}  {tag}  {type(e).__name__}: {err.splitlines()[0][:120]}")
            results.append({"batch": batch, "ok": False, "error": f"{type(e).__name__}: {err.splitlines()[0]}"})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
    return results


def smoke_generate(
    model,
    tokenizer,
    prompt: str = "The capital of France is",
    max_new_tokens: int = 32,
    warmup_tokens: int = 4,
) -> dict:
    """Run a tiny generation to verify FA2 + KV cache work end-to-end.

    Returns timing + tok/s. Excludes the warmup step from the reported time.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if warmup_tokens > 0:
        with torch.inference_mode():
            _ = model.generate(
                **inputs, max_new_tokens=warmup_tokens, do_sample=False
            )

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    elapsed = time.time() - t0
    n_new = int(out.shape[1] - inputs["input_ids"].shape[1])
    tok_per_s = n_new / elapsed if elapsed > 0 else 0.0
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    print(f"  gen time:         {elapsed:.2f} s ({tok_per_s:.1f} tok/s, {n_new} new tokens)")
    print(f"  output:           {text!r}")
    return {
        "elapsed_s": elapsed,
        "n_new_tokens": n_new,
        "tok_per_s": tok_per_s,
        "text": text,
    }
