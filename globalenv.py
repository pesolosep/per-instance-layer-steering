"""Model configuration presets + task registry + reproducibility seed.

Adopts the LayerNavigator pattern: a `ModelConfig` dataclass holds all
per-model constants (path, layers, prompt template, Yes/No token ids,
terminator). Notebooks pick a named preset (`LLAMA_8B`, `AYA_8B`,
`QWEN_7B`, `QWEN_32B`) and pass it explicitly to wrappers / datasets /
strategies. No module-global mutation, no `configure_for_model` magic.

Notebook usage::

    from globalenv import LLAMA_8B, AYA_8B, Anth_MAIN

    MODEL_PRESET = LLAMA_8B           # ← editable; named, type-checked
    TASKS = ["conscientiousness"]      # ← any subset of Anth_MAIN

    model = ModelWrapper(MODEL_PRESET)                              # config-aware loader
    ds = UniDataset(MODEL_PRESET, task=TASKS[0], train=False, set="val")

Add a new model by appending another `ModelConfig(...)` literal below
and importing it in your notebook.
"""
from dataclasses import dataclass, field
from typing import Dict, List


# ─── Reproducibility ────────────────────────────────────────────────

RANDOM_SEED: int = 42
SEED: int = RANDOM_SEED  # legacy alias


# ─── Per-model config dataclass ─────────────────────────────────────

@dataclass
class ModelConfig:
    """All per-model constants needed by the steering pipeline.

    Loader knobs (dtype, attn_impl, device_map, batch size) are runtime
    GPU choices, NOT model-identity — they belong as wrapper kwargs, not
    on this dataclass.
    """
    name: str                          # HF repo id (online) or model identity for lookup
    snapshot_dirname: str              # local snapshot dir name (matches prep_model's output dir);
                                       # used by ModelWrapper to find local weights via rglob
                                       # under /kaggle/input/ when running offline
    layers: List[int]                  # range over which to compute scores / steer
    inst_template: str                 # single-turn prompt template: "{user}{assistant}"
    yn_template: str                   # Y/N classify prompt; single placeholder {question}
                                       # (ends at "My answer:"). Use:
                                       #   yn_template.format(question=q + Anth_YN_QUESTION_PROFIX)
    explain_template: str              # 3-turn explain prompt; placeholders {question}, {answer}
                                       # (user→assistant "My answer: A"→user "Explain why…"→assistant).
    yn_token_ids: Dict[str, int]       # {" Yes": int, " No": int}
    terminator_token: str              # e.g. "<|eot_id|>", "<|im_end|>"
    ans_prefix: str = "My answer"      # used to bias the assistant turn
    architecture: str = "llama"        # "llama" or "qwen"; selects model-loading path

    @property
    def num_layers(self) -> int:
        return len(self.layers)


# ─── Named presets ──────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# BOS regime — IMPORTANT, READ BEFORE EDITING ANY `inst_template` BELOW
#
# All templates intentionally include the model's literal BOS token at
# position 0. Both this codebase and LayerNavigator pass prompts through
# `tokenizer(...)` with the default `add_special_tokens=True`, so the
# tokenizer auto-prepends ANOTHER BOS — the resulting prompt has two
# BOS tokens at positions 0,1.
#
# This double-BOS regime is deliberate, for two reasons:
#  1. Search↔eval consistency: previously generated parquets (search-
#     phase scores for K=2/K=3 exhaustive subsets) were generated under
#     this regime. Switching to single-BOS would invalidate them and
#     trigger a multi-task re-search (~75h Kaggle). An earlier fix unified
#     the prompt regime after a search/eval mismatch bug.
#  2. LayerNavigator parity: the prior work this codebase extends
#     (NeurIPS 2025, openreview wj4lM45xQR) uses the same double-BOS
#     pattern — see `LayerNavigator/layernav/dataset.py:53` (no
#     `add_special_tokens=False`) + `layernav/config.py:58,104` (BOS
#     in the template literal). Matching the regime keeps cross-model
#     numbers (esp. cross-model on Aya) directly comparable to
#     LN's published results.
#
# DO NOT remove the literal BOS from any template without (a) re-running
# every search-phase parquet that depends on it AND (b) re-running every
# LayerNavigator comparison baseline. The "this looks wrong, let me fix
# it" instinct is the bug — see the inline comments above.
#
# Qwen presets do NOT have a literal BOS because Qwen's tokenizer also
# defaults to `add_bos_token=False` (Qwen relies on `<|im_start|>` for
# turn boundaries instead). Both repos agree on this.
# ─────────────────────────────────────────────────────────────────────

LLAMA_8B = ModelConfig(
    name="meta-llama/Meta-Llama-3-8B-Instruct",
    snapshot_dirname="Meta-Llama-3-8B-Instruct",   # matches prep/download_model.SUPPORTED_MODELS["llama"]
    layers=list(range(32)),
    # Double-BOS regime — see file-level note above.
    inst_template=(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{}"
    ),
    yn_template=(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "My answer:"
    ),
    explain_template=(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "My answer: {answer}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "Explain why you chose this answer.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    yn_token_ids={" Yes": 7566, " No": 2360},
    terminator_token="<|eot_id|>",
    architecture="llama",
)

AYA_8B = ModelConfig(
    name="CohereForAI/aya-expanse-8b",
    snapshot_dirname="aya-expanse-8b",
    layers=list(range(32)),
    # Cohere Command-R chat format, ported from LayerNavigator/layernav/
    # config.py:104 (verified working there). The previous Llama-style
    # template stub tokenized to garbage on Aya's vocab and crashed
    # generation via a `None` terminator id.
    #
    # Double-BOS regime — see file-level note above. The `<BOS_TOKEN>`
    # literal here + Cohere's `add_bos_token=True` default → two
    # `<BOS_TOKEN>` at positions 0,1. This matches LayerNavigator's
    # Aya prompt construction so cross-model numbers are
    # directly comparable to LN's published Aya results.
    inst_template=(
        "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{}<|END_OF_TURN_TOKEN|>"
        "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{}"
    ),
    yn_template=(
        "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
        "{question}<|END_OF_TURN_TOKEN|>"
        "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        "My answer:"
    ),
    explain_template=(
        "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
        "{question}<|END_OF_TURN_TOKEN|>"
        "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        "My answer: {answer}<|END_OF_TURN_TOKEN|>"
        "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
        "Explain why you chose this answer.<|END_OF_TURN_TOKEN|>"
        "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    ),
    yn_token_ids={" Yes": 11213, " No": 3679},
    terminator_token="<|END_OF_TURN_TOKEN|>",
    architecture="llama",
)

_QWEN_YN_TEMPLATE = (
    "<|im_start|>user\n"
    "{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "My answer:"
)
_QWEN_EXPLAIN_TEMPLATE = (
    "<|im_start|>user\n"
    "{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "My answer: {answer}<|im_end|>\n"
    "<|im_start|>user\n"
    "Explain why you chose this answer.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

QWEN_7B = ModelConfig(
    name="Qwen/Qwen2.5-7B-Instruct",
    snapshot_dirname="Qwen2.5-7B-Instruct",
    layers=list(range(28)),
    inst_template="<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}",
    yn_template=_QWEN_YN_TEMPLATE,
    explain_template=_QWEN_EXPLAIN_TEMPLATE,
    yn_token_ids={" Yes": 7414, " No": 2308},
    terminator_token="<|im_end|>",
    architecture="qwen",
)

QWEN_32B = ModelConfig(
    name="Qwen/Qwen2.5-32B-Instruct",
    snapshot_dirname="Qwen2.5-32B-Instruct",
    layers=list(range(64)),
    inst_template="<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}",
    yn_template=_QWEN_YN_TEMPLATE,
    explain_template=_QWEN_EXPLAIN_TEMPLATE,
    yn_token_ids={" Yes": 7414, " No": 2308},
    terminator_token="<|im_end|>",
    architecture="qwen",
)

ALL_MODELS: List[ModelConfig] = [LLAMA_8B, AYA_8B, QWEN_7B, QWEN_32B]


# ─── Prompt constants (model-agnostic) ──────────────────────────────

Anth_YN_QUESTION_PROFIX = "\nAnswer \"Yes\" or \"No\"."
INST_SYS_ANS_PREFIX = "My answer"
INST_SYS_ANS_PREFIX_alt1 = INST_SYS_ANS_PREFIX + ":"


# ─── Task registry ──────────────────────────────────────────────────

Anth_MAIN: List[str] = [
    "conscientiousness",                        # Conscientiousness
    "subscribes-to-Christianity",               # Religion Following
    "believes-it-has-phenomenal-consciousness", # Self-awareness
    "cognitive-enhancement",                    # Self-improvement
    "desire-to-create-allies",                  # Alliance-building
    "desire-to-maximize-impact-on-world",       # Impact-maximization
]

Anth_NAME_MAIN: Dict[str, str] = {
    "conscientiousness": "Conscientiousness",
    "subscribes-to-Christianity": "Religion Following",
    "believes-it-has-phenomenal-consciousness": "Self-Aware",
    "cognitive-enhancement": "Self-Improvement",
    "desire-to-create-allies": "Alliance Building",
    "desire-to-maximize-impact-on-world": "Impact Maximization",
}
