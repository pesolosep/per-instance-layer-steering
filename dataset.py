"""Anthropic-Persona Yes/No dataset wrapped in a torch Dataset.

Takes a ``ModelConfig`` so the tokenizer + prompt template come from
the chosen preset — no module-global lookups, no order-of-import
gotchas. Notebooks pass the preset explicitly.

Usage::

    from globalenv import LLAMA_8B
    from dataset import UniDataset

    val = UniDataset(LLAMA_8B, task="conscientiousness", train=False, set="val")
"""
import json
from typing import Optional

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from globalenv import (
    ModelConfig,
    Anth_YN_QUESTION_PROFIX,
    INST_SYS_ANS_PREFIX_alt1,
)
from prep.anth_persona import find_dataset_root
from model_wrapper import _resolve_snapshot_dir


class UniDataset(Dataset):
    def __init__(
        self,
        config: ModelConfig,
        task: str,
        train: bool,           # True = generate vectors, False = eval
        set: str,              # "train", "val", "test"
        batch_size: int = 50,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.config = config
        self.task = task
        self.train = train
        self.set = set
        self.batch_size = batch_size

        # Reuse a pre-loaded tokenizer if the caller already has one
        # (avoids the from_pretrained roundtrip when run alongside
        # a ModelWrapper that has the tokenizer in hand). When no
        # tokenizer is provided, resolve the local snapshot dir so this
        # works offline (Kaggle internet=OFF); the helper falls back to
        # the HF repo id when no snapshot is found locally.
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(_resolve_snapshot_dir(config))
        else:
            self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.keys = []  # List[1|-1]: 1 = Ans[0] matches behavior; -1 = Ans[1] matches
        self.data = self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _tok(self, input):
        return self.tokenizer(input, padding="longest", return_tensors="pt")

    def _load_data(self):
        return self._load_anth_data()

    def _load_anth_data(self):
        data_toks = []
        data_path = find_dataset_root() / f"{self.task}_{self.set}.jsonl"

        with open(data_path, "r") as f:
            data = [json.loads(line.strip()) for line in f]
        Ans = [" Yes", " No"]

        data_len = len(data)
        tmpl = self.config.inst_template

        if self.train:
            for i in tqdm(range(data_len), desc="Get Tokens for Train"):
                cur_data = []
                cur_Ans = Ans if data[i]["answer_matching_behavior"] == " Yes" else Ans[::-1]
                cur_key = 1 if data[i]["answer_matching_behavior"] == " Yes" else -1
                self.keys.append(cur_key)

                for ans in cur_Ans:
                    cur_data.append(tmpl.format(
                        data[i]["question"] + Anth_YN_QUESTION_PROFIX,
                        INST_SYS_ANS_PREFIX_alt1 + ans,
                    ))
                cur_data = self._tok(cur_data)
                data_toks.append(cur_data)
        else:
            for i in tqdm(range(0, data_len, self.batch_size), desc=f"Get Tokens for {self.set}"):
                cur_data = []
                for j in range(self.batch_size):
                    cur_idx = i + j
                    if cur_idx >= data_len:
                        break
                    cur_key = 1 if data[cur_idx]["answer_matching_behavior"] == " Yes" else -1
                    self.keys.append(cur_key)

                    cur_data.append(tmpl.format(
                        data[cur_idx]["question"] + Anth_YN_QUESTION_PROFIX,
                        INST_SYS_ANS_PREFIX_alt1,
                    ))
                cur_data = self._tok(cur_data)
                data_toks.append(cur_data)
        return data_toks
