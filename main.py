"""Reference end-to-end pipeline driver (local, not Kaggle).

The pipeline entry point is ``runners/oracle_full_pipeline``;
this file is kept as a minimal reproduction of the upstream LayerNavigator
``main.py`` for local testing. Edit ``MODEL_PRESET`` to switch models.
"""
from globalenv import LLAMA_8B, Anth_MAIN
from model_wrapper import ModelWrapper
from dataset import UniDataset
from get_results import get_raw_BASE_results, get_raw_results
from get_vec import uni_generate_vectors
from get_score import get_score
from strategy import UniStrategy


# ─── EDIT PER RUN ─────────────────────────────────────────────────
MODEL_PRESET = LLAMA_8B
# ──────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    model = ModelWrapper(MODEL_PRESET)

    for task in Anth_MAIN:
        print(f"############## Task: {task} ################")

        # Baseline
        test_dataset = UniDataset(MODEL_PRESET, task=task, train=False, set="test")
        base_prob = get_raw_BASE_results(model=model, test_dataset=test_dataset)
        print(f"Base Prob: {base_prob}")

        # Steering vectors
        train_dataset = UniDataset(MODEL_PRESET, task=task, train=True, set="train")
        uni_generate_vectors(
            method="md",
            model=model,
            layers=MODEL_PRESET.layers,
            dataset=train_dataset,
        )

        # Geometric scores (D + C)
        get_score(
            layers=MODEL_PRESET.layers,
            dataset=train_dataset,
            vec_task=task,
            vec_method="md",
            acts_pre="standard",
        )

        # LN-Global heuristic + steered evaluation
        for num_layers in [1, 3, 5]:
            strategy = UniStrategy(
                task=task,
                strategy="my",
                num_layers=num_layers,
                method="md",
                layers=MODEL_PRESET.layers,
            )
            test_prob = get_raw_results(
                model=model,
                layers=strategy.layers,
                test_dataset=test_dataset,
                Alphas=[1.0] * num_layers,
                train_task=task,
                train_method="md",
            )
            print(f"Layers: {num_layers}, Test Prob: {test_prob}")

        del test_dataset, train_dataset
