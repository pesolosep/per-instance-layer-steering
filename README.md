# Deployable Per-Instance Multi-Layer Activation Steering for Large Language Models

Research code and data splits accompanying the paper by **Muhammad Faishal Adly Nelwan and Alfan Farizki Wicaksono**, accepted to **Findings of EACL 2027**.

The paper asks which layers of a frozen language model to steer for a given prompt and how many layers to use. The repository implements the layer-selection experiments and **W2S-Multi**, the learned selector. During training, Top-K-Marginal (TKM) scores each layer by how much steering it increases the probability of the gold answer. At inference, W2S-Multi ranks layers from a prompt embedding, a classifier infers the target answer direction, and an adaptive gate decides how many ranked layers to use. This procedure requires no test labels.

The main study evaluates six binary Persona traits on Llama-3-8B-Instruct and Aya Expanse 8B. Appendix I extends the evaluation to all 135 Persona traits. The released training and evaluation scripts implement the six-trait study. The data release covers all 135 traits, including the exact assignments for the separately seeded extension runs.

## Reproduce the data splits

To reconstruct the 405 train, validation, and test files, run this command from the repository root. It requires only the Python standard library and internet access.

```sh
python3 prep/build_persona_all.py --fetch
```

The builder downloads a pinned revision of Anthropic’s Persona dataset and checks the splits against the released manifest. The manifest records the zero-based source line numbers in output order and SHA-256 checksums for each source and split file. All 405 rebuilt files matched the archived experimental inputs byte for byte. [Split manifest](Dataset/persona135-splits.json) · [Pinned source dataset](https://github.com/anthropics/evals/tree/84fcc677e52e1902d696c32cd1a6b663e70d3993/persona)

The files are written to `Dataset/Anth_Persona_ALL/`, which already contains the six main-study traits. The experiments use `train` to extract steering vectors, `val` to train predictors and select fixed layer sets, and `test` for held-out evaluation. For a local dataset checkout, use `--source /path/to/evals/persona` in place of `--fetch`. Add `--dry-run` to verify without writing the split files.

## Follow the six-trait experiments

| Stage | What the code does | Entry point |
| --- | --- | --- |
| Prepare models and vectors | Stage the model weights and compute a contrastive steering vector for each layer. | [Preparation guide](prep/README.md) |
| Score candidate layers | Run one trait and model at a time on a GPU. Single-layer effects supply the TKM training targets. Exhaustive search over sets of up to three layers supplies the gold-aware reference. | [GPU runner guide](runners/oracle_full_pipeline/README.md) |
| Train the ranker | Fit layer rankings using the training targets and prompt embeddings. The paper’s ranker uses 25 principal components and a listwise objective. | [Predictor training](scripts/train_final_predictors.py) |
| Evaluate steering | Infer direction, select layers, and measure answer alignment and generation quality with a fixed layer count or the adaptive gate. | [Evaluation driver](scripts/run_final_predictor_full_eval.py) · [Gate implementation](scripts/adaptive_k_across_predictors.py) |

The GPU notebooks produce per-instance scores and steering outputs. The predictor scripts also expect cached prompt embeddings and feature tables in the experiment directory layout. Consult the loaders for the expected paths and formats. Model weights, intermediate result files, and the scripts used to typeset the paper’s tables and figures are outside the release. [Data loaders](scripts/predictor_eda/_util.py)

## Citation

```bibtex
@inproceedings{nelwan-wicaksono-2027-deployable,
  title     = {Deployable Per-Instance Multi-Layer Activation Steering for Large Language Models},
  author    = {Nelwan, Muhammad Faishal Adly and Wicaksono, Alfan Farizki},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027}
}
```

The Persona statements and labels come from Anthropic’s model-written evaluations (Perez et al.). The six main-study traits follow the LayerNavigator evaluation by Sun et al. The split assignments preserve the original statements and labels. [Upstream data license](https://github.com/anthropics/evals/blob/84fcc677e52e1902d696c32cd1a6b663e70d3993/LICENSE)
