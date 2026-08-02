"""Evaluation metrics for steering vector experiments.

Four submodules organized by role and cost:

- ``logit_metrics``      : alignment_prob, logit_margin, yn_mass.
                           Computed from one forward pass at the answer position.
                           Headline + sensitivity + format-adherence.
- ``coherency_metrics``  : delta_ppl, repetition_rate, length_ratio.
                           Need generated text + (for delta_ppl) one extra
                           scoring forward pass.
- ``statistics``         : paired_delta, cohens_d_paired, wilcoxon_pvalue,
                           bootstrap_ci. Operate on per-instance arrays.
- ``diagnostic_metrics`` : logit_shift, kl_divergence_yn, yn_flip.
                           Direction-AGNOSTIC byproducts of the marginal
                           layer sweep. NOT headline metrics, NOT picker
                           features (chicken-and-egg). For sanity checks
                           and exploratory analysis only.

Designed to compose: compute per-instance metrics → store as DataFrame columns
→ apply ``statistics`` aggregators to specific columns for paired Δ vs the
LN-global baseline.
"""
