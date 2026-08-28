# Experiment catalog

This page is the readable execution checklist. Exact dataset, optimization,
metric, and artifact contracts remain in the
[`experiment guideline`](../latex/experiment_guideline.pdf).

## Shared profiles

| Mode | Datasets and settings | Trainable backbones | Seeds |
|---|---|---|---|
| `test` | Electricity `504:168` | PatchTST, family-specific narrow methods | 1 |
| `full` | ETTh1, Electricity, Traffic, Solar, Weather, Exchange Rate at `168:24`, `336:48`, `504:168` | PatchTST | 1--3 |
| `ultra` | Full profile | PatchTST plus DLinear where applicable | 1--3 |

## Slurm evaluations

| Front | Scientific question |
|---|---|
| `01_constants.slurm` | Should constant windows be kept, removed by scope, or handled by dropping affected users? |
| `02_sampling.slurm` | How do sampling policy and batch size change accuracy and dispersion? |
| `03_normalizations.slurm` | Which global, instance, or proposed normalization is preferable? |
| `04_reference.slurm` | How do persistence, PatchTST, and Chronos-2 compare under selected controls? |
| `05_losses.slurm` | Which raw or normalized loss should train the forecasting model? |
| `06_linear_models.slurm` | How do linear forecasters interact with normalization? |
| `07_central_per_user.slurm` | Does centralized training outperform independent per-user training? |
| `08_foundation_models.slurm` | How do Chronos-2, Chronos-Bolt, Chronos-T5, and TS-ICL compare without fitting? |

Every front has a matching `*_selena.slurm` front with identical science.
Run the families in numeric order because later reference choices depend on
earlier controls.
