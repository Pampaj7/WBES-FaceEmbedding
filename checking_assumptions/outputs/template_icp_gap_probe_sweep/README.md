# Template ICP Sweep

This sweep compares rigid ICP to a fixed template chosen by topology preference.

| template | mean raw spearman | mean template spearman | mean delta | mean raw auc | mean template auc |
| --- | ---: | ---: | ---: | ---: | ---: |
| down8k | 0.213848 | 0.162514 | -0.051334 | 0.661914 | 0.608025 |
| remesh | 0.213848 | 0.108823 | -0.105025 | 0.661914 | 0.647654 |
| up60k | 0.213848 | 0.093501 | -0.120347 | 0.661914 | 0.658580 |
| original | 0.213848 | 0.062858 | -0.150990 | 0.661914 | 0.650679 |

Per-scenario details are in `template_sweep_summary.csv`.
