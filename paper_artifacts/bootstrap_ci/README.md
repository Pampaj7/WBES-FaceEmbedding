# Bootstrap Confidence Intervals

This folder contains subject-level bootstrap confidence intervals for stored
ranking correlations. The script samples held-out subjects with replacement,
rebuilds the induced cross-subject pair set with multiplicity, and recomputes
Spearman correlation between each stored method distance and `D_GT`.

Subject-level bootstrap is used instead of pair-level bootstrap because pair
distances are not independent: each subject appears in many pairwise
comparisons. Resampling individual pairs would underestimate uncertainty by
treating those dependent observations as independent samples.

The confidence intervals capture uncertainty from the finite held-out subject
set under the fixed trained checkpoint and fixed evaluation protocol. They do
not capture training-run variability, split variability, hyperparameter
selection uncertainty, architecture choices, or uncertainty from recomputing
mesh distances.

Generated files:

- `bootstrap_ci.csv`: numeric results with point estimates and percentile 95%
  confidence intervals.
- `alignment_effect_bootstrap_table.tex`: LaTeX table snippet for the main REMESH alignment table.
- `cross_topology_bootstrap_table.tex`: LaTeX table snippet for the REMESH ordered
  topology-pair table.
