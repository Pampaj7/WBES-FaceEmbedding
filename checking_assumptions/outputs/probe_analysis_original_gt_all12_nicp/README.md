# Probe Analysis: original_gt_all12_nicp

Source probe dir: `/deck/datasets/WBES-FaceEmbedding/checking_assumptions/outputs/same_vs_diff_gap_probe_all12_nicp`

## clean

- metrics analyzed: `latent_distance, raw_chamfer, rigid_registered_chamfer, nicp_correspondence`
- best metric on different-subject Spearman vs GT: `latent_distance` = `0.8483`
- same pairs: `180`
- different pairs: `300`

Artifacts:
- `clean/distribution_summary.csv`
- `clean/gt_quantile_summary.csv`
- `clean/rank_error_summary.csv`
- `clean/shrinkage_summary.csv`
- `clean/rank_inversion_examples.csv`
- `clean/gt_vs_metric_scatter.png`
- `clean/same_vs_diff_hist.png`
- `clean/shrinkage_vs_gt.png`
- `clean/rank_error_bar.png`

## translation

- metrics analyzed: `latent_distance, raw_chamfer, rigid_registered_chamfer, nicp_correspondence`
- best metric on different-subject Spearman vs GT: `latent_distance` = `0.8433`
- same pairs: `180`
- different pairs: `300`

Artifacts:
- `translation/distribution_summary.csv`
- `translation/gt_quantile_summary.csv`
- `translation/rank_error_summary.csv`
- `translation/shrinkage_summary.csv`
- `translation/rank_inversion_examples.csv`
- `translation/gt_vs_metric_scatter.png`
- `translation/same_vs_diff_hist.png`
- `translation/shrinkage_vs_gt.png`
- `translation/rank_error_bar.png`
