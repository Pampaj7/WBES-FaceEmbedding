# Full Method Comparison

This bundle compares the completed REMESH mesh-pair benchmarks under two GT definitions:

- `original_gt`: the historical benchmark GT matrix
- `surrogate_gt_normalized`: a GT surrogate built directly in the normalized geometry space used by the loader

Main files:
- `scenario_method_comparison.csv`: scenario-level Spearman comparison across methods
- `raw_vs_rigid_flip_table.csv`: topology-pair level raw-vs-rigid comparison under both GTs
- `scenario_bars_original_gt.png`: macro comparison under original GT
- `scenario_bars_surrogate_gt.png`: macro comparison under surrogate GT

Winner flips raw vs rigid by scenario:
- `clean`: `4` topology-pair winner flips
- `translation`: `2` topology-pair winner flips

Interpretation target:
- If many topology-pair winners flip when the GT is moved into normalized space, the rigid-only conclusion is GT-sensitive rather than universally stable.
