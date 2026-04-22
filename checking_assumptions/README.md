# Checking Assumptions

This directory is a two-track investigation of the question:

"Why does raw Chamfer on perturbed meshes preserve the REMESH ranking better than perturbation + ICP?"

The work is intentionally split into two branches:

- `ifTrue/`: assume the observed result is real and try to confirm it from multiple angles.
- `ifFalse/`: assume the observed result is misleading or caused by a bug / protocol artifact and try to break it.

The goal is not to defend a preferred story. The goal is to make both stories compete against the same evidence.

## Layout

- `experiment_registry.csv`: living index of tests, status, outputs, and interpretation.
- `scripts/`: reproducible diagnostics launched from here.
- `outputs/`: generated CSV / JSON / MD reports from the diagnostics.
- `ifTrue/status.md`: current evidence in favor of the observed result being meaningful.
- `ifFalse/status.md`: current evidence in favor of the observed result being wrong, fragile, or misleading.

## Current Core Observation

On the existing REMESH cross-topology `mesh_pair_level` benchmark, using the same GT ranking:

- raw perturbed Chamfer > rigid-only ICP + Chamfer
- rigid-only ICP + Chamfer > rigid ICP + non-rigid CPD + Chamfer
- latent stays above all geometric variants

The job of this directory is to stress-test that conclusion from both sides.
