# WBES-FaceEmbedding Codebase Guide

Last updated from the current checkout on 2026-05-03.

This document is a practical guide to the repository as it exists on disk, not just as it is described in the README. The repo is a research workspace, not a cleanly packaged library. Code, datasets, experiment outputs, ad hoc utilities, and third-party material all live together.

The main goal of this guide is to help a new contributor answer five questions quickly:

1. What is this repository trying to do?
2. Which directories contain core code versus generated artifacts?
3. What are the main data formats and pipelines?
4. Which scripts are still useful entry points?
5. What will break if you try to run things on a different machine?

## 1. Executive Summary

At a high level, this repository is the official research workspace for the NeurIPS 2026 submission on identity-preserving 3D face embeddings under topology changes.

The current top model for the submission is:

- `face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1`

The main run/checkpoint inside that branch is:

- run directory: `face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best`
- checkpoint: `checkpoints/best_by_xtopo_mesh_clean.pth`
- config: `config.json`

At a technical level, this repository has one clear modeling center and three evaluation/validation branches:

- `face_embedding`: the main modeling branch, where 3D face identity embeddings are learned directly from meshes using DiffusionNet-style architectures, spectral variants, and robustness experiments.
- `faceBench/latentVSpipeline`: the paper-facing latent-vs-geometry branch, used to compare learned rankings against transparent FaceBench-style geometry pipelines.
- `datasets/FaceVerse`: the external FaceVerse validation branch, including downsampling, remesh10k topology variants, post-perturbation ICP benchmarks, sigma sweeps, and few-shot fine-tuning.
- `WBES`: the identity-effect-size branch, used to measure whether identity separability is actually preserved.

There is also a more isolated component:

- `BFM_to_FLAME`: a vendored external project used for topology/model conversion between Basel Face Model and FLAME.

If you only have time to understand one subtree, read `face_embedding/gt_encdec/` first. That is where most of the architecture decisions, representation-learning work, and topology-robust identity experiments live. Then read the current top run under `dn_mixed_topology_v1`, followed by `faceBench/latentVSpipeline/` and `datasets/FaceVerse/` for the current submission evaluation surface.

This is not a single end-to-end application with one orchestrator. It is closer to a research lab notebook in code form:

- some scripts are stable and reusable,
- some are one-off experiments,
- many outputs are saved next to the code that produced them,
- several scripts assume absolute local paths,
- multiple generations of approaches coexist.

The best way to understand the repo is to treat it as five layers:

1. Data preparation and topology generation
2. Operator precomputation and dataset adaptation
3. Mesh embedding / latent-space learning
4. Topology robustness and top-model selection
5. FaceBench, FaceVerse, and WBES validation

## 2. Current Checkout Snapshot

These are facts about the current workspace, not abstract expectations.

### 2.1 Top-level layout

Main directories:

- `faceBench/`
- `WBES/`
- `face_embedding/`
- `datasets/`
- `BFM_to_FLAME/`
- `tmp/`

### 2.2 Data and output volume currently present

The repository contains a lot of local data and generated artifacts that are normally ignored by git.

Current counts:

- `datasets/GT_ready/*.obj`: `4999`
- `datasets/GT_ready/npz_data/*.npz`: `4999`
- `datasets/GT_ready/npz_data_cropped_23470_with_ops/*.npz`: `4999`
- `datasets/REMESH/npz_data_topo_500/*.npz`: `3000`
- `datasets/REMESH/npz_data_topo_500_withops/*.npz`: `3000`
- `datasets/FaceVerse/downsampled_with_ops/*.npz`: `110`
- `datasets/FaceVerse/remesh10k_with_ops/*.npz`: `110`
- `datasets/FaceVerse/cross_topology_10k_with_ops/*.npz`: `220` symlink entries
- `datasets/FaceVerse/FINE_tuning/shot75_cross_topology_with_ops/*.npz`: `150` symlink entries
- `WBES/results/`: `159` files
- `WBES/results_landmarks/`: `134` files
- `WBES/plots/`: `127` files
- `face_embedding/gt_encdec/autoencoder/results_diffusionAE/`: `65` files
- `faceBench/latentVSpipeline/outputs/baseline_dn_mixed_topology_v1/`: `4` summary files
- `face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/figures/`: `10` figure files
- `datasets/FaceVerse/FINE_tuning/evals_mixedaug/`: `219` files

Observed REMESH variant split in `datasets/REMESH/npz_data_topo_500/`:

- `original`: `500`
- `remesh`: `500`
- `crop`: `500`
- `noisy`: `500`
- `down8k`: `500`
- `up60k`: `500`

Observed operator-enriched REMESH split in `datasets/REMESH/npz_data_topo_500_withops/`:

- `original`: `500`
- `remesh`: `500`
- `crop`: `500`
- `noisy`: `500`
- `down8k`: `500`
- `up60k`: `500`

This matters because the current top model and the FaceBench comparison both assume all six topology labels are available with operators. The current checkout now satisfies that assumption for REMESH.

### 2.2.1 Hugging Face dataset package

The no-operator mesh package is on Hugging Face:

- `Pampaj/wbes-faceembedding-noops`
- <https://hf.co/datasets/Pampaj/wbes-faceembedding-noops>

Published archives:

- `REMESH_npz_data_topo_500_noops.tar.zst`
  - REMESH, `500` subjects
  - `6` topology variants per subject
  - mesh-only `.npz` files with no intrinsic operators
- `FaceVerse_cross_topology_10k_noops.tar.zst`
  - FaceVerse cross-topology release
  - `original` and `remesh_10k` variants per subject
  - mesh-only `.npz` files with no intrinsic operators

Local export copies were observed under `hf_exports/`:

- `hf_exports/REMESH_npz_data_topo_500_noops.tar.zst`
- `hf_exports/FaceVerse_cross_topology_10k_noops.tar.zst`

This package should be described as the mesh-only dataset package. It is not the same as the operator-enriched training/evaluation directories used by the top DiffusionNet model, and it does not include checkpoints or heavy pair-level evaluation dumps. If the generated meshes are derived from upstream 3DMM assets such as BFM, check the upstream license before treating the package as a public redistribution artifact.

Related public artifact links:

- code repository: <https://github.com/Pampaj7/WBES-FaceEmbedding>
- selected model checkpoint: <https://hf.co/Pampaj/wbes-faceembedding-dn-mixed-topology-v1>
- full workspace artifact snapshot: <https://hf.co/datasets/Pampaj/wbes-faceembedding-repo-snapshot>

### 2.3 Stored Results Snapshot

The repository already contains enough saved artifacts to extract a few concrete observations.

The most important narrative point is that the model-side artifacts under `face_embedding/` already show strong embedding behavior, and the WBES artifacts then provide an independent evaluation view of the same identity-preservation question.

#### Current top submission model: `dn_mixed_topology_v1`

The current top model branch is:

- `face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1`

The selected run is:

- `mixed_xtopo_rank0p5_id0p25_bs5_best`

Important files:

- `config.json`
- `checkpoints/best_by_xtopo_mesh_clean.pth`
- `best_by_clean.txt`
- `best_by_auc.txt`
- `best_by_xtopo_mesh_clean.txt`
- `train_log.csv`
- `mixed_train_log.csv`
- `xtopo_mesh_log.csv`
- `robustness_grid.csv`

The selected checkpoint is also published on Hugging Face:

- model repo: `Pampaj/wbes-faceembedding-dn-mixed-topology-v1`
- URL: <https://hf.co/Pampaj/wbes-faceembedding-dn-mixed-topology-v1>
- uploaded files: `best_by_xtopo_mesh_clean.pth`, `config.json`, `best_by_xtopo_mesh_clean.txt`, `train_log.csv`, `xtopo_mesh_log.csv`
- Hub commit: <https://huggingface.co/Pampaj/wbes-faceembedding-dn-mixed-topology-v1/commit/c8e42d79d5606690e72d1091823a96b1f9e30726>

The run configuration is an `xyz_dn` DiffusionNet encoder with:

- latent dim: `256`
- width: `128`
- blocks: `4`
- pooling: `meanmax`
- epochs: `120`
- batch subjects: `5`
- training level: `mixed`
- training pair mode: `cross_topology`
- lambda rank: `0.5`
- lambda id: `0.25`
- lambda mesh: `1.0`
- noise probability: `0.6`
- noise modes: translation, rotation, jitter
- REMESH data root: `datasets/REMESH/npz_data_topo_500_withops`

Stored checkpoint-selection markers:

- `best_by_clean.txt`: `best_epoch=82`, clean Spearman `0.823418`
- `best_by_xtopo_mesh_clean.txt`: `best_epoch=82`, cross-topology mesh clean Spearman `0.748369`
- `best_by_auc.txt`: `best_epoch=16`, `best_auc_r=1.004153`

The online mesh evaluation summary records:

- `96` samples
- `16` subjects
- `3600` mesh pairs
- `6` topology labels: `crop`, `down8k`, `noisy`, `original`, `remesh`, `up60k`
- pair mode: `cross_topology`
- train level: `mixed`

At epoch `82`, the saved logs show:

- clean robustness Spearman: `0.823418`
- clean robustness Pearson: `0.846031`
- cross-topology mesh clean Spearman: `0.748369`
- cross-topology mesh Pearson: `0.774026`

At epoch `120`, the final logged values are still strong:

- clean robustness Spearman: `0.794944`
- clean robustness Pearson: `0.824230`
- cross-topology mesh clean Spearman: `0.734213`
- cross-topology mesh Pearson: `0.764397`

For the submission narrative, this should be treated as the current reference model unless a newer run is explicitly selected.

#### Top-model perturbation ranking results on REMESH

The stored REMESH ranking summary for the current top model is:

- `face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/perturbation_ranking_vs_chamfer/best_by_xtopo_mesh_clean_split-eval_pairs-cross_topology_agglvl-subject_pair_mean_subjects-100_meshes-10_scenarios-clean-jitter-translation-rotation-mixed/ranking_summary.csv`

It evaluates `100` subjects and `4950` subject pairs. Stored Spearman summaries:

- clean: latent `0.828020`, Chamfer `0.484707`, delta `0.343313`
- jitter: latent `0.817001`, Chamfer `0.446133`, delta `0.370868`
- translation: latent `0.825684`, Chamfer `0.480884`, delta `0.344800`
- rotation: latent `0.826496`, Chamfer `0.480426`, delta `0.346070`
- mixed: latent `0.799530`, Chamfer `0.432088`, delta `0.367442`

The stored `model_beats_chamfer` flag is `1` in all five scenarios.

The sigma sweep in the same branch covers jitter, translation, rotation, and mixed scenarios at sigmas:

- `0.00`
- `0.02`
- `0.05`
- `0.10`
- `0.15`
- `0.20`

For example, under jitter the latent Spearman decreases from `0.828020` at sigma `0.00` to `0.578171` at sigma `0.20`, while still beating the stored Chamfer Spearman at every jitter sweep point.

#### FaceBench latent-vs-geometry aggregate

The current FaceBench comparison summaries live in:

- `faceBench/latentVSpipeline/outputs/baseline_dn_mixed_topology_v1`

The method-level aggregate table reports:

- `raw_chamfer`: metric Spearman mean `0.338320`, latent Spearman mean `0.725293`, model-beats-metric rate `1.0`
- `rigid_chamfer`: metric Spearman mean `0.292456`, latent Spearman mean `0.729672`, model-beats-metric rate `1.0`
- `nicp_correspondence`: metric Spearman mean `0.186387`, latent Spearman mean `0.729672`, model-beats-metric rate `1.0`
- `rigid_cpd_chamfer`: metric Spearman mean `0.011102`, latent Spearman mean `0.729672`, model-beats-metric rate `1.0`

This is the cleanest artifact-backed summary for the paper claim that the learned latent ranking preserves identity structure better than several transparent geometry-only registration pipelines on the REMESH benchmark.

#### FaceVerse external validation snapshot

The FaceVerse branch is now populated enough to treat it as a substantial external validation surface.

Important directories:

- `datasets/FaceVerse/downsampled_with_ops`: `110` operator NPZ files
- `datasets/FaceVerse/remesh10k_with_ops`: `110` operator NPZ files
- `datasets/FaceVerse/cross_topology_10k_with_ops`: `220` original/remesh symlinked NPZ entries
- `datasets/FaceVerse/gt_distance_matrix`: FaceVerse GT distance matrices
- `datasets/FaceVerse/FINE_tuning`: few-shot fine-tuning manifests, runs, and held-out evaluations

Base full-neutral post-perturb ICP evaluation:

- path: `datasets/FaceVerse/faceverse_ranking_vs_gt_neutral_full_mixed_xtopo_9a81466d_best_by_xtopo_mesh_clean_postperturb_icp/ranking_summary.csv`
- subjects: `110`
- samples: `110`
- pairs: `5995`
- clean latent Spearman: `0.531864`
- clean Chamfer Spearman: `0.605202`
- mixed latent Spearman: `0.476996`
- mixed Chamfer Spearman: `0.638449`

This result is important because it shows nontrivial zero-shot transfer of the REMESH-trained model to FaceVerse, while also showing that Chamfer remains stronger in the within-topology full-neutral FaceVerse setting.

Cross-topology FaceVerse remesh10k post-perturb ICP evaluation:

- path: `datasets/FaceVerse/faceverse_xtopo10k_ranking_vs_gt_mixed_xtopo_9a81466d_best_by_xtopo_mesh_clean_postperturb_icp/ranking_summary.csv`
- subjects: `110`
- samples: `220`
- subject pairs: `5995`
- mesh pairs: `11990`
- clean latent Spearman: `0.114807`
- clean Chamfer Spearman: `0.598032`

This is a hard external cross-topology setting and should not be presented as a win for the zero-shot model. It is useful because it motivated the few-shot FaceVerse fine-tuning branch.

Few-shot FaceVerse fine-tuning:

- manifest: `datasets/FaceVerse/FINE_tuning/faceverse_finetune_manifest.json`
- available subjects: `110`
- shot datasets include `5`, `6`, `10`, `20`, `50`, `75`, and `100` subject variants in the current workspace
- selected 75-shot cross-topology dataset: `150` symlinked NPZ entries

One strong stored held-out result is:

- `datasets/FaceVerse/FINE_tuning/evals_mixedaug/shot75_heldout_xtopo10k_meshpair_pnoise0.6_sig5e-2_mixed5_id0.5_best_by_xtopo_mesh_clean_postperturb_icp/sigma_sweep_summary.csv`

It evaluates `35` held-out subjects, `70` samples, and `1190` subject pairs. At clean sigma:

- latent Spearman: `0.537433`
- Chamfer Spearman: `0.403222`
- delta: `0.134211`
- model-beats-Chamfer: `1`

At translation sigma `0.10`:

- latent Spearman: `0.485680`
- Chamfer Spearman: `0.416691`
- delta: `0.068990`
- model-beats-Chamfer: `1`

Under heavy jitter, the model degrades and Chamfer can overtake it. That is useful context for the robustness limits of the current FaceVerse fine-tuning branch.

#### WBES trends visible in stored CSVs

The mesh-level WBES results in `WBES/results/*/*-wbes_inter_F.csv` show the expected direction for every method that has more than one frame-group size:

- `3DDFAV2_23470_neutral`: WBES `0.4497 -> 1.3784` from `F=1` to `F=15`
- `3DDFAV3_neutral`: WBES `0.8707 -> 1.9182`
- `Deep3DFace_23470_neutral`: WBES `1.0888 -> 2.0070`
- `Faceverse_cropped_neutral`: WBES `0.6994 -> 1.8505`
- `Smirk_cropped_neutral`: WBES `0.9043 -> 2.3308`
- `SynergyNet_neutral`: WBES `0.2903 -> 1.1876`
- `INORig_23470_neutral`: WBES `1.6878 -> 2.0200` from `F=1` to `F=3`

The relative gain is especially large for:

- `SynergyNet_neutral`: about `4.09x`
- `3DDFAV2_23470_neutral`: about `3.07x`
- `Faceverse_cropped_neutral`: about `2.65x`
- `Smirk_cropped_neutral`: about `2.58x`

The landmark WBES files in `WBES/results_landmarks/*/*-wbes_landmark_inter_F.csv` show the same qualitative pattern:

- `3DDFAV3_neutral`: `0.8756 -> 1.9969`
- `Deep3DFace_23470_neutral`: `1.2630 -> 2.1422`
- `Smirk_cropped_neutral`: `0.9731 -> 2.3010`
- `Faceverse_cropped_neutral`: `0.8245 -> 1.7913`
- `SynergyNet_neutral`: `0.1718 -> 1.0492`

All stored WBES CSVs use `99` subjects, which matches the code-level hardcoded `BANNED_SUBJECTS = {"id0099"}` in the WBES scripts.

#### Geometry-error change is much smaller than WBES change

The stored `*-geom_error_vs_F.csv` files indicate that mean geometric error changes only slightly while WBES changes substantially. Examples:

- `3DDFAV3_neutral`: mean vertex error `0.0027083 -> 0.0026032`
- `Deep3DFace_23470_neutral`: `0.0026069 -> 0.0025459`
- `SynergyNet_neutral`: `0.0032642 -> 0.0032302`
- `INORig_23470_neutral`: `0.0030555 -> 0.0030457`

The landmark geometry summaries show the same pattern:

- `3DDFAV3_neutral`: mean landmark error `0.0020883 -> 0.0019793`
- `Deep3DFace_23470_neutral`: `0.0021704 -> 0.0020960`
- `SynergyNet_neutral`: `0.0028757 -> 0.0028727`

This is important because it supports the main claim of the project: identity separability can improve more strongly than pure geometric error would suggest.

#### Correlation summaries already stored under `WBES/plots/`

The repository contains summary CSVs for several cross-metric analyses.

From `WBES/plots/z_correlation_wbes_geom/wbes_vs_geomerror_summary.csv`:

- `3DDFAV3_neutral`: Pearson `-0.8838`
- `Deep3DFace_23470_neutral`: Pearson `-0.9384`
- `SynergyNet_neutral`: Pearson `-0.9020`
- `INORig_23470_neutral`: Pearson `-0.99999` on only `3` points

This means the stored summaries already encode a strong negative relationship between WBES and geometric error for several BFM-based methods.

From `WBES/plots/z_correlation_within_geom/wbes_vs_geomerror_summary.csv`:

- `3DDFAV3_neutral`: Pearson `0.9576`
- `Deep3DFace_23470_neutral`: Pearson `0.9002`
- `SynergyNet_neutral`: Pearson `0.9430`

This suggests that within-subject dispersion grows with geometric error, which is consistent with the WBES interpretation.

From `WBES/plots/z_correlation_geom_cosine/cosine_vs_geom_summary.csv`:

- `3DDFAV3_neutral`: Pearson `-0.9898`
- `Deep3DFace_23470_neutral`: Pearson `-0.9848`
- `SynergyNet_neutral`: Pearson `-0.9549`

From `WBES/plots/z_correlation_geom_complex/complex_vs_geom_summary.csv`:

- `3DDFAV3_neutral`: Pearson `-0.9719`
- `Deep3DFace_23470_neutral`: Pearson `-0.9873`
- `SynergyNet_neutral`: Pearson `-0.9705`

These are small-`n` summaries, so they should be treated as internal experiment evidence rather than final statistical claims.

#### Autoencoder evaluation artifacts

The strongest stored summary in the baseline autoencoder branch is:

- `face_embedding/gt_encdec/autoencoder/results_diffusionAE/eval_report.txt`

Its current contents report, for checkpoint `diffusionAE_5000_epoch45.pth` on `1000` evaluated samples:

- Spearman rho: `0.999`
- Diversity ratio: `1.077`
- HF retention mean: `0.976`
- Mean-face collapse ratio: `0.0%`
- Alive dims: `100.0%`

The companion `eval_metrics.csv` adds:

- `rank_top1_acc = 0.9`
- `individuality_mean = 1.1555`
- `latent_alive_ratio = 1.0`

The stored `mean_errors_53k.csv` currently contains `50` values with:

- mean: `0.01858`
- min: `0.01662`
- max: `0.02004`

There is also a useful artifact-level caveat:

- `results_diffusionAE/train_log.csv` currently contains only `3` rows and appears structurally malformed relative to its header, while the evaluated checkpoint is `epoch45`

So the evaluation artifacts look strong, but the logging artifact in that folder should not be treated as a clean full training history.

#### Older robustness run stored in `intrinsic/xyz_baseline`

An older saved robustness run is:

- `face_embedding/gt_encdec/remeshing/intrinsic/xyz_baseline/mxyz_dn_z256_w128_b4_ks100_poolmeanmax_bs4_tn0_pn0.00_s1.0e-04-3.0e-02_seed1234__d5cbb737/`

Its `config.json` describes:

- model: `xyz_dn`
- latent dim: `256`
- width: `128`
- blocks: `4`
- `k_spec = 100`
- `pool_mode = meanmax`
- `epochs = 50`
- `batch_subjects = 4`
- `p_noise = 0.0`
- evaluation over sigma range `1e-4 .. 3e-2`

Its `robustness_grid.csv` indicates:

- best clean epoch in the stored run: `epoch 46`
- clean Spearman at that epoch: `0.831891`
- clean Pearson at that epoch: `0.855226`
- worst noisy Spearman at that epoch: `0.831317`
- worst noisy ratio at that epoch: `0.999310`

At the final stored epoch `50`:

- clean Spearman: `0.829552`
- clean Pearson: `0.851018`
- noisy Spearman range across tested sigmas: `0.829634 .. 0.829969`
- noisy ratio range across tested sigmas: `1.000100 .. 1.000503`

In other words, this specific stored baseline run looks very stable across the tested perturbation range. Since the model is `xyz_dn`, `gate_mean` is `nan` throughout these artifacts, which is expected and not an error.

## 3. What Each Top-Level Directory Is For

## 3.1 `README.md`

The root README is a good conceptual overview, but it is not a reliable execution guide. It describes the research intent well, but many operational details live only in code.

Use the README to understand the thesis of the project.
Use this guide and the scripts themselves to understand how the repo actually runs.

## 3.2 `WBES/`

This is the identity-aware evaluation branch.

Subdirectories:

- `WBES/code/`: main WBES evaluation and plotting scripts
- `WBES/utils/`: topology indices, landmark index files, helper functions
- `WBES/results/`: mesh-based WBES outputs
- `WBES/results_landmarks/`: landmark-based WBES outputs
- `WBES/plots/`: generated figures and exploratory plots
- `WBES/raw_data/`: minimal raw inputs currently present

This area is mostly script-driven. There is no central package entry point. The scripts assume external study folders or local result folders already exist.

## 3.3 `faceBench/`

This is the FaceBench-based comparison branch. It is now important for the NeurIPS 2026 submission because it connects the learned latent ranking to transparent geometry baselines.

Main area:

- `faceBench/latentVSpipeline/`

What this branch does:

- builds REMESH pair tables
- runs the real `facebench` library, not only local DIY metrics
- computes raw Chamfer, rigid ICP, NICP, and registered correspondence distances
- compares those geometry-only rankings against the current top model's latent ranking
- aggregates already-produced ranking summaries
- produces distance-compression figures and density/scatter plots

Important files:

- `faceBench/latentVSpipeline/README.md`
- `faceBench/latentVSpipeline/run_facebench_remesh.py`
- `faceBench/latentVSpipeline/run_facebench_remesh_perturbed.py`
- `faceBench/latentVSpipeline/run_facebench_full_pipeline_perturbed.py`
- `faceBench/latentVSpipeline/summarize_existing_rankings.py`
- `faceBench/latentVSpipeline/compare_existing_methods.py`
- `faceBench/latentVSpipeline/analyze_distance_compression.py`
- `faceBench/latentVSpipeline/plot_distance_compression_png.py`

The branch is currently pinned to the top model:

- `face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/checkpoints/best_by_xtopo_mesh_clean.pth`

Use `faceBench/latentVSpipeline/README.md` for the most direct commands.

## 3.4 `face_embedding/`

This is the mesh embedding and learning area, and it is the real center of gravity of the repository.

Main subtree:

- `face_embedding/gt_encdec/`

Important subareas:

- `alignment/`: preparation of GT-ready meshes
- `autoencoder/`: baseline DiffusionNet autoencoder, encoder-only variants, latent analysis
- `mse/`: pairwise GT distance and alignment sanity checks
- `remeshing/`: topology robustness, cross-topology, intrinsic, and voxel experiments

This is the densest part of the repo in terms of model experimentation.
If someone asks where the main technical substance is, this is the answer.

## 3.5 `datasets/`

This directory is both:

- a large local data store, and
- a place containing utility scripts for data conversion, cropping, remeshing, previewing, and validation.

Subdirectories:

- `datasets/GT_ready/`
- `datasets/REMESH/`
- `datasets/FaceVerse/`

Root-level scripts in `datasets/` are important and still useful.

## 3.6 `BFM_to_FLAME/`

This is effectively a vendored external repository. It contains its own nested `.git` directory, its own README, model data, and helper libraries such as `psbody_mesh` and `smpl_webuser`.

It should be treated as a related utility, not as a core part of the WBES or DiffusionNet training pipeline.

## 3.7 `tmp/`

Scratch space for temporary experiment outputs, probes, smoke tests, and diagnostics.

Examples include:

- closure checks
- FaceVerse inspection outputs
- ranking summaries
- jitter probes
- operator checks

Do not treat `tmp/` as source of truth for the main pipeline. It is best seen as exploratory residue.

## 4. Mental Model of the Whole Repository

The easiest mental model is:

### 4.1 Ground-truth preparation

Raw meshes are aligned and normalized into a canonical "GT_ready" representation.

### 4.2 Format conversion

Those meshes are converted from `.obj` into `.npz` so downstream scripts can load geometry quickly.

### 4.3 Optional cropping / topology transformation

The GT data is cropped to a canonical face region and then used to generate topology variants:

- original
- remesh
- crop
- noisy
- down8k
- up60k

### 4.4 Operator precomputation

DiffusionNet spectral operators are precomputed and stored into enriched `.npz` files.

### 4.5 Main modeling branch

The core branch is:

- train mesh encoders / autoencoders
- build identity-sensitive latent spaces
- compare latent structure against geometric structure

### 4.6 Robustness branch

The most interesting extension is that the repo does not stop at clean canonical meshes.
It studies whether embeddings remain meaningful when topology changes, noise is added, or geometry is perturbed.

### 4.7 Evaluation branch

The current evaluation layer is split across three branches.

FaceBench acts as the transparent geometry-comparison layer:

- raw geometry
- rigid registration
- non-rigid registration/correspondence
- distance-compression analysis

FaceVerse acts as the external dataset stress test:

- zero-shot transfer to a different face model family
- 10k remesh cross-topology variants
- post-perturbation ICP baselines
- few-shot fine-tuning and held-out evaluation

WBES acts as the explicit identity-effect-size layer:

- quantify within-subject versus between-subject separation
- compare identity separability against geometric error
- check whether better geometry actually implies better identity

## 5. Data Contracts and File Formats

This repo uses a few recurring mesh formats. Understanding them is critical.

## 5.1 OBJ meshes

Used especially in:

- `datasets/GT_ready/*.obj`
- alignment and legacy preprocessing scripts

These are triangle meshes on disk and are often the first "clean" aligned representation.

## 5.2 Plain NPZ geometry files

Two key conventions exist:

- `verts` / `faces`
- `V` / `F`

Both appear across the repo, and utility scripts usually support both.

Typical plain geometry NPZ payload:

```text
verts: float array [N, 3]
faces: int array [M, 3]
```

or

```text
V: float array [N, 3]
F: int array [M, 3]
```

## 5.3 DiffusionNet-operator NPZ files

Operator-enriched NPZ files usually contain:

```text
verts or V
faces or F
mass
evals
evecs
L_indices
L_values
L_shape
gradX_indices
gradX_values
gradX_shape
gradY_indices
gradY_values
gradY_shape
```

These files are the main input for DiffusionNet-based training and inference.

## 5.4 WBES input meshes

The WBES scripts primarily expect:

- `.txt` full-mesh files for mesh-level WBES
- `.npy` landmark arrays for landmark-level WBES

Important detail:

`WBES/code/WBES_pipeline.py` loads mesh data from `.txt`, not from `.npz`.

## 5.5 Naming convention

Subject naming usually follows:

```text
id0000_GTready_original.npz
id0000_GTready_remesh.npz
id0000_GTready_down8k.npz
```

The subject ID is almost always parsed from the filename stem. Many scripts assume names contain tokens like `id0000`.

## 6. Main Directory-by-Directory Guide

## 6.1 `datasets/`

This is one of the most important directories despite being gitignored in principle.

### `datasets/obj_to_npz.py`

Purpose:

- converts GT-ready `.obj` files into compressed `.npz`

Input:

- `datasets/GT_ready/*.obj`

Output:

- `.npz` files saved next to the OBJ files

Notes:

- stores keys as `verts` and `faces`
- uses `trimesh`

### `datasets/crop.py`

Purpose:

- crops full GT NPZ meshes to a canonical subset of vertices using a fixed index file

Input:

- `datasets/GT_ready/npz_data`

Output:

- `datasets/GT_ready/npz_data_cropped`

Important index source:

- `WBES/utils/ix_23470_relative_to_53215.txt`

Notes:

- output keys are `V` and `F`
- optional HTML preview generation exists but is disabled by default

### `datasets/remesh.py`

Purpose:

- creates topology variants from cropped GT NPZ meshes

Generated variants:

- original
- remesh
- crop
- noisy

Output directory:

- `datasets/REMESH/npz_data_topo_500`

Notes:

- uses Open3D
- currently described as a smaller-scale test-style generator with `N_SUBJECTS = 500`
- does not cover the later `down8k` and `up60k` variants

### `datasets/expand_remesh_topologies.py`

Purpose:

- extends the REMESH dataset with stronger topology variants:
  - `down8k`
  - `up60k`

Key behavior:

- starts from full GT meshes
- crops them using `ix_23470_relative_to_53215.txt`
- generates canonical mesh
- decimates or subdivides + decimates
- tries to patch small boundary loops to improve topology quality

Key defaults:

- GT source: `datasets/GT_ready/npz_data`
- output: `datasets/REMESH/npz_data_topo_500`
- subjects: first 500 by default

This is a more mature topology-generation script than `datasets/remesh.py`.

### `datasets/render_mesh_preview.py`

Purpose:

- generates interactive HTML mesh previews with topology diagnostics

Capabilities:

- renders mesh surface
- overlays wireframe
- highlights boundary edges
- highlights non-manifold edges
- reports:
  - vertex count
  - face count
  - boundary edge count
  - non-manifold edge count
  - degenerate faces
  - bad indices

This is one of the most useful utility scripts in the repo for quick mesh QA.

### `datasets/compare_operator_closure_effect.py`

Purpose:

- compare DiffusionNet operators before and after closing boundary loops

Why it matters:

- topology closure can affect spectral operators
- this script quantifies that effect numerically

This is a diagnostic script for topology-operator sensitivity.

### `datasets/view_remeshed.py`

Purpose:

- likely an interactive or visualization-oriented remesh inspection helper

Important note:

- this file is currently modified in the local worktree
- treat the current local version as user-owned unless you intend to update it carefully

### `datasets/FaceVerse/`

This subtree has become one of the main external-validation branches for the submission.

Important scripts:

- `downsample_faceverse.py`
- `validate_downsampled_faceverse.py`
- `compare_downsampled_suffix_groups.py`
- `remesh_faceverse_from_npz.py`
- `assemble_faceverse_cross_topology_dataset.py`
- `compare_model_vs_chamfer_rankings_faceverse.py`
- `compare_model_vs_chamfer_rankings_faceverse_sigma_sweep.py`
- `compare_faceverse_result_dirs.py`
- `plot_downsampled_mesh.py`
- `plot_subject_suffix_grid.py`
- `FINE_tuning/prepare_faceverse_finetune.py`

What this branch does:

- downsample FaceVerse PLY meshes using libigl to about 10k vertices
- validate counts and alignment statistics
- precompute DiffusionNet operators for FaceVerse meshes
- create remesh10k topology variants through Open3D reconstruction/decimation
- assemble original/remesh cross-topology datasets through symlinks
- compare learned ranking behavior versus Chamfer-like baselines and post-perturbation ICP
- run progressive sigma sweeps
- prepare few-shot fine-tuning datasets where FaceVerse subjects are exposed as `idXXXX`
- evaluate held-out FaceVerse cross-topology transfer

Current local state:

- `downsampled_with_ops`: `110` operator NPZ files
- `remesh10k_with_ops`: `110` operator NPZ files
- `cross_topology_10k_with_ops`: `220` symlinked original/remesh NPZ entries
- `gt_distance_matrix`: present
- `FINE_tuning/faceverse_finetune_manifest.json`: present, with `110` available subjects
- `FINE_tuning/shot75_cross_topology_with_ops`: `150` symlinked NPZ entries
- `FINE_tuning/evals_mixedaug`: populated with held-out sigma sweeps

The raw extracted FaceVerse PLY directory is not present in this checkout, but the operator-ready downstream assets needed by the current evaluation scripts are present.

## 6.2 `WBES/`

This directory is the evaluation branch. The scripts are relatively flat and mostly run directly.

### `WBES/utils/WBES_helper.py`

This is the main reusable helper module for the WBES branch.

It provides:

- landmark loading
- Procrustes-like landmark-based alignment
- vertexwise L2 error
- Cohen's d computation
- mesh loading from `.txt`
- landmark loading from `.npy`
- generation of disjoint repeated averages with `reps_disjoint()`
- safe KDE plotting helper

This file is the closest thing WBES has to a shared utility layer.

### `WBES/code/WBES_pipeline.py`

Purpose:

- compute mesh-based WBES for each reconstruction method

What it does:

- loads per-subject reconstructed meshes from `.txt`
- builds repeated averages over frame groups
- computes within-subject and between-subject distances
- computes Cohen's d as WBES
- saves CSV and `.npy` distributions
- for BFM-topology methods, optionally compares mean reps against GT using landmark-based alignment

Code-level details that matter:

- methods hardcoded in `METHOD_DIRS`:
  - `3DDFAv2`
  - `3DDFAv3`
  - `Deep3DFace`
  - `SynergyNet`
  - `INORig`
  - `3DI`
  - `Smirk`
  - `FaceVerse`
- most methods use frame groups `[1, 3, 5, 10, 15]`
- `INORig` uses `[1, 2, 3]`
- `3DI` uses `[1]`
- `N_REPS = 3`
- `BANNED_SUBJECTS = {"id0099"}`
- BFM GT comparison is enabled only for:
  - `3DDFAv2`
  - `3DDFAv3`
  - `Deep3DFace`
  - `SynergyNet`
  - `INORig`
  - `3DI`

Important assumptions:

- `STUDY_ROOT` is empty and must be edited
- GT paths are hardcoded to `/Users/pampaj/...`
- BFM landmark JSON path is hardcoded to `/Users/pampaj/...`

Meaning:

- the logic is important,
- but the script is not portable as-is.

### `WBES/code/WBES_pipeline_multi.py`

Purpose:

- extended WBES script with per-subject outputs

Additional outputs:

- per-subject within values
- per-subject WBES values
- per-subject geometry error CSVs

This is probably the better starting point if you want richer WBES outputs than the original pipeline.

### `WBES/code/WBES_pipeline_landmarks.py`

Purpose:

- landmark-only WBES pipeline

Differences from mesh WBES:

- loads `.npy` landmark arrays instead of full mesh `.txt`
- computes WBES over landmark geometry
- writes to `WBES/results_landmarks`

This is useful when topology mismatch makes full-mesh comparisons harder, but you still want identity-separation metrics.

### `WBES/code/landmark_extractor.py`

Purpose:

- generate 51-landmark subsets for different topology families

Supported topology families:

- BFM methods
- FLAME methods
- FaceVerse methods

Inputs:

- `.txt` full meshes

Outputs:

- `_lmk.npy` files

Important caveat:

- one path is hardcoded to `/home/pampalonil/data/utils/faceverse_lmk_indices_51_cropped.npy`

### Other WBES plotting and analysis scripts

These scripts mainly consume outputs already written by the main pipelines:

- `WBES_within_bet.py`
- `line_plot.py`
- `line_plot_geom_vs_WBES.py`
- `grid_plot.py`
- `plot_geom_wbes_persubj.py`
- `plot_geom_within_persubj.py`
- `correlation_wbes_geom.py`
- `correlation_within_geom.py`
- `correlation_cosine_geom.py`
- `correlation_geom_cv.py`
- `correlation_wbes_complex.py`
- `correlation_byF_across_methods.py`
- `tSNE.py`
- `birkan_plot.py`
- `adapt.py`
- `test_error.py`

Best way to think about them:

- the WBES branch has one core metric pipeline,
- several plotters and correlation scripts fan out from its CSV/NPY outputs.

## 6.3 `face_embedding/gt_encdec/alignment/`

### `prepare_GT_ready.py`

Purpose:

- align raw synthetic meshes into a "GT_ready" space

What it does:

- loads meshes from a raw source folder
- chooses a reference mesh
- performs rigid alignment using `trimesh.registration.procrustes`
- computes a global centroid offset
- exports centered OBJ meshes
- runs a geometric verification pass on a subset

Important properties:

- batch-oriented to reduce memory pressure
- uses explicit RAM cleanup and `malloc_trim()`
- highly path-dependent

Hardcoded source:

- `../../../render3d_Leonardo/data_creator/synthetic_meshes`

Hardcoded target:

- `../../../datasets/GT_ready/`

This is a foundational script for the GT pipeline, but it is tied to a local environment.

## 6.4 `face_embedding/gt_encdec/autoencoder/`

This is the baseline learning branch for mesh embeddings and reconstruction.
It is also one of the most important directories in the whole repo.

### What this area contains

It mixes:

- model definitions
- dataset loaders
- training scripts
- debugging helpers
- latent analysis scripts
- saved checkpoints
- example outputs

Because this directory stores both source and results, read it carefully.

### `dataset_gtready.py`

This is one of the most important files in the whole repo.

It defines two dataset classes:

- `GTReadyDataset`: legacy loader
- `GTReadyDatasetNPZ`: current NPZ-based loader

The NPZ loader:

- reads geometry and precomputed operators
- normalizes geometry per sample
- reconstructs sparse operators from COO triplets
- rescales spectral values and operators numerically
- performs finite-value checks
- returns a dictionary compatible with DiffusionNet training

This file is the main data adapter between precomputed mesh/operator assets and neural training code.

### `diffusion_autoencoder.py`

This is the main model-definition file.

It contains multiple model families, including:

- `DiffusionAutoencoder`
- `DiffusionEncoderOnly`
- `DiffusionEncoderOnlyIntrinsec`
- `DiffusionEncoderXYZSpectrum`
- `TwoTowerGatedRobust`
- `TwoTowerConcat`

Key design idea in `DiffusionAutoencoder`:

- encoder produces a per-vertex latent field
- decoder reconstructs geometry from local latent field plus spectral basis
- a pooled global latent is returned for analysis, not used directly for reconstruction

This separation between:

- local latent for reconstruction
- global latent for identity analysis

is a central design principle of the embedding branch.

#### Deep Focus: why `diffusion_autoencoder.py` matters so much

This is arguably the single most central model file in the whole repository.

Observed coupling in the current checkout:

- `35` Python files import from `diffusion_autoencoder.py`
- `DiffusionEncoderOnly` alone is instantiated in `21` files
- `DiffusionAutoencoder` is instantiated in `9` files

That means this file is not just "the autoencoder definition". It is the shared model zoo used across:

- baseline autoencoder training
- latent-analysis scripts
- cross-topology experiments
- intrinsic experiments
- voxel/grid-based probes
- robustness-aware training via `robustness/model_helpers.py`

In practice, if you change an interface in this file, you are touching a large fraction of the research codebase.

#### What the file really contains

The file is organized as a progression of architectural variants:

1. `TwoTowerGatedRobust`
2. `TwoTowerConcat`
3. `DiffusionAutoencoder`
4. `DiffusionEncoderOnly`
5. `DiffusionEncoderOnlyIntrinsec`
6. `DiffusionEncoderXYZSpectrum`

So despite its filename, this is not only an autoencoder file. It is the canonical repository of encoder variants.

#### Shared design language across the classes

Most classes in this file follow the same high-level template:

- use `DiffusionNet` as the main geometric backbone
- operate on operator-enriched mesh inputs:
  - `V`
  - `mass`
  - `L`
  - `evals`
  - `evecs`
  - `faces`
  - `gradX`
  - `gradY`
- produce a per-vertex latent field first
- apply a small per-vertex MLP bottleneck
- optionally add Gaussian latent noise with scale `0.01`
- pool to a global embedding when needed

That recurring pattern is important because it makes the model family internally coherent even though several research directions coexist.

#### `DiffusionAutoencoder`: what is special about it

`DiffusionAutoencoder` is the only class in the file that explicitly reconstructs geometry.

Its defining choices are:

- encoder input is pure XYZ
- encoder output is a per-vertex latent field
- decoder input is:
  - per-vertex latent field
  - truncated spectral basis from `evecs`
- global latent is computed only after encoding, by mean pooling
- the global latent is not fed back into the decoder

This is a strong architectural statement:

- reconstruction is driven by local structure plus intrinsic basis
- identity analysis is driven by a pooled summary of local features

So the code deliberately decouples:

- "what reconstructs the surface"
- from "what is used as the identity embedding"

That choice explains many downstream analysis scripts in `latent_analysis/`.

#### `DiffusionEncoderOnly`: why it appears everywhere

`DiffusionEncoderOnly` is the simplest reusable encoder in the file:

- no decoder
- same DiffusionNet encoder backbone
- same bottleneck
- optional noise
- global latent from mean pooling
- optional return of the full per-vertex latent field

This makes it ideal for:

- latent distance studies
- topology-transfer experiments
- voxel/grid pooling
- ranking and confusion tests

Its broad reuse across the repo is a good sign that this class became the de facto baseline embedding interface.

#### `DiffusionEncoderOnlyIntrinsec`: what it adds

This class extends the encoder-only idea with optional intrinsic descriptors:

- XYZ features
- HKS
- WKS

Important implementation details:

- it truncates eigenpairs for stability
- it computes HKS/WKS internally from `evals` and `evecs`
- it supports `mean` and `meanmax` pooling

This is the branch to read if you want to understand how the repo moved from pure extrinsic geometry toward intrinsic spectral descriptors.

#### `DiffusionEncoderXYZSpectrum`: why it exists separately

This class injects the Laplacian spectrum as a global descriptor replicated per vertex and concatenated to XYZ.

It is conceptually different from `DiffusionEncoderOnlyIntrinsec`:

- `DiffusionEncoderOnlyIntrinsec` uses intrinsic descriptors derived per vertex from eigenpairs
- `DiffusionEncoderXYZSpectrum` uses the global eigenvalue spectrum itself as a replicated conditioning feature

That distinction matters because the repo explores both:

- intrinsic geometry descriptors
- and global spectral identity priors

#### `TwoTowerGatedRobust` and `TwoTowerConcat`

These are the main architectural bridge into the robustness branch.

Both split the representation into two streams:

- an extrinsic XYZ DiffusionNet tower
- a spectrum MLP tower

They differ only in fusion:

- `TwoTowerGatedRobust` uses learned per-dimension sigmoid gating
- `TwoTowerConcat` uses concatenation followed by a fusion layer

These two classes are the architectural core behind the "two-tower" naming used elsewhere in the repo.

#### What to edit in this file versus elsewhere

Edit `diffusion_autoencoder.py` if you want to change:

- encoder architecture
- pooling strategy
- whether and where latent noise is injected
- fusion between XYZ and spectral information
- the forward interface shared by downstream experiments

Do not edit this file if you only want to change:

- training losses
- run configuration
- noise schedules
- evaluation metrics
- run directory conventions

Those belong elsewhere:

- losses in `geometric_loss.py` or `latent_loss.py`
- training loops in the relevant training runner
- robustness behavior in `intrinsic/robustness/`

### `train_autoencoder.py`

Purpose:

- baseline training for the DiffusionNet autoencoder

Current behavior:

- loads `GTReadyDatasetNPZ`
- points to `datasets/GT_ready/npz_data_cropped_23470_with_ops/`
- uses only the first 3000 samples
- splits train/validation
- logs to CSV and TensorBoard
- saves checkpoints every 5 epochs

Important caveats:

- output directory is currently hardcoded as `howwwwwwwww`
- this script is operational but clearly still in a research/prototyping state

Code-level baseline hyperparameters in the current file:

- `LATENT_DIM = 256`
- `WIDTH = 128`
- `N_BLOCKS = 4`
- `EPOCHS = 50`
- `LR = 1e-4`
- `BATCH_SIZE = 16`
- `VAL_SPLIT = 0.1`
- loss weights:
  - `W_L1 = 0.3`
  - `W_NORMAL = 1.0`
  - `W_LAPLACIAN = 0.7`

Artifact-level observation:

- the saved evaluation folder `results_diffusionAE` looks stronger and more complete than the currently checked-in `train_autoencoder.py` output configuration, which suggests part of the best run history may have been produced from an earlier or slightly different training script state

### `precompute_operators_npz.py`

Purpose:

- compute DiffusionNet operators for NPZ or mesh inputs

This is a key infrastructure script. It creates the enriched NPZ format used across the learning code.

Default input:

- `datasets/REMESH/npz_data_topo_500`

Default output:

- `datasets/REMESH/npz_data_topo_500_withops`

Important dependency:

- local DiffusionNet source tree under `/equilibrium/lpampaloni/diffusion-net/src`

### `geometric_loss.py` and `latent_loss.py`

These files define the training losses for:

- mesh reconstruction
- latent-space smoothness / structure / stress constraints

They are central to both the baseline autoencoder and later robustness experiments.

### `latent_analysis/`

This directory contains analysis utilities for the learned latent space.

Representative scripts:

- `extract_latents.py`
- `extract_latents_stage1.py`
- `compute_latent_dist.py`
- `build_distance_matrices.py`
- `compute_gt_distance_matrix_normalized.py`
- `analyze_latent_correlations.py`
- `plot_latent_vs_gt.py`
- `latent_interpolation.py`
- `latent_perturbation_test.py`
- `latent_retrival.py`
- `visualize_error_map.py`
- `kde_shift_tests.py`

Purpose of this branch:

- compare learned latent distances against GT geometric distances
- inspect whether latent spaces preserve subject-level structure
- visualize reconstruction error and latent separability

### Other notable scripts

- `check_model.py`: likely load-and-inspect helper
- `debug.py`: debugging entry point
- `decoder_only.py`, `encoder_only.py`, `decoder_stage2_from_latents.py`, `encoder_decoder.py`: variant experiments

These indicate that the autoencoder branch evolved through multiple intermediate designs rather than one linear model pipeline.

## 6.5 `face_embedding/gt_encdec/mse/`

This is a simpler metric branch focused on pairwise geometric distances.

Key scripts:

- `compute_pairwise_mse.py`
- `test_align.py`

This area seems to provide:

- baseline geometric pairwise distance computation
- alignment sanity checks

It is useful context for latent-vs-geometry comparisons.

## 6.6 `face_embedding/gt_encdec/remeshing/`

This subtree is about topology robustness.

It has several sub-branches:

- `analysis/`
- `cross_topo_model/`
- `intrinsic/`
- `voxel/`

### 6.6.1 Cross-topology branch

This branch contains models such as:

- `encoder_1stage_topo.py`
- `decoder_2stage_topo.py`
- `encoder_intrinsic_coords.py`
- `encoder_pos_invariant.py`

This area appears to investigate whether latent identity structure can be made less sensitive to topology changes.

Much of its corresponding output directories are gitignored.

### 6.6.2 Intrinsic branch

This is likely the most actively structured robustness area in the repo.

Important files:

- `intrinsic_utils.py`
- `train_diffusion_xyz_spectrum.py`
- `train_twotower_dn_spec_robust.py`
- `hks_spectral_l2_baseline.py`
- `hks_intrinsic_diagnostics.py`
- `evaluate_latent_vs_chamfer_misalignment.py`
- `evaluate_xyz_breakdown.py`
- `spectral_mlp_ranking.py`
- `sweep_intrinsic_spectral_configs.py`
- `two_path_intrinsic_report.py`

#### Deep Focus: `train_twotower_dn_spec_robust.py`

This file deserves special emphasis because its name suggests "main training implementation", but in the current repo that is no longer true.

What the file is now:

- a thin compatibility entry point
- `68` lines long in the current checkout
- primarily an import-and-reexport facade
- ends with `if __name__ == "__main__": main()`

What it is not anymore:

- the place where the optimization loop lives
- the place where the evaluation logic lives
- the place where noise injection is implemented
- the place where the model factory is defined

Its real purpose is backward compatibility.

#### What it actually re-exports

The file re-exports symbols from `intrinsic/robustness/`, including:

- `parse_args`
- `run_training`
- `main`
- `make_run_dir`
- `build_model`
- `forward_model`
- `smooth_term_from_model`
- `evaluate_robustness_grid`
- `evaluate_at_sigma`
- `_build_sigma_grid`
- `_ratio_auc`
- `PerturbationParams`
- `apply_xyz_perturbation`
- `parse_noise_modes`
- `sample_log_uniform_sigma`
- `sample_to_device`

That means legacy code can still import from:

- `train_twotower_dn_spec_robust.py`

even though the actual implementation has been moved into modular subfiles.

#### Why this wrapper exists

Historically, the robustness branch likely started as a monolithic training script. The current file preserves the old import surface while delegating to a more structured package.

This is a good refactor pattern in a research repo:

- old notebooks and scripts do not immediately break
- new code can grow in a cleaner internal structure

#### Where the real implementation lives now

If you run `train_twotower_dn_spec_robust.py`, the actual work is delegated into:

- `robustness/train_runner.py`
- `robustness/model_helpers.py`
- `robustness/eval_utils.py`
- `robustness/noise.py`
- `robustness/data_utils.py`
- `robustness/paths.py`

These files divide responsibility cleanly.

#### Responsibility split behind the wrapper

`robustness/train_runner.py`

- real argument parsing
- run directory creation
- checkpoint loading and teacher-model loading
- training loop
- logging to CSV
- online robustness evaluation
- checkpoint selection and saving

This is the true engine of the training pipeline.

`robustness/model_helpers.py`

- model factory for:
  - `xyz_dn`
  - `intrinsic_dn`
  - `xyz_spec_dn`
  - `gated_twotower`
  - `concat_twotower`
  - `spec_mlp`
- normalization of gate outputs
- unified `forward_model()` adapter
- optional smoothness term extraction

This is where model selection logic lives.

`robustness/noise.py`

- parse supported perturbation modes
- parse per-mode weights
- sample sigma log-uniformly
- apply perturbations:
  - jitter
  - rigid
  - rotation
  - translation
  - outliers

This is where geometric corruption policy is defined.

`robustness/eval_utils.py`

- evaluate subject embeddings at fixed sigma
- evaluate full robustness grids
- aggregate Spearman, Pearson, ratio, gate statistics
- compute AUC-like robustness summaries

This is where "is the embedding stable under perturbation?" is actually measured.

`robustness/data_utils.py`

- dataset-side loading and sample movement
- subject split helpers
- preparation of fixed evaluation contexts

`robustness/paths.py`

- canonical default dataset locations
- GT distance-matrix path
- run-root defaults

#### What to edit if you want to change behavior

If you want to change CLI behavior or run orchestration:

- edit `robustness/train_runner.py`

If you want to change model choice or model dispatch:

- edit `robustness/model_helpers.py`

If you want to change perturbation families or sigma behavior:

- edit `robustness/noise.py`

If you want to change robustness metrics or grid evaluation:

- edit `robustness/eval_utils.py`

If you want to change the old public entry surface:

- edit `train_twotower_dn_spec_robust.py`

In other words, `train_twotower_dn_spec_robust.py` is now the door, not the machinery behind the door.

#### Deep Focus: `newdata/dn_mixed_topology_v1`

This is the current top-model branch for the NeurIPS 2026 submission.

Main directory:

- `face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1`

The selected run is:

- `mixed_xtopo_rank0p5_id0p25_bs5_best`

Why it matters:

- it trains on the six REMESH topology variants with operators
- it uses mixed cross-topology training
- it stores both robustness-grid and cross-topology mesh logs
- downstream FaceBench and FaceVerse scripts point to its `best_by_xtopo_mesh_clean.pth` checkpoint

Key result files:

- `best_by_clean.txt`
- `best_by_auc.txt`
- `best_by_xtopo_mesh_clean.txt`
- `train_log.csv`
- `mixed_train_log.csv`
- `xtopo_mesh_log.csv`
- `robustness_grid.csv`
- `perturbation_ranking_vs_chamfer*/`
- `perturbation_ranking_vs_registered_chamfer_topology_breakdown/`
- `perturbation_ranking_vs_nicp_correspondence_topology_breakdown/`
- `figures/`

Do not treat this directory as just another experiment folder. In the current workspace, it is the reference model/output bundle for paper-facing evaluation.

#### `intrinsic_utils.py`

This is a core utility file for the intrinsic branch.

It provides:

- subject ID extraction
- subject splits
- GT distance-matrix loading
- ranking/correlation utilities
- pairwise distance utilities
- rank-based latent loss helpers

This file is one of the best indicators that the intrinsic branch is becoming more organized than the older script-only parts of the repo.

#### `train_twotower_dn_spec_robust.py`

This file is now a lightweight compatibility entry point. The real implementation lives under:

- `face_embedding/gt_encdec/remeshing/intrinsic/robustness/`

That is a good sign: this branch has started to refactor research code into a more modular structure.

### 6.6.3 `intrinsic/robustness/`

This is currently one of the cleanest subpackages in the repo.

Important files:

- `paths.py`
- `data_utils.py`
- `noise.py`
- `model_helpers.py`
- `eval_utils.py`
- `train_runner.py`
- `posthoc_runner.py`

What it does:

- builds subject-wise training/evaluation splits
- injects geometric perturbations
- supports multiple model families
- trains robustness-aware intrinsic encoders
- evaluates embedding stability over sigma sweeps

#### `paths.py`

Defines important defaults:

- default training data:
  - `datasets/REMESH/npz_data_topo_500_withops`
- default GT distance matrix:
  - `face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz`
- default runs root:
  - `face_embedding/gt_encdec/remeshing/intrinsic/perturbated`

#### `model_helpers.py`

This file is the model factory for robustness experiments.

Supported model modes include:

- `xyz_dn`
- `intrinsic_dn`
- `xyz_spec_dn`
- `gated_twotower`
- `concat_twotower`
- `spec_mlp`

This is a good place to start if you want to know which encoder family is currently considered "official" for robustness experiments.

#### `train_runner.py`

This is the real backbone of the robustness-aware training pipeline.

It provides:

- argument parsing
- run directory creation
- config fingerprinting
- checkpoint organization
- online robustness evaluation
- noise scheduling
- subject-level training loops

Compared to older scripts, this file is much closer to a maintainable training runner.

The current checkout also contains at least one concrete run under `intrinsic/xyz_baseline/`, so this branch is not only code scaffolding; it already has saved metrics, configs, and a full 50-epoch training log.

### 6.6.4 Voxel branch

This branch is experimental and spatial/intrinsic pooling oriented.

Important files:

- `canonical_data.py`
- `check_alignment.py`
- `laplace_beltrami/LB_voxelization.py`
- `laplace_beltrami/LB_voxel_inference_test.py`
- `test/run_grid_inference.py`
- several plotting and evaluation scripts in `voxel/test/`

#### `canonical_data.py`

Purpose:

- normalize topology variants per subject using the `original` variant as reference

Output:

- `datasets/REMESH/data_CANONICAL`

This makes cross-variant comparisons easier.

#### `check_alignment.py`

Purpose:

- verify canonicalized variants by comparing centroid, scale, and ICP-like RMS

This is a quality check for the canonicalization step.

#### `LB_voxelization.py`

Purpose:

- define intrinsic "voxels" by quantizing spectral coordinates

Use case:

- pool per-vertex latents in intrinsic spectral regions instead of Euclidean grid cells

#### `test/run_grid_inference.py`

Purpose:

- inference-only probe using grid-based pooling for identity-distance comparisons

This is clearly an experiment/probe, not a production pipeline.

## 6.7 `BFM_to_FLAME/`

This is a separately structured utility repo embedded into the workspace.

Based on its own README, it is intended to:

- create a FLAME texture model from BFM color space
- convert BFM meshes to FLAME topology

Important files:

- `mesh_convert.py`
- `col_to_tex.py`
- `conv.py`
- `mesh_convert.py`
- `data/BFM_to_FLAME_corr.npz`
- `model/model2017-1_bfm_nomouth.h5`

Dependencies in this subtree are older and specialized:

- `chumpy`
- `opencv-python`
- `psbody_mesh`
- FLAME/BFM assets

Treat this subtree as an external toolchain included for convenience.

## 7. Important Entry Points by Use Case

If you are new to the repo, these are the most useful starting scripts.

### 7.1 If you want to inspect mesh quality

Start with:

- `datasets/render_mesh_preview.py`
- `datasets/compare_operator_closure_effect.py`

### 7.2 If you want to generate REMESH variants

Start with:

- `datasets/remesh.py`
- `datasets/expand_remesh_topologies.py`

### 7.3 If you want to precompute DiffusionNet operators

Start with:

- `face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py`

### 7.4 If you want to understand WBES evaluation

Start with:

- `WBES/utils/WBES_helper.py`
- `WBES/code/WBES_pipeline.py`
- `WBES/code/WBES_pipeline_multi.py`
- `WBES/code/WBES_pipeline_landmarks.py`

### 7.5 If you want to understand the baseline embedding model

Start with:

- `face_embedding/gt_encdec/autoencoder/dataset_gtready.py`
- `face_embedding/gt_encdec/autoencoder/diffusion_autoencoder.py`
- `face_embedding/gt_encdec/autoencoder/train_autoencoder.py`

### 7.6 If you want to understand the current robustness branch

Start with:

- `face_embedding/gt_encdec/remeshing/intrinsic/robustness/paths.py`
- `face_embedding/gt_encdec/remeshing/intrinsic/robustness/model_helpers.py`
- `face_embedding/gt_encdec/remeshing/intrinsic/robustness/train_runner.py`
- `face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/config.json`

### 7.7 If you want to reproduce the FaceBench paper comparison

Start with:

- `faceBench/latentVSpipeline/README.md`
- `faceBench/latentVSpipeline/run_facebench_remesh.py`
- `faceBench/latentVSpipeline/run_facebench_remesh_perturbed.py`
- `faceBench/latentVSpipeline/summarize_existing_rankings.py`
- `faceBench/latentVSpipeline/compare_existing_methods.py`

Useful queue/run wrappers:

- `scripts/run_mixed_xtopo_registered_chamfer_eval.sh`
- `scripts/run_mixed_xtopo_registered_chamfer_rigid_only_eval.sh`
- `scripts/run_mixed_xtopo_nicp_correspondence_eval.sh`
- `scripts/run_mixed_xtopo_chamfer_topology_breakdown_scenarios_eval.sh`

### 7.8 If you want to reproduce the FaceVerse validation path

Start with:

- `datasets/FaceVerse/downsample_faceverse.py`
- `datasets/FaceVerse/remesh_faceverse_from_npz.py`
- `datasets/FaceVerse/assemble_faceverse_cross_topology_dataset.py`
- `datasets/FaceVerse/compare_model_vs_chamfer_rankings_faceverse.py`
- `datasets/FaceVerse/compare_model_vs_chamfer_rankings_faceverse_sigma_sweep.py`
- `datasets/FaceVerse/FINE_tuning/prepare_faceverse_finetune.py`

Useful queue/run wrappers:

- `scripts/run_mixed_xtopo_faceverse_eval.sh`
- `scripts/run_mixed_xtopo_faceverse_postperturb_icp_eval.sh`
- `scripts/prepare_faceverse_remesh10k_xtopo.sh`
- `scripts/run_mixed_xtopo_faceverse_remesh10k_postperturb_icp_eval.sh`

## 8. End-to-End Workflow Map

There is no single canonical script for the whole repo, but the workflows below represent the intended progression.

## 8.1 Ground-truth preparation workflow

1. Align raw synthetic meshes into GT-ready OBJ files
   - `face_embedding/gt_encdec/alignment/prepare_GT_ready.py`
2. Convert OBJ to NPZ
   - `datasets/obj_to_npz.py`
3. Crop to canonical facial region if needed
   - `datasets/crop.py`
4. Precompute DiffusionNet operators
   - `face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py`

## 8.2 REMESH topology workflow

1. Start from GT NPZ data
2. Generate topology variants
   - `datasets/remesh.py`
   - `datasets/expand_remesh_topologies.py`
3. Inspect mesh/topology quality
   - `datasets/render_mesh_preview.py`
   - `datasets/compare_operator_closure_effect.py`
4. Precompute operators for generated variants
   - `face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py`

## 8.3 WBES evaluation workflow

1. Prepare per-frame reconstructed meshes in `.txt` form by method
2. Optionally extract landmarks
   - `WBES/code/landmark_extractor.py`
3. Run:
   - `WBES/code/WBES_pipeline.py` for full mesh WBES
   - `WBES/code/WBES_pipeline_landmarks.py` for landmark WBES
4. Use plotting/correlation scripts for downstream analysis

## 8.4 Baseline embedding workflow

1. Prepare operator-enriched GT-ready NPZ data
2. Train baseline autoencoder
   - `face_embedding/gt_encdec/autoencoder/train_autoencoder.py`
3. Extract latents and distance matrices
   - scripts in `autoencoder/latent_analysis/`
4. Compare latent geometry to GT geometry / WBES-like separation

## 8.5 Robustness training workflow

1. Prepare operator-enriched REMESH variants
2. Prepare a GT distance matrix
3. Train robustness-aware intrinsic encoders
   - `face_embedding/gt_encdec/remeshing/intrinsic/robustness/train_runner.py`
4. Evaluate under perturbations and sigma sweeps
   - `eval_utils.py`
   - `posthoc_runner.py`
   - plotting/analysis scripts in the intrinsic branch
5. Select the reference model/checkpoint
   - current top branch: `face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1`
   - current selected checkpoint: `mixed_xtopo_rank0p5_id0p25_bs5_best/checkpoints/best_by_xtopo_mesh_clean.pth`

## 8.6 FaceBench latent-vs-geometry workflow

1. Use the top model checkpoint and config from `dn_mixed_topology_v1`
2. Run FaceBench geometry stages on REMESH pairs
   - `faceBench/latentVSpipeline/run_facebench_remesh.py`
   - `faceBench/latentVSpipeline/run_facebench_remesh_perturbed.py`
3. Collect existing ranking summaries
   - `faceBench/latentVSpipeline/summarize_existing_rankings.py`
4. Aggregate method/scenario/topology summaries
   - `faceBench/latentVSpipeline/compare_existing_methods.py`
5. Generate distance-compression diagnostics
   - `faceBench/latentVSpipeline/analyze_distance_compression.py`
   - `faceBench/latentVSpipeline/plot_distance_compression_png.py`

## 8.7 FaceVerse validation workflow

1. Downsample FaceVerse meshes to about 10k vertices
   - `datasets/FaceVerse/downsample_faceverse.py`
2. Precompute DiffusionNet operators
   - `face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py`
3. Evaluate within-topology FaceVerse ranking against GT distances
   - `datasets/FaceVerse/compare_model_vs_chamfer_rankings_faceverse.py`
   - `scripts/run_mixed_xtopo_faceverse_postperturb_icp_eval.sh`
4. Build remesh10k FaceVerse cross-topology data
   - `datasets/FaceVerse/remesh_faceverse_from_npz.py`
   - `datasets/FaceVerse/assemble_faceverse_cross_topology_dataset.py`
   - `scripts/prepare_faceverse_remesh10k_xtopo.sh`
5. Evaluate cross-topology post-perturb ICP ranking and sigma sweeps
   - `scripts/run_mixed_xtopo_faceverse_remesh10k_postperturb_icp_eval.sh`
6. Prepare few-shot fine-tuning splits
   - `datasets/FaceVerse/FINE_tuning/prepare_faceverse_finetune.py`
7. Train/evaluate held-out FaceVerse variants under `datasets/FaceVerse/FINE_tuning`

## 9. Dependency Surface

There is no single universal environment file for the whole repo. The current workspace does include partial environment/dependency entry points:

- `environment.twotower_robust.yml`
- `faceBench/requirements.txt`
- `faceBench/facebench/pyproject.toml`
- local virtual environments such as `.venv_twotower_robust_312/`

Dependency management is therefore semi-explicit for the recent robustness/FaceBench work, but still not packaged as one reproducible root environment for every branch.

From the code, the major dependencies are:

- `numpy`
- `torch`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `plotly`
- `trimesh`
- `open3d`
- `igl` / libigl Python bindings
- `diffusion_net`
- `sklearn`
- `tqdm`
- `psutil`
- `tensorboard`
- `opencv-python`
- `chumpy`
- `psbody_mesh`

### 9.1 External code assumed to exist

Several scripts assume these external resources exist outside the repo:

- local DiffusionNet source tree:
  - `/equilibrium/lpampaloni/diffusion-net/src`
- GT or study assets under user-specific locations such as:
  - `/Users/pampaj/...`
  - `/home/pampalonil/...`

### 9.2 Environment reality

The recent scripts and queue wrappers mostly assume:

- `.venv_twotower_robust_312/bin/python`
- `WBES_DIFFUSION_NET_SRC=/deck/datasets/WBES-FaceEmbedding/diffusion-net/src`

Older FaceVerse downsampling code also references a conda environment at:

- `/home/lpampaloni/miniconda3/envs/3d/bin/python`

Important practical note:

- some scripts need `numpy`, `plotly`, and `igl`
- some need `open3d`
- some need `torch` and DiffusionNet
- no single repo-managed environment guarantees all of that

In practice, this repo probably relies on multiple local environments rather than one reproducible environment spec.

## 10. Pathing and Portability Problems

This is one of the most important sections in the guide.

The repository contains many absolute paths and machine-specific assumptions.

Common examples:

- `/equilibrium/lpampaloni/...`
- `/Users/pampaj/...`
- `/home/pampalonil/...`

Consequences:

- many scripts are not portable without editing constants
- running on a different machine will often fail immediately
- reproducing experiments depends on reconstructing local folder structure

Typical path-sensitive areas:

- WBES GT comparison
- landmark index locations
- DiffusionNet source imports
- FaceVerse input directories
- raw synthetic mesh locations

If portability becomes a goal, this is the first area to refactor.

## 11. Git Hygiene and Artifact Layout

The `.gitignore` confirms that the repo is intended to keep most large data and experiment outputs out of version control.

Ignored categories include:

- `/datasets/`
- `/WBES/raw_data/`
- `/WBES/results/`
- `/WBES/results_landmarks/`
- many autoencoder result folders
- many remeshing experiment folders
- voxel test outputs

However, the current checkout contains those local ignored directories and files. That means:

- the workspace is useful for research,
- but it is not a clean source-only checkout.

This also means a newcomer should not infer "core code" simply by file count. A lot of files are outputs.

## 12. Which Parts Look Stable vs Experimental

## 12.1 Relatively stable

- `WBES/utils/WBES_helper.py`
- the WBES pipeline logic itself
- GT-to-NPZ conversion utilities
- mesh preview / topology diagnostics
- operator precomputation logic
- parts of `intrinsic/robustness/` that are already modularized

## 12.2 Research-prototype but still useful

- `train_autoencoder.py`
- many latent-analysis scripts
- remeshing experiments
- voxel experiments
- cross-topology model experiments

## 12.3 Clearly ad hoc or legacy

- scripts with empty `STUDY_ROOT`
- scripts with multiple commented-out alternatives
- scripts with placeholder output names like `howwwwwwwww`
- scripts that depend on absolute user paths outside the repo
- exploratory outputs committed only locally

## 13. Recommended Reading Order for a New Contributor

If someone needs to become productive in this codebase quickly, this is the best order.

1. `README.md`
2. `CODEBASE_GUIDE.md` (this file)
3. `datasets/expand_remesh_topologies.py`
4. `face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py`
5. `face_embedding/gt_encdec/autoencoder/dataset_gtready.py`
6. `face_embedding/gt_encdec/autoencoder/diffusion_autoencoder.py`
7. `face_embedding/gt_encdec/autoencoder/train_autoencoder.py`
8. `face_embedding/gt_encdec/remeshing/intrinsic/robustness/train_runner.py`
9. `face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/config.json`
10. `faceBench/latentVSpipeline/README.md`
11. `faceBench/latentVSpipeline/run_facebench_remesh.py`
12. `datasets/FaceVerse/compare_model_vs_chamfer_rankings_faceverse.py`
13. `datasets/FaceVerse/FINE_tuning/prepare_faceverse_finetune.py`
14. `WBES/utils/WBES_helper.py`
15. `WBES/code/WBES_pipeline.py`

This order moves from:

- conceptual overview
- to data and operator infrastructure
- to model code
- to robustness experiments
- to current top-model artifacts
- to FaceBench and FaceVerse evaluation
- to WBES identity-effect-size analysis

## 14. What Is Actually Core to the Research Question

If you strip away plots, outputs, and side experiments, the conceptual core of the repo is:

### 14.1 Geometry-aware learned representation

- encode 3D face meshes directly
- reconstruct or embed them
- analyze whether latent spaces preserve identity structure

This is where most of the architectural complexity and most of the reusable research code live.

### 14.2 Topology robustness

- see whether identity representations survive remeshing, cropping, decimation, and perturbation

### 14.3 Identity-aware evaluation

- compute within-subject versus between-subject distances
- express separability as effect size
- compare that signal with geometric error

Everything else is either:

- preparation to make those experiments possible,
- or analysis to interpret the results.

## 15. Practical Risks and Rough Edges

These are the most important operational issues in the current repo.

### 15.1 Reproducibility risk

No root environment spec means environment recreation is manual.

### 15.2 Portability risk

Absolute paths are everywhere.

### 15.3 Data-layout coupling

Many scripts assume specific local directories already exist and are populated.

### 15.4 Source-output mixing

Code and artifacts coexist in the same folders, which makes orientation slower.

### 15.5 Multiple generations of code

The repository contains:

- old pipeline scripts,
- newer modular branches,
- duplicated ideas implemented in more than one place.

That is normal for research code, but it must be acknowledged explicitly.

## 16. Suggested Cleanup Roadmap

This is not required to use the repo, but it would make the codebase much easier to maintain.

### 16.1 Highest-value cleanup

- add a root `environment.yml` or `requirements.txt`
- replace absolute paths with config objects or CLI args
- separate source code from generated artifacts more clearly
- create a small `configs/` area for dataset roots and output roots

### 16.2 Structural cleanup

- promote `intrinsic/robustness/` style modularization to other branches
- define one canonical dataset schema for NPZ keys
- centralize subject parsing and variant naming rules

### 16.3 Documentation cleanup

- add per-subtree READMEs for:
  - `datasets/`
  - `WBES/`
  - `face_embedding/gt_encdec/autoencoder/`
  - `face_embedding/gt_encdec/remeshing/intrinsic/robustness/`
  - `face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/`
  - `datasets/FaceVerse/`

## 17. Bottom Line

This repository is best understood as the official NeurIPS 2026 submission workspace for learning identity-preserving 3D face embeddings under topology change.

The current top model is:

- `face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1`

The current paper-facing evaluation surface is:

- REMESH robustness and perturbation ranking under `dn_mixed_topology_v1`
- FaceBench latent-vs-geometry comparison under `faceBench/latentVSpipeline`
- FaceVerse external validation and few-shot adaptation under `datasets/FaceVerse`
- WBES identity-effect-size analysis under `WBES`

The cleanest operational path through the repo today is:

1. understand the dataset formats,
2. use the preprocessing scripts in `datasets/`,
3. read the operator precompute and dataset adapter code,
4. read the autoencoder / encoder model code,
5. focus on `intrinsic/robustness/` and `newdata/dn_mixed_topology_v1` for the current top model,
6. use `faceBench/latentVSpipeline/` for paper-facing geometry comparisons,
7. use `datasets/FaceVerse/` for external validation and few-shot adaptation,
8. use `WBES/` to evaluate the identity story from the effect-size side.

If you treat the repo like a polished package, it will feel inconsistent.
If you treat it like an active research workspace with several generations of experiments, it becomes coherent.
