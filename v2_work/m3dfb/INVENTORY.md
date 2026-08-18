# M3DFB error-estimator inventory

Source: <https://github.com/sariyanidi/M3DFB>, cloned at `external/M3DFB`
(`--depth 1`, not tracked by our git — `/external/` added to `.gitignore`).
Paper: Sariyanidi et al., *A Modular 3D Face Reconstruction Benchmark*, IEEE FG 2025.
Reviewed 2026-08-17 against our REMESH cross-topology pair benchmark.

## 0. What the repo actually is

M3DFB is **not** a bag of 16 estimator implementations. It is a 6-stage pipeline
framework with 11 concrete component classes; an "estimator" is a JSON recipe
naming one class per stage. The paper's 16 estimators are the cross product

```
rigid {ICP, RLR} x nonrigid {none, ELR, NICP, ELR+NICP} x corrector {none, ETC}
```

with `ChamferCorrespondence` + `DenseP2PDistance` fixed. The repo ships only
4 of the 16 as JSON (`info/error_computers/E01,E08,E12,E16.json`) plus
`known.json`, the synthetic-data oracle. `v2_work/m3dfb/m3dfb_adapter.py`
reconstructs all 16 from the same classes; the numbering was recovered from the
4 shipped recipes and verified against them (E1/E8/E12/E16 reproduce exactly).

The evaluation entry point (`run.py` -> `BaseReporter`) is unusable for us for
reasons unrelated to the estimators: it wants a fixed on-disk layout
(`DATA_DIR/<db>/Rmeshes/<mm>/<version>/<method>/id%04d.txt` plus
`Gmeshes/id%04d.txt` + `.lmks`), `.txt` vertex dumps only, one *ground-truth*
mesh per subject and a set of *reconstruction methods* to rank. Our benchmark is
subject-pair ranking, not method ranking. So the adapter bypasses the reporters
and drives `ErrorComputer.compute(R, G, Rlmks, Glmks)` directly — that class is
the whole estimator and is dependency-clean.

## 1. The blocking constraint: every estimator needs landmarks

`ErrorComputer.compute` takes `Rlmks`/`Glmks` (51 iBUG-51 points each) and
**both** rigid aligners consume them:

* `LandmarkBasedRigidAligner` — Procrustes on `opts['ref_lmk_indices']`
  (default `[13, 19, 28, 31, 37]`, i.e. 5 points: eye corners + nose tip).
* `ICPRigidAligner` — constructs a `LandmarkBasedRigidAligner` internally and
  runs it as a **mandatory pre-alignment** before calling open3d ICP
  (`rigid_aligners.py:154-160`). There is no landmark-free ICP path.

`Rlmks` is not annotated data: upstream takes it as `R[mm['lmk_indices']]`, i.e.
it requires the reconstruction to be in a **known template topology** (BFM or
FLAME) whose landmark vertex indices are hard-coded in `info/BFM-p23470.json`.
`Glmks` is read from a per-subject `.lmks` file shipped with the dataset.

**The repo contains no landmark predictor.** `grep -i "detect|predictor|dlib|
face_alignment|mediapipe"` over the tree returns nothing;
`facebenchmark/utils/save_landmarks_of_23470.py` is a 10-line script that slices
a hard-coded BFM index list. So on genuinely landmark-free, template-free
cross-topology pairs, **all 16 M3DFB estimators are inapplicable**, and no
amount of glue changes that. That is the honest headline.

### Why we can run them anyway (and the caveat that comes with it)

Our topologies are not template-free after all:

* `original` and `noisy` **are BFM p23470 in M3DFB's own vertex order.**
  Verified: Procrustes of our `id0000_GTready_original` onto
  `info/BFM-p23470.json`'s `mean_face_shape` under identity correspondence gives
  residual `d = 0.0015` (mean vertex error 0.023 IOD), versus `d = 0.9999` for a
  randomly permuted control. `noisy` gives `d = 0.0019`. So their 51 landmarks
  are **exact**, for free.
* `crop`, `down8k`, `up60k`, `remesh` have no template at all — vertex counts
  differ *per subject* (`down8k`: 8129 for id0000, 8136 for id0001), so there is
  not even a shared topology within a topology class. For these, the adapter
  transfers landmarks by nearest raw-coordinate vertex from the same subject's
  `original` mesh (all six variants live in one common world frame; `crop`
  vertices sit within 0.4 % of bbox extent of their `original` counterparts).

**This transfer is auxiliary information a real cross-topology evaluation would
not have.** It is not faked (no synthesized or predicted points — every landmark
is a real vertex of the mesh, located via a real correspondence to the subject's
own BFM mesh), but it is a privilege our benchmark grants M3DFB. Any number
below should be read as an *upper bound* on M3DFB estimator performance in the
landmark-free setting. Stated plainly in the paper, this is a point in our
favour, not against: our latent distance needs no landmarks, no template and no
alignment.

## 2. Per-component inventory

| Stage | Class | Alignment | Correspondence | Distance | Needs landmarks | Needs same topology | Needs template | Usable on our data |
|---|---|---|---|---|---|---|---|---|
| Mesh cropping | `PointBasedCropper` | – | – | – | yes | no | no | **NO — broken as shipped**: `__init__` asserts `opts['dist_threshold_ratio']` but `crop()` reads `opts['dist_threshold']` → `KeyError` on every call. Also needs dataset-specific landmark indices with no defaults. No shipped recipe uses it. |
| Rigid | `LandmarkBasedRigidAligner` (RLR) | similarity (Procrustes **with** scaling + best reflection) | – | – | yes (5 by default) | no | template, for `Rlmks` | yes, via §1 |
| Rigid | `ICPRigidAligner` (ICP) | similarity landmark Procrustes **then** open3d point-to-point ICP (`max_corr_dist=1000`, default 30 iters) | – | – | yes | no | template, for `Rlmks` | yes, via §1 |
| Non-rigid | `LandmarkBasedElasticAligner` (ELR) | non-rigid: per-landmark RBF-ish displacement field, weights from 3 cvxpy/SCS least-squares programs with an inf-norm constraint | – | – | yes (all 51) | no | template | runs, but O(N²): builds dense `squareform(pdist(R))`. **Measured 3.3 s / 1.1 GB at N=8k, 53 s / 6.5 GB at N=23k**; N=60k projects to ~350 s / ~43 GB. Not viable over 9k+ pairs on one core. |
| Non-rigid | `NonrigidICPAligner` (NICP) | non-rigid affine-per-vertex NICP, stiffness on a Delaunay graph of R's **xy projection** (so it needs no faces) | internal KD-tree NN | – | only if `prealign_ELR=1` | no | only if `prealign_ELR=1` | runs, **measured 185 s/pair at N=8k**. Not viable at scale. All 4 shipped JSONs set `prealign_ELR=1`, inheriting ELR's cost too. |
| Correspondence | `ChamferCorrespondence` | – | nearest-vertex, one-way R→G | – | no | no | no | yes (see patch 1 below) |
| Correspondence | `IdentityCorrespondence` | – | `pidx = arange(N)`, i.e. **dense pre-existing vertex correspondence** | – | no | **yes** | yes | **NO.** This is M3DFB's synthetic-data oracle (`known.json`, the "True" column). Two meshes of different topologies have no vertex-wise correspondence, so it cannot be evaluated on any of our pairs. |
| Distance | `DenseP2PDistance` (P2P) | – | – | point-to-point, per R-vertex | no | no | no | yes |
| Distance | `DenseP2TriDistance` (P2Tri) | – | – | point-to-"triangle" | no | no | no | **NO as shipped**: `compute()` calls `ptd.pointTriangleDistance` and the name `ptd` is imported nowhere in the repo → `NameError` on every call. Patched, it is still an O(N² log N) python loop, and its "triangle" is the 3 nearest *corresponding* points, not a real mesh face. |
| Distance | `LandmarksDistance` | – | – | landmark-to-landmark | yes | no | template | **NO by design here.** Our landmarks are exact-or-transferred (§1), so this would score our landmark transfer, not the meshes. |
| Correction | `TopologyConsistencyCorrector` (ETC) | – | – | – | yes | no | **yes** | partly. Needs `mm` = `mean_face_shape` + `lmk_indices`, and its per-vertex weight vector must have length `N_R`. So only when **R is BFM p23470** — 10 of our 30 ordered topology pairs (`original__to__*`, `noisy__to__*`). Marked non-applicable (empty cell) elsewhere. |

## 3. The 16 estimators

`ok` = runs over the full pair sets. `template` = same, but only where R is BFM
p23470. `slow` = runs correctly, cost forbids a 9k-pair sweep on one core.
`unusable` = cannot produce a number (see reason).

| Name | Recipe | Status | Note |
|---|---|---|---|
| **E1** | ICP + Chamfer + P2P | **ok** | the "widely used error estimator" of the README; our v1 ICP+Chamfer baseline |
| **E2** | ICP + Chamfer + P2P + ETC | **template** | R must be BFM p23470 |
| E3 | ICP + ELR + P2P | slow | ELR O(N²) |
| E4 | ICP + ELR + P2P + ETC | slow | ELR O(N²) + template |
| E5 | ICP + NICP + P2P | slow | 185 s/pair @ N=8k |
| E6 | ICP + NICP + P2P + ETC | slow | as E5 + template |
| E7 | ICP + ELR+NICP + P2P | slow | both |
| E8 | ICP + ELR+NICP + P2P + ETC | slow | shipped as `E08.json` |
| **E9** | RLR + Chamfer + P2P | **ok** | landmark-only rigid alignment (5 points) |
| **E10** | RLR + Chamfer + P2P + ETC | **template** | R must be BFM p23470 |
| E11 | RLR + ELR + P2P | slow | measured 53 s/pair @ N=23k |
| E12 | RLR + ELR + P2P + ETC | slow | shipped as `E12.json` |
| E13 | RLR + NICP + P2P | slow | |
| E14 | RLR + NICP + P2P + ETC | slow | |
| E15 | RLR + ELR+NICP + P2P | slow | |
| E16 | RLR + ELR+NICP + P2P + ETC | slow | shipped as `E16.json` |
| E1_p2tri | ICP + Chamfer + P2Tri | unusable | `ptd` NameError; O(N² log N) once patched |
| E9_p2tri | RLR + Chamfer + P2Tri | unusable | same |
| ("True") | RLR + **IdentityCorrespondence** + P2P | unusable | needs identical topology; `known.json` |

Summary: **4 of 16 usable at benchmark scale** (2 of them only on a third of the
topology pairs), 12 runnable-but-too-slow, and all 16 conditional on the
landmark privilege of §1.

## 4. Dependencies

`requirements.txt` = numpy, cvxpy, matplotlib, scikit-learn, open3d, trimesh
(plus scipy, undeclared but imported). All were **missing from `.conda_env`**
except numpy/scipy/sklearn. Installed with
`.conda_env/bin/pip install open3d trimesh cvxpy matplotlib`
→ open3d 0.19.0, trimesh 5.0.0, cvxpy 1.7.5, matplotlib 3.10.9 (+ pandas 2.3.3,
scs, osqp, clarabel, dash/flask/ipywidgets as open3d/cvxpy deps).
**No conflicts**: numpy stayed 2.2.6, scipy 1.15.3, torch 2.5.1+cu121, all still
import. No separate venv was needed. Note `cvxpy`, `matplotlib` and `trimesh`
are import-time requirements of `facebenchmark.performance_reporters.
error_computation`, so they must be present even for recipes that never use ELR.

## 5. Patches applied by the adapter (monkeypatch, clone left untouched)

1. `ChamferCorrespondence.establish` → `scipy.spatial.cKDTree(Y).query(X)`.
   Upstream is a python `for i in range(N)` doing a full N×M distance evaluation
   per vertex. The KD-tree returns the **identical** argmin (ties aside, which
   float coordinates do not produce) but takes ~20 ms instead of minutes for a
   20k-vs-20k pair. Without this nothing at benchmark scale is possible.
2. `distance_computers.ptd = distance_computers` — binds the undefined name so
   `DenseP2TriDistance` at least resolves. Kept for the record; P2Tri is still
   marked unusable on cost/semantics grounds.

Nothing else in `external/M3DFB` is modified; the clone is read-only.

## 6. Other conventions the adapter fixes

* **R vs G.** Estimators are asymmetric and the error is per R-vertex. We map
  `A → R` (reconstruction) and `B → G` (ground truth), so `crop__to__down8k`
  means R = crop mesh of subject A, G = down8k mesh of subject B.
* **Reduction.** `ErrorComputer.compute` returns a per-vertex array; upstream's
  `ResultsTable` reduces with a mean over a landmark-weight-thresholded vertex
  mask. That mask needs a template, so we use a plain mean over all R vertices
  (`reduce='median'` also available).
* **Scale.** Meshes are normalized per mesh exactly as the v1 benchmark did
  (centre on the vertex mean, divide by `max|coord|`, cf.
  `v2_work/phase0/normalization_confound.py`), so magnitudes are comparable to
  the stored `raw_chamfer` column. Rigid alignment includes scaling, so this
  mostly fixes the unit of the G side.
* **In-place mutation.** With ETC active, `ErrorComputer.compute` does
  `G[pidx] -= dG`, mutating the caller's array. The adapter always passes private
  copies of R, G and both landmark sets.
* **`ErrorComputer` reuse.** Cached per `(estimator, N_R)`; `ec.lmk_indices` is
  re-pointed per call. Single-threaded only.
* **Prior art in this repo.** `faceBench/latentVSpipeline/fg_metrics.py` is an
  earlier *hand-written re-implementation* of these ideas (`fg_p2p`, `fg_p2tri`,
  `fg_corrected_p2p`, own ICP, own `topology_consistency_corrector`) and is where
  the v1 paper's ICP+Chamfer / NICP numbers came from. The work here runs the
  **upstream code itself** instead, so the comparison is no longer a re-write of
  ours against theirs.

## 7. Measured on our benchmark (20 subjects, 190 pairs x 30 topology pairs = 5700)

`v2_work/m3dfb/m3dfb_summary.csv`, Spearman vs `gt_distance`, OVERALL:

| method | Spearman | n | v1 reference (100 subjects) |
|---|---|---|---|
| `latent_distance` (ours) | **+0.710** | 5700 | +0.751 |
| `raw_chamfer` | +0.265 | 5700 | +0.237 |
| **E9** RLR+Chamfer+P2P | **+0.456** | 5700 | not tried in v1 |
| **E10** RLR+Chamfer+P2P+ETC | +0.417 | 1900 | – |
| **E2** ICP+Chamfer+P2P+ETC | +0.290 | 1900 | – |
| **E1** ICP+Chamfer+P2P | +0.284 | 5700 | our ICP+Chamfer ≈ +0.29 |

Notes:

1. E1 reproduces our v1 ICP+Chamfer baseline (+0.284 vs ≈+0.29), so
   `fg_metrics.py` was a faithful re-implementation — but the comparison now runs
   the authors' own code.
2. RLR beats ICP by a wide margin (+0.456 vs +0.284). ICP-with-scaling optimises
   away exactly the identity differences we want to rank; a 5-landmark Procrustes
   preserves them. Consistent with M3DFB's own claim that the alignment stage
   dominates the estimator.
3. ETC barely moves the ranking here (E2−E1 = +0.006, E10−E9 = −0.039).
4. `crop` as the R side is the weak case for every estimator (E1 ≈ +0.22,
   `raw_chamfer` even negative): the surface is truncated on the side the error is
   measured per-vertex on.
5. Sanity check that the estimators behave like surface metrics: E9 on
   `crop__to__down8k` vs `crop__to__original` has Spearman 0.993 between the two
   columns and 3 % mean relative difference — the G-side topology is nearly
   irrelevant, as it should be.
6. Cost, measured (ms/pair, min–max across topology pairs): E1 180–1315,
   E2 590–1000, E9 15–93, E10 217–259. Scales with the R mesh size.
7. Subject count: 30 subjects (435 pairs/topology pair) projected to 3.3 h on one
   core, over budget, so the sweep ran at 20 subjects (~87 min). Also note that on
   this node background jobs do not advance while the agent is idle, so the sweep
   was driven as 12 foreground chunks via `--pair-labels`, relying on the
   per-topology-pair cache.
