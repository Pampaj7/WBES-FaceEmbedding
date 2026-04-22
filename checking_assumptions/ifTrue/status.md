# ifTrue: Assume The Result Is Real

Hypothesis:

The current benchmark result is genuinely telling us that pairwise target-aware registration harms identity ranking on REMESH cross-topology, even when plain perturbations alone do not.

## Current Evidence Imported From Existing Runs

All numbers below refer to the same REMESH cross-topology `mesh_pair_level` evaluation protocol with 100 subjects and 148500 mesh pairs per scenario.

- Raw perturbed Chamfer:
  - clean `0.2373`
  - jitter `0.1422`
  - translation `0.2345`
  - rotation `0.2306`
  - mixed `0.1381`
- Rigid-only ICP + Chamfer:
  - clean `0.2280`
  - jitter `0.1308`
  - translation `0.2267`
  - rotation `0.2273`
  - mixed `0.1288`
- Rigid ICP + non-rigid CPD + Chamfer:
  - clean `0.0112`
  - jitter `0.0080`
  - translation `0.0103`
  - rotation `0.0103`
  - mixed `0.0119`
- nICP-correspondence metric:
  - clean `0.1614`
  - jitter `0.1108`
  - translation `0.1606`
  - rotation `0.1603`
  - mixed `0.1063`

Provisional reading:

- Raw perturbations hurt, but still preserve a useful ordering signal.
- Rigid ICP already degrades the ranking.
- Free non-rigid CPD almost destroys it.
- A correspondence-style non-rigid refinement also fails to recover the ranking.

## Why This Could Still Be Correct

- Perturbations alone are target-agnostic.
- Pairwise registration is target-aware and can turn the metric into a measure of alignability rather than identity distance.
- The benchmark is cross-topology, so geometry-only alignment has no reason to preserve semantic identity correspondences.
- At subject-level aggregation, nuisance perturbations can average out, while pair-dependent registration bias does not.

## New Evidence Added On 2026-04-14

### 1. Synthetic ICP Sanity

Output:

- `checking_assumptions/outputs/synthetic_icp_sanity/summary.md`

Result:

- On identical meshes with synthetic translation only, raw mean `0.001267` drops to rigid `0.000304`.
- On identical meshes with synthetic rotation only, raw mean `0.005746` drops to rigid `0.001648`.
- On identical meshes with a combined rigid perturbation, rigid improves in `4/6` cases.
- CPD is already suspicious here:
  - identity self-pairs go from raw `~0` to CPD `0.008181`
  - translation-only self-pairs go to CPD `0.008468`
  - rotation-only self-pairs go to CPD `0.009157`

Interpretation:

- The rigid ICP branch is not obviously broken.
- The non-rigid CPD branch looks methodologically dangerous even on trivial same-mesh tests.

### 2. Same-vs-Different Gap Probe On A Larger Controlled Subset

Output:

- `checking_assumptions/outputs/same_vs_diff_gap_probe_all12/README.md`

Setup:

- 12 subjects
- 72 samples
- 180 same-subject cross-topology pairs
- 300 different-subject cross-topology pairs
- scenarios: `clean`, `translation`

Key results on `clean`:

- latent:
  - gap `0.3468`
  - AUC `0.9977`
  - different-only Spearman vs GT `0.8483`
- raw Chamfer:
  - gap `0.00358`
  - AUC `0.7720`
  - different-only Spearman vs GT `0.2432`
- rigid ICP + Chamfer:
  - gap `0.00665`
  - AUC `0.7821`
  - different-only Spearman vs GT `0.2360`
- CPD:
  - gap `0.00039`
  - AUC `0.5409`
  - different-only Spearman vs GT `0.0921`

Key results on `translation`:

- raw Chamfer:
  - gap `0.00359`
  - AUC `0.7741`
  - different-only Spearman vs GT `0.2490`
- rigid ICP + Chamfer:
  - gap `0.00526`
  - AUC `0.7849`
  - different-only Spearman vs GT `0.2443`
- CPD:
  - gap `0.00002`
  - AUC `0.5286`
  - different-only Spearman vs GT `0.0753`

Interpretation:

- CPD collapse is strongly confirmed.
- Rigid ICP does not simply "break identity" in every possible sense:
  - it slightly improves coarse same-vs-different separation on this controlled subset
  - but it does not improve the fine-grained ranking among different-subject pairs
- This gives a more precise story:
  - rigid ICP may help binary separation a little
  - but still fail to preserve the GT ordering among different identities
  - CPD is much worse and nearly destroys both

### 3. Aggregation Note

The existing subject-level results can still remain close to clean under rigid perturbations, because averaging over 30 cross-topology mesh-pairs per subject-pair reduces nuisance variance. That does not contradict the observation that pairwise registration introduces a structured pair-dependent bias rather than plain zero-mean noise.

### 4. GT Matrix Sanity

Output:

- `checking_assumptions/outputs/gt_matrix_sanity/README.md`

Result:

- the GT matrix is structurally clean:
  - shape `4999 x 4999`
  - exactly symmetric
  - exact zero diagonal
  - `4999 / 4999` names parsed to subject ids
  - no duplicate parsed subject ids
- the overlap with the REMESH benchmark is also clean:
  - all `500` REMESH subjects are found in the GT matrix
  - each REMESH subject contributes exactly `6` topology variants
- on an `80`-subject subset, the GT matrix matches the raw `original`-topology per-vertex mean-L2 distances essentially perfectly:
  - raw original vs GT: Pearson `1.0000`, Spearman `1.0000`
- but it matches the per-mesh normalized version much less well:
  - normalized original vs GT: Pearson `0.8034`, Spearman `0.8146`

Interpretation:

- This strongly argues against the idea that the GT matrix is randomly wrong, corrupted, or mis-indexed.
- However, it raises a more subtle protocol concern:
  - the GT behaves like a matrix built from raw original-space vertex distances
  - the benchmark metrics are computed on meshes after per-mesh normalization in `GTReadyDatasetNPZ`
- So the live suspicion is no longer "bad matrix", but "possible GT / metric-space mismatch".

This does not rescue CPD:

- CPD still behaves badly on self-pairs and on the controlled same-vs-different probe.
- But it does mean we should be careful before over-interpreting small raw-vs-rigid gaps, because the benchmark target itself may not live in exactly the same geometric space as the observed metric.

### 5. Controlled Probe With A Matched Normalized-Space GT Surrogate

Outputs:

- `checking_assumptions/outputs/gt_surrogate_normalized_original/README.md`
- `checking_assumptions/outputs/same_vs_diff_gap_probe_surrogate_gt_all12/README.md`

Setup:

- built a new `500`-subject GT surrogate directly from REMESH `original` meshes
- applied the same per-mesh normalization used by `GTReadyDatasetNPZ`
- reran the same all-12 controlled probe as before, but now against this surrogate GT
- CPD was skipped on purpose here; the question was specifically raw vs rigid

Key result:

Using the original GT, clean different-only Spearman was:

- raw `0.2432`
- rigid `0.2360`

Using the matched surrogate GT, clean different-only Spearman becomes:

- raw `0.3680`
- rigid `0.3853`

For translation:

- original GT:
  - raw `0.2490`
  - rigid `0.2443`
- surrogate GT:
  - raw `0.3668`
  - rigid `0.3833`

Interpretation:

- This is the strongest evidence so far that the GT definition matters materially for the rigid-only story.
- On the controlled subset, once the GT is moved into the same normalized geometry space as the benchmark, rigid ICP slightly overtakes raw Chamfer instead of slightly trailing it.
- So the current "raw > rigid" conclusion is not robust enough to be stated as a universal truth yet.
- The CPD conclusion remains much safer than the rigid-only conclusion.

### 6. nICP Probe, Full Surrogate-GT Benchmark, And Example Visuals

Outputs:

- `checking_assumptions/outputs/same_vs_diff_gap_probe_all12_nicp/overall_metric_summary.csv`
- `checking_assumptions/outputs/same_vs_diff_gap_probe_surrogate_gt_all12_nicp/overall_metric_summary.csv`
- `checking_assumptions/outputs/full_method_comparison/README.md`
- `checking_assumptions/outputs/rigid_icp_example_visuals/clean_rigid_registered_chamfer_manifest.json`
- `checking_assumptions/outputs/rigid_icp_example_visuals/translation_rigid_registered_chamfer_manifest.json`

Controlled all-12 probe, original GT:

- clean different-only Spearman:
  - raw `0.2432`
  - rigid `0.2360`
  - nICP `0.1612`
- translation different-only Spearman:
  - raw `0.2490`
  - rigid `0.2443`
  - nICP `0.1690`

Controlled all-12 probe, matched surrogate GT:

- clean different-only Spearman:
  - raw `0.3680`
  - rigid `0.3853`
  - nICP `0.3204`
- translation different-only Spearman:
  - raw `0.3668`
  - rigid `0.3833`
  - nICP `0.3256`

Larger 100-subject mesh-pair benchmark with surrogate GT:

- clean:
  - raw `0.2852`
  - rigid `0.2749`
- translation:
  - raw `0.2797`
  - rigid `0.2743`

Topology-pair sensitivity under surrogate GT:

- raw-vs-rigid winner flips:
  - clean `4`
  - translation `2`

Visual examples:

- `clean` example 1:
  - pair `id0051_GTready_down8k -> id0456_GTready_up60k`
  - raw `0.0036`
  - rigid `0.0114`
- `clean` example 2:
  - pair `id0084_GTready_down8k -> id0478_GTready_up60k`
  - raw `0.0070`
  - rigid `0.0027`

Interpretation:

- nICP does not rescue the geometric metric:
  - against the original GT it is clearly worse than both raw and rigid
  - against the surrogate GT it improves, but still stays below both raw and rigid
- The GT mismatch is real enough to change some local conclusions:
  - the controlled subset flips slightly in favor of rigid under the surrogate GT
  - several topology-pair winners flip under the larger surrogate-GT rerun
- But the larger surrogate-GT benchmark does not fully overturn the original story:
  - globally, raw still remains slightly above rigid on the 100-subject mesh-pair benchmark
- The visual examples support the more precise mechanism:
  - rigid ICP is not applying a constant correction
- some pairs get expanded, others get over-compressed
- that pair-dependent behavior is exactly what can scramble ranking without making all distances trivially equal

### 7. New Controlled Perturbation And Transform Analyses

Outputs:

- `checking_assumptions/outputs/perturbation_stability_rigid_scenarios/README.md`
- `checking_assumptions/outputs/icp_transform_dispersion_rigid_subset/README.md`
- `checking_assumptions/outputs/same_vs_diff_gap_probe_rigid_scenarios/README.md`

Perturbation stability on the controlled 12-subject subset:

- translation:
  - `|Ec - Ep|` mean `2.31e-04`
  - Pearson `(Ec, Ep)` `0.9976`
  - Spearman `(Ec, Ep)` `0.9967`
- rotation:
  - `|Ec - Ep|` mean `5.03e-05`
  - Pearson `(Ec, Ep)` `0.9999`
  - Spearman `(Ec, Ep)` `0.9995`
- mixed:
  - `|Ec - Ep|` mean `2.30e-03`
  - Pearson `(Ec, Ep)` `0.9884`
  - Spearman `(Ec, Ep)` `0.9754`

Interpretation:

- rigid perturbations are mostly rank-preserving and act like small scenario-dependent offsets on this probe.
- mixed perturbations are more disruptive, but still not enough to explain the strong ICP-collapse story by themselves.

Rigid-transform dispersion on the same controlled subset:

- clean:
  - rotation mean `3.11°`
  - rotation std `3.89°`
  - translation norm mean `0.0310`
- translation:
  - rotation mean `2.86°`
  - rotation std `1.64°`
  - translation norm mean `0.0301`
- rotation:
  - rotation mean `3.41°`
  - rotation std `4.65°`
  - translation norm mean `0.0310`
- mixed:
  - rotation mean `4.16°`
  - rotation std `14.32°`
  - translation norm mean `0.0280`

Interpretation:

- pairwise ICP is not a single tiny correction.
- the transform statistics spread enough to support the concern that pairwise registration is a non-comparable, pair-dependent correction.
- the mean-ICP sanity check does not rescue the ranking in a way that would overturn the core conclusion.

## Tests To Confirm This Story

- `synthetic_icp_sanity`: if ICP behaves correctly on trivial synthetic rigid nuisance, then the bad benchmark result is less likely to be a coding bug and more likely to be a protocol effect.
- `same_vs_diff_gap_probe`: if rigid / CPD shrink the gap between same-subject and different-subject pairs, that would directly support the identity-collapse interpretation.

## What Looks Most Likely Right Now

- The strongest and safest conclusion is still about CPD:
  - the benchmark result against CPD is very likely real
  - CPD behaves badly on the full benchmark, on synthetic self-pairs, and on controlled probes
- The next safest conclusion is about nICP:
  - it does not rescue the ranking
  - on the controlled probes it remains below both raw and rigid under both GT choices
- The rigid-only story is still conditional, but better pinned down now:
  - against the original GT, rigid is slightly worse than raw on the full benchmark and on the controlled probe
  - against a matched normalized-space surrogate GT, rigid slightly beats raw on the controlled subset
  - however, on the larger 100-subject surrogate-GT benchmark, raw still remains slightly above rigid globally
- So the most likely refined reading is:
  - CPD is genuinely harmful here
  - nICP correspondence is also not a viable fix here
  - rigid-only is GT-sensitive and not settled in a universal sense
  - but the larger evidence still leans toward raw being at least as safe as rigid for ranking preservation
  - part of the original rigid-only deficit is likely amplified by GT / metric-space mismatch, but not fully created by it
- The new template-based rigid probe does not rescue the ranking either:
  - aligning every mesh to a single fixed `original` template keeps the same-vs-different gap, but the diff-only Spearman remains below raw on clean / translation / rotation
  - on `mixed` the template-based alignment is roughly tied with raw rather than clearly better
  - so the problem does not disappear just by removing pairwise registration; the canonical frame still has enough geometry bias to disturb the fine ordering
- A small template sweep shows the result is template-sensitive but not template-saved:
  - `down8k` is the least harmful of the tested template choices on average
  - `remesh` and `up60k` sit in the middle
  - `original` is the worst of the four in terms of preserving the raw-vs-GT ranking
  - even the best template remains below raw on average, so switching to a fixed canonical reference helps only partially

## Next Tests

- Probe ranking only inside the different-subject pool on a larger subset, to isolate the "fine ordering" failure mode.
- Sweep rigid ICP parameters (`icp_points`, `max_correspondence_distance`) on the controlled probe to see whether the residual raw-vs-rigid gap is stable.
- Compare template-based alignment against a more neutral canonical template, or a learned/landmark template, if we want to check whether the specific `original` template is part of the remaining bias.
- If we want to keep pushing the template idea, try a landmark-anchored or mean-face template, because the simple topology-based templates still look biased.
- Add explicit directionality checks (`A->B` vs `B->A`) on the same controlled pairs, because the visual examples suggest strongly pair-dependent behavior.
- If needed, render a few additional example overlays from the surrogate-GT winner-flip topology pairs to see whether the same geometric story persists there.

## Update Log

- 2026-04-14: Imported existing benchmark evidence and set up the first confirm/falsify diagnostics.
- 2026-04-14: Added synthetic sanity results and the larger same-vs-different subset probe. CPD looks strongly suspect; rigid-only remains mixed and needs finer analysis.
- 2026-04-14: Added GT matrix sanity checks. The GT does not look corrupted; the real concern is a raw-vs-normalized geometry mismatch between the GT definition and the benchmark metric space.
- 2026-04-14: Built a normalized-space GT surrogate and reran the all-12 controlled probe against it. On that matched surrogate, rigid slightly outperforms raw, so the rigid-only conclusion is now explicitly unresolved.
- 2026-04-16: Added nICP controlled probes, full raw-vs-rigid surrogate-GT comparisons, and rigid-ICP example visuals. nICP remains clearly below raw and rigid; the GT mismatch is real but does not fully overturn the larger raw-vs-rigid benchmark.
- 2026-04-20: Added rigid-perturbation stability analysis and rigid-transform dispersion / mean-ICP sanity checks. Rotation and translation behave like tiny rank-preserving offsets on the controlled subset, while mixed is more heterogeneous; rigid ICP transforms remain pair-dependent enough to support the non-comparability concern.
- 2026-04-20: Added a template-based rigid ICP probe. Using a fixed `original` template does not recover the raw ranking cleanly, which suggests that removing pairwise registration alone is not enough to eliminate the geometry-space bias.
- 2026-04-20: Swept multiple fixed templates (`original`, `remesh`, `up60k`, `down8k`). The choice matters, but none of the tested templates beats raw on average; `down8k` is just the least harmful option.
