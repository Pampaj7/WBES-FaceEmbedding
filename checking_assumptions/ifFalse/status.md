# ifFalse: Assume The Result Is Wrong Or Misleading

Hypothesis:

The observed ranking degradation after ICP may be caused by an implementation issue, a bad protocol choice, a hidden asymmetry, or a mismatch between what the metric is asked to do and what the benchmark actually measures.

## Main Attack Surface

If the result is wrong, the weak points are likely to be one of these:

- ICP / CPD implementation bug
- Bad hyperparameters or bad correspondence distance threshold
- Source / target directionality artifact
- Wrong use of the non-rigid deformation in the final metric
- GT / aggregation mismatch
- Pair construction or topology-ordering artifact

## Tests Designed To Break The Current Conclusion

- `synthetic_icp_sanity`
  - identical mesh vs rigidly transformed copy
  - expectation if the code is sane: rigid ICP should recover near-zero registered distance
  - failure here would strongly suggest an implementation / parameter problem
- `same_vs_diff_gap_probe`
  - controlled subset with both same-subject and different-subject cross-topology pairs
  - if ICP improves separation instead of collapsing it, then the big benchmark result may be misleading

## What Would Count As A Serious Red Flag

- ICP fails to recover obvious synthetic translation / rotation on the same mesh
- Rigid-only already behaves nonsensically on trivial self-pairs
- Same-subject cross-topology pairs are not pulled closer by rigid ICP on a controlled subset
- The result flips under small and reasonable parameter changes

## New Evidence Added On 2026-04-14

### 1. Synthetic ICP Sanity Did Not Expose A Basic Rigid-ICP Failure

Output:

- `checking_assumptions/outputs/synthetic_icp_sanity/summary.md`

Result:

- translation-only self-pairs: rigid improves `0.001267 -> 0.000304`
- rotation-only self-pairs: rigid improves `0.005746 -> 0.001648`
- rigid-combo self-pairs: rigid improves in most cases

This weakens the simplest bug hypothesis:

- the rigid ICP implementation is not plainly failing on trivial rigid nuisance

### 2. The Non-Rigid CPD Branch Still Looks Suspicious

The same synthetic sanity run shows:

- identity self-pairs become worse after CPD: `~0 -> 0.008181`
- translation-only self-pairs become much worse after CPD than after rigid ICP
- rotation-only self-pairs also become much worse after CPD than after rigid ICP

So a strong falsification branch survives:

- even if rigid ICP is sane, the way CPD is used in the benchmark may still be wrong or methodologically unsuitable

### 3. Same-vs-Different Gap Probe Shows The Strongest Counter-Evidence Against An Over-Simple Story

Output:

- `checking_assumptions/outputs/same_vs_diff_gap_probe_all12/README.md`

On the larger all-12 subset:

- raw clean AUC on same-vs-different: `0.7720`
- rigid clean AUC: `0.7821`
- raw translation AUC: `0.7741`
- rigid translation AUC: `0.7849`

Also the same-vs-different gap increases under rigid ICP on this subset:

- clean:
  - raw gap `0.00358`
  - rigid gap `0.00665`
- translation:
  - raw gap `0.00359`
  - rigid gap `0.00526`

This means the statement

"applying ICP after perturbation simply breaks the identity structure"

is too crude.

A more precise falsification-friendly reading is:

- rigid ICP can help coarse same-vs-different separation
- but still fail to preserve the fine GT ranking among different identities
- CPD remains the branch with the clearest evidence of collapse

### 4. Important Nuance

The same-vs-different probe does not overturn the full benchmark:

- on the full benchmark, rigid-only still sits below raw in global Spearman
- on the controlled subset, rigid-only is slightly better on coarse binary separation but not on fine different-only ranking

So the live counter-hypothesis is now more specific:

- maybe the original conclusion is partly right for CPD
- but partly too broad for rigid ICP
- and the true failure mode is a mismatch between coarse discrimination and fine ranking preservation

### 5. GT Matrix Sanity Opens A Serious Protocol-Level Attack

Output:

- `checking_assumptions/outputs/gt_matrix_sanity/README.md`

What this test rules out:

- the GT matrix is not obviously corrupted
- it is not asymmetric
- it is not missing REMESH subject ids
- it is not scrambled by duplicate parsed ids

What it does suggest instead:

- on an `80`-subject subset, the GT aligns almost perfectly with raw `original`-topology per-vertex distances:
  - Pearson `1.0000`, Spearman `1.0000`
- but it aligns much less well with the per-mesh normalized version used by the benchmark loader:
  - Pearson `0.8034`, Spearman `0.8146`

This is a strong protocol-level concern:

- the matrix does not look "wrong" in a random sense
- but it may be "wrong for this benchmark space" if we intended the target to reflect the same normalized geometry that the model and Chamfer actually see

This matters because:

- `GTReadyDatasetNPZ` normalizes each mesh independently before evaluation
- the script that generated `normalized_matrix_distances.npz` claims to use the same normalization, but the current implementation loads raw vertices and computes distances directly

So one falsification branch is now very credible:

- part of the raw-vs-rigid story may be driven by a mismatch between the GT reference space and the metric space used at evaluation time
- this does not explain away CPD self-pair failures
- but it could absolutely affect how we interpret smaller differences involving raw Chamfer and rigid ICP

### 6. The Matched Surrogate-GT Probe Strengthens The Attack On The Rigid-Only Conclusion

Outputs:

- `checking_assumptions/outputs/gt_surrogate_normalized_original/README.md`
- `checking_assumptions/outputs/same_vs_diff_gap_probe_surrogate_gt_all12/README.md`

We built a new surrogate GT directly in the same normalized geometry space used by the benchmark loader, then reran the same controlled all-12 probe against that surrogate.

The result is important:

On `clean`, different-only Spearman vs GT changes from:

- original GT:
  - raw `0.2432`
  - rigid `0.2360`

to:

- surrogate GT:
  - raw `0.3680`
  - rigid `0.3853`

On `translation`, it changes from:

- original GT:
  - raw `0.2490`
  - rigid `0.2443`

to:

- surrogate GT:
  - raw `0.3668`
  - rigid `0.3833`

This is exactly the kind of flip we were looking for.

Interpretation:

- The matrix itself is not corrupted.
- But the rigid-only conclusion is not stable under a change to a GT that lives in the same normalized geometry space as the benchmark.
- Therefore, the current evidence no longer supports a strong claim that "rigid ICP harms ranking" in general.
- What it supports is a narrower claim:
  - CPD is bad
  - rigid-only may look bad against the original GT partly because that GT is defined in a different geometric space

### 7. The Larger Surrogate-GT Benchmark Partly Weakens That Attack

Outputs:

- `checking_assumptions/outputs/full_meshpair_surrogate_gt_raw_vs_rigid/raw_noicp_clean_translation/overall_scenario_summary.csv`
- `checking_assumptions/outputs/full_meshpair_surrogate_gt_raw_vs_rigid/rigid_only_clean_translation/overall_scenario_summary.csv`
- `checking_assumptions/outputs/full_method_comparison/README.md`
- `checking_assumptions/outputs/same_vs_diff_gap_probe_all12_nicp/overall_metric_summary.csv`
- `checking_assumptions/outputs/same_vs_diff_gap_probe_surrogate_gt_all12_nicp/overall_metric_summary.csv`

What survives from the falsification branch:

- On the controlled all-12 probe, moving to the matched surrogate GT still helps rigid relative to raw:
  - clean: raw `0.3680`, rigid `0.3853`
  - translation: raw `0.3668`, rigid `0.3833`
- Some topology-pair winners flip under the larger surrogate-GT benchmark:
  - clean `4`
  - translation `2`

What does **not** survive at scale:

- On the larger 100-subject mesh-pair benchmark with the surrogate GT, raw still remains slightly above rigid globally:
  - clean:
    - raw `0.2852`
    - rigid `0.2749`
  - translation:
    - raw `0.2797`
    - rigid `0.2743`

What the new nICP probes add:

- original GT, controlled all-12:
  - clean: raw `0.2432`, rigid `0.2360`, nICP `0.1612`
  - translation: raw `0.2490`, rigid `0.2443`, nICP `0.1690`
- surrogate GT, controlled all-12:
  - clean: raw `0.3680`, rigid `0.3853`, nICP `0.3204`
  - translation: raw `0.3668`, rigid `0.3833`, nICP `0.3256`

Interpretation:

- The GT mismatch attack is real:
  - it can flip local winner comparisons
  - it can flip the small controlled subset in favor of rigid
- But it is not a full explanation of the global benchmark:
  - the larger surrogate-GT benchmark does not overturn raw > rigid
- nICP also weakens the "maybe a better non-rigid ICP fixes it" counter-story:
  - even under the surrogate GT, nICP remains below both raw and rigid

### 8. The Visual Examples Argue Against A Simple “Constant Offset” Story

Outputs:

- `checking_assumptions/outputs/rigid_icp_example_visuals/clean_rigid_registered_chamfer_manifest.json`
- `checking_assumptions/outputs/rigid_icp_example_visuals/translation_rigid_registered_chamfer_manifest.json`

Examples from the controlled probe show two opposite behaviors:

- one pair gets much worse after rigid ICP:
  - raw `0.0036`
  - rigid `0.0114`
- another pair gets strongly over-compressed:
  - raw `0.0070`
  - rigid `0.0027`

This matters for falsification because:

- it argues against the idea that rigid ICP is just applying a nearly constant or monotone correction
- instead, the effect is visibly pair-dependent
- so the right attack surface is no longer "bad constant correction"
- it is "GT sensitivity plus pair-dependent alignment bias"

### 9. New Controlled Perturbation And Transform Analyses

Outputs:

- `checking_assumptions/outputs/perturbation_stability_rigid_scenarios/README.md`
- `checking_assumptions/outputs/icp_transform_dispersion_rigid_subset/README.md`
- `checking_assumptions/outputs/same_vs_diff_gap_probe_rigid_scenarios/README.md`

What the new perturbation stability analysis says:

- rigid translation and rotation are very close to rank-preserving offsets on the controlled subset
- `mixed` is more heterogeneous, but still mostly preserves the clean-vs-perturbed ordering
- this weakens the idea that the ICP result is just a trivial byproduct of the perturbations themselves

What the new transform-dispersion analysis says:

- rigid ICP produces a broad spread of rotation angles and translation norms on the same subset
- the spread is especially large under `mixed`
- the mean-ICP sanity check does not obviously rescue the ranking

Falsification interpretation:

- the original critique of rigid ICP as pair-dependent remains alive
- but the perturbation-only analysis says the benchmark perturbations themselves are not the main driver of the collapse
- so the current attack on the original story should focus on pairwise ICP and GT-space mismatch, not on rigid perturbations alone

## Best Current Attack On The Existing Story

- Attack the claim about rigid ICP, not the claim about CPD.
- The data now support "CPD is bad" much more strongly than "rigid ICP is bad."
- The matched surrogate-GT probe remains the clearest local falsification evidence for the rigid-only story.
- But the larger surrogate-GT benchmark weakens the stronger attack:
  - it suggests the GT mismatch affects the size and sometimes even the sign of the raw-vs-rigid gap
  - but it does not fully explain away the global raw > rigid result
- So the best current falsification-friendly reading is narrower:
  - the original benchmark likely overstates the rigid-only deficit
  - but it probably does not invent it from nothing

## Next Tests

- Compare raw vs rigid ranking on the same fixed different-subject subset while sweeping rigid ICP parameters.
- Check whether ordered source-target directionality changes the rigid result enough to explain part of the full-benchmark degradation.
- Test a template-based or landmark-anchored alignment on a small subset: if that behaves better than free pairwise ICP/CPD, the issue is likely methodological rather than purely geometric.
- Quantify how much of the raw-vs-rigid gap comes from a small number of high-impact topology pairs versus a broad diffuse effect.
- If the visual examples are convincing, render the surrogate-GT winner-flip pairs too and compare whether the same geometric mechanism still appears.

## Template-Based Check

- The fixed-template rigid probe has now been run on the controlled subset.
- It does not recover a clean improvement over raw Chamfer across the core scenarios, so the hope that "just align to one canonical face" would remove the bias is not fully supported.
- This weakens the strongest version of the falsification attempt: the issue is not only pairwise registration, but also the fact that a single canonical frame still changes the geometry space in a way that is not obviously GT-neutral.

## Template Sweep

- We also swept several fixed template choices.
- The best one among the tested topology-based templates is `down8k`, but it still stays below raw on average.
- `original` is the most harmful of the four tested templates, which suggests that template choice matters, but not enough to fix the ranking problem.
- The remaining gap points to a deeper protocol/geometry-space mismatch rather than to pairwise registration alone.

## Update Log

- 2026-04-14: Opened the falsification branch and queued the first sanity checks against implementation / protocol mistakes.
- 2026-04-14: Rigid ICP passed the synthetic sanity tests and slightly improved coarse same-vs-different separation on a larger controlled subset. The strongest suspicion now points at CPD and at fine-ranking failure rather than at a basic rigid-ICP bug.
- 2026-04-14: Added GT matrix sanity checks. The matrix itself looks structurally healthy, but there is now strong evidence for a raw-vs-normalized geometry mismatch that could contaminate the benchmark interpretation.
- 2026-04-14: Built a matched normalized-space surrogate GT and reran the all-12 controlled probe. The raw-vs-rigid ordering flips slightly in favor of rigid, which substantially strengthens the protocol-mismatch attack on the rigid-only conclusion.
- 2026-04-16: Added nICP probes, the full surrogate-GT method comparison, and rigid-ICP example visuals. The falsification attack now survives mainly as a GT-sensitivity argument, not as a full reversal of the larger raw-vs-rigid benchmark.
- 2026-04-20: Added a fixed-template rigid ICP probe. The template-based protocol does not restore the raw ranking cleanly, so pairwise ICP is not the only place where geometry-space bias can enter.
- 2026-04-20: Swept several fixed templates. `down8k` is the least bad of the tested topology-based templates, but none of them beats raw on average.
