# Synthetic ICP Sanity

Question: can the current registration stack recover obvious rigid nuisance on identical meshes?

- Cases: `24`
- Subjects covered: `2`

## Summary By Transform

| transform | n | raw mean | rigid mean | cpd mean | rigid<raw frac | cpd<rigid frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 6 | 0.000000 | 0.000235 | 0.008181 | 0.000 | 0.000 |
| rigid_combo | 6 | 0.008173 | 0.006092 | 0.009534 | 0.667 | 0.333 |
| rotation_only | 6 | 0.005746 | 0.001648 | 0.009157 | 1.000 | 0.000 |
| translation_only | 6 | 0.001267 | 0.000304 | 0.008468 | 1.000 | 0.000 |

Interpretation guide:

- If `translation_only` and `rotation_only` do not improve strongly after rigid ICP, something is wrong in the registration pipeline or parameters.
- If rigid ICP succeeds here but still hurts the benchmark, that supports the idea that the benchmark failure is not a basic implementation bug.
