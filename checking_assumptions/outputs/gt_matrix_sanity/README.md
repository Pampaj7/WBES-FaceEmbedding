# GT Matrix Sanity

## Structural checks
- matrix shape: 4999 x 4999
- diagonal abs max: 0
- symmetry abs max: 0
- parsed subject ids: 4999 / 4999
- duplicate parsed ids: 0

## Dataset overlap
- REMESH subjects: 500
- GT subjects: 4999
- overlap: 500
- meshes per REMESH subject: 6 to 6

## Original-topology subset comparison
- subset size: 80 subjects
- raw original vs GT: pearson 1.000000, spearman 1.000000
- per-mesh normalized original vs GT: pearson 0.803395, spearman 0.814585
- raw original vs per-mesh normalized original: pearson 0.803395, spearman 0.814585

Interpretation:
- If `raw original vs GT` is ~1.0 while `per-mesh normalized original vs GT` is noticeably lower, then the GT matrix is not arbitrary or randomly wrong.
- Instead, it is behaving like a matrix built from raw original-space vertex distances, while the benchmark metrics are evaluated on per-mesh normalized geometry.
