# FaceVerse neutra contro FaceVerse con espressioni, graduata

Generato da `v3_work/faceverse_neutral/fvn_summary.py`; protocollo `PROTOCOL.md` (sha256 in `PROTOCOL.sha256`). Soggetti [100] (gli stessi nelle due viste), righe per metodo [99000] (coppie di mesh senza crop, topologie diverse). IC 95% bootstrap per soggetto (1000 repliche, seme 1234, `eval_factorized.boot_rows`), lo stesso codice per le due viste. GT identiche nelle due viste (definite sull'identita' neutra): cambia solo la geometria osservata. FR e maxabs con d_F (||z|| per ctrlfr), SR con d_P.

Job: embed 1066573, scalari 1066574, fast mm 1066575, fast cs 1066576, template 1066577, nicp 1066578 1066579, riepilogo 1066580.

## Regola preregistrata (PROTOCOL.md, sezione 2)

- GT FR (primaria): **INTERMEDIO** (massimo neutro ICP + Chamfer in mm 0.409; minimo dei modelli ctrlfr s2345 0.299)
- GT SR (secondaria): **INTERMEDIO** (massimo neutro factorized2 s1234 0.407; minimo dei modelli ctrlfr s2345 0.305)

## Tabella

| metodo | FR neutra | FR espressioni | delta FR | SR neutra | SR espressioni | delta SR | maxabs neutra | maxabs espressioni |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| factorized s1234 | 0.337 [0.257, 0.409] | 0.297 [0.228, 0.360] | +0.040 | 0.333 [0.256, 0.402] | 0.286 [0.218, 0.353] | +0.047 | 0.384 [0.307, 0.458] | 0.328 [0.259, 0.396] |
| factorized s2345 | 0.373 [0.297, 0.444] | 0.317 [0.248, 0.383] | +0.056 | 0.372 [0.294, 0.442] | 0.313 [0.240, 0.381] | +0.059 | 0.388 [0.304, 0.476] | 0.321 [0.240, 0.401] |
| factorized2 s1234 | 0.396 [0.320, 0.464] | 0.337 [0.274, 0.392] | +0.059 | 0.407 [0.338, 0.477] | 0.337 [0.273, 0.392] | +0.069 | 0.437 [0.362, 0.505] | 0.373 [0.302, 0.437] |
| factorized2 s2345 | 0.375 [0.302, 0.446] | 0.340 [0.271, 0.404] | +0.035 | 0.374 [0.295, 0.444] | 0.332 [0.255, 0.397] | +0.041 | 0.382 [0.290, 0.462] | 0.340 [0.259, 0.418] |
| ctrlfr s1234 | 0.327 [0.257, 0.396] | 0.282 [0.218, 0.343] | +0.045 | 0.326 [0.247, 0.397] | 0.273 [0.198, 0.340] | +0.053 | 0.352 [0.262, 0.430] | 0.308 [0.231, 0.379] |
| ctrlfr s2345 | 0.299 [0.215, 0.378] | 0.258 [0.187, 0.322] | +0.041 | 0.305 [0.223, 0.384] | 0.259 [0.185, 0.328] | +0.046 | 0.369 [0.293, 0.442] | 0.313 [0.241, 0.379] |
| C3M e205 | 0.357 [0.269, 0.434] | 0.305 [0.232, 0.372] | +0.052 | 0.382 [0.302, 0.456] | 0.314 [0.244, 0.380] | +0.067 | 0.388 [0.304, 0.461] | 0.335 [0.256, 0.406] |
| e108 (cieco alla taglia) | 0.226 [0.157, 0.287] | 0.184 [0.127, 0.240] | +0.041 | 0.241 [0.174, 0.304] | 0.190 [0.131, 0.246] | +0.051 | 0.352 [0.278, 0.419] | 0.265 [0.198, 0.326] |
| ICP + Chamfer in mm | 0.409 [0.328, 0.485] | 0.337 [0.262, 0.405] | +0.072 | 0.383 [0.300, 0.463] | 0.309 [0.235, 0.380] | +0.073 | 0.447 [0.376, 0.512] | 0.364 [0.296, 0.425] |
| NICP per coppia in mm | 0.278 [0.196, 0.367] | 0.201 [0.126, 0.283] | +0.077 | 0.314 [0.233, 0.397] | 0.223 [0.148, 0.303] | +0.091 | 0.266 [0.183, 0.348] | 0.175 [0.098, 0.255] |
| NICP su template in mm | 0.223 [0.124, 0.320] | 0.212 [0.117, 0.303] | +0.012 | 0.156 [0.067, 0.254] | 0.152 [0.061, 0.246] | +0.004 | 0.250 [0.146, 0.343] | 0.245 [0.151, 0.327] |
| NICP per coppia, modo cs | 0.286 [0.198, 0.377] | 0.212 [0.129, 0.292] | +0.075 | 0.323 [0.239, 0.404] | 0.235 [0.156, 0.312] | +0.088 | 0.287 [0.198, 0.373] | 0.197 [0.117, 0.278] |
| taglia stimata (non oracolo) | 0.143 [0.034, 0.232] | 0.126 [0.026, 0.208] | +0.017 | 0.090 [-0.008, 0.173] | 0.080 [-0.007, 0.155] | +0.010 | 0.128 [0.020, 0.225] | 0.124 [0.030, 0.212] |
| taglia oracolo | 0.209 [0.114, 0.308] | 0.209 [0.114, 0.308] | +0.000 | 0.019 [-0.070, 0.110] | 0.019 [-0.070, 0.110] | +0.000 | 0.024 [-0.059, 0.110] | 0.024 [-0.059, 0.110] |

## Controlli

| controllo | sorgente | confronti | max abs diff |
| --- | --- | --- | --- |
| espressioni | factorized_results.csv / form_spearman.csv | 63 | 0.00e+00 |
| neutra | faceverse_neutral/form (eval_factorized.py) | 63 | 0.00e+00 |
| espressioni | baselines_mm/spearman.csv (punto) | 21 | 3.31e-07 |
| espressioni | baselines_mm/spearman.csv (IC, seme di E12 contro 1234) | 42 | 1.45e-02 |
