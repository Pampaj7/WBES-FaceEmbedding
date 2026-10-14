# Ava-256, prova confermativa: rapporto finale (passo 8 della sez. 7)

Protocollo v9 CONGELATO (sha256 `8690afbe6c8d7b6530cc928ca9a78a79416e82e0c133b213ce6ed64096a2c7d2`, commit `067f9c5`, tag `ava256-frozen`). Codice `8a6e184`, tutti i job dal worktree con `CONF_FROZEN_COMMIT`. Valutazione unica, 13-14 ottobre 2026: `conf/rules.csv` alle 00:12, `desc/spearman.csv` alle 01:45. Nessuna modifica al codice e nessuna correzione della §12.

Dati: 200 soggetti; 597.000 righe a topologie diverse (`nocrop_cross` 398.000, `crop` 199.000) e 119.400 di stessa topologia; 1000 repliche, seme 20261011.

## Verifiche (§11)

Superate in ogni job:
- manifest e impronte di viste, GT, scala e calibrazione;
- checkpoint dei due semi e di C3M; rilevatore; mappa dei landmark;
- HEAD = `8a6e184` con albero pulito; diffusion-net a `b1019b0`; ULRSSM.

Moduli confrontati col commit: nessuno scarto. Controllo bloccante del rilevatore: 0 pixel di scarto.

## Regole (+inf; Holm per seme, m = 9)

| Regola | Seme | Delta [IC 95%] | p | Holm i/soglia/rifiuta | Esito | Motivo |
|---|---|---|---|---|---|---|
| R1 | s1234 | +0.017 [-0.038, +0.069] | 0.257 | 7/0.0083/no | non confermata | IC comprende 0 |
| R1 | s2345 | +0.023 [-0.031, +0.079] | 0.197 | 8/0.0125/no | non confermata | IC comprende 0 |
| R2a | s1234 | +0.126 [+0.057, +0.194] | 0.001 | 6/0.0063/si | **confermata** | |
| R2a | s2345 | +0.133 [+0.065, +0.197] | <0.001 | 1/0.0028/si | **confermata** | |
| R2b | s1234 | +0.197 [+0.136, +0.260] | <0.001 | 1/0.0028/si | non risolta | NICP per coppia: righe non finite 9,06% > 5% |
| R2b | s2345 | +0.203 [+0.141, +0.269] | <0.001 | 2/0.0031/si | non risolta | idem |
| R3a | - | +0.119 [+0.061, +0.181] | <0.001 | 2 e 3/si | **confermata** | |
| R3b | - | +0.105 [+0.063, +0.145] | <0.001 | 3 e 4/si | non risolta | idem |
| R5a | s1234 | -0.028 [-0.076, +0.013] | 0.903 | 9/0.025/no | non confermata | delta < 0 |
| R5a | s2345 | -0.031 [-0.078, +0.011] | 0.916 | 9/0.025/no | non confermata | delta < 0 |
| R5b | s1234 | +0.159 [+0.101, +0.214] | <0.001 | 4/0.0042/si | non risolta | idem |
| R5b | s2345 | +0.157 [+0.098, +0.211] | <0.001 | 5/0.0050/si | non risolta | idem |
| R6 | s1234 | -0.000 [-0.024, +0.026] | 0.492 | 8/0.0125/no | non confermata | delta ~ 0 |
| R6 | s2345 | +0.020 [-0.004, +0.046] | 0.050 | 7/0.0083/no | non confermata | p sopra la soglia |
| R7 | - | +0.127 [+0.088, +0.172] | <0.001 | 5 e 6/si | **confermata** | |

- **NICP per coppia:** 45.071 / 45.067 righe su 597.000 senza distanza finita (mm / cs). I suoi delta restano riportati e nella famiglia di Holm col loro p.
- **Testa di R6:** su i256-a10-10, convergente in entrambi i semi (249 e 244 iterazioni), K0 superato.
- **R4 (descrittiva, fuori dalla famiglia):** +0.017 [-0.030, +0.066] / -0.005 [-0.052, +0.044].
- **Lettura mascherata (361.936 righe):** R2b +0.116 / +0.123, R3b +0.032 [-0.005, +0.073], R5b +0.063 / +0.060.
- **Spearman `nocrop`:** d_F cal. 0.641 / 0.647, ICP mm 0.624, NICP su template 0.515 (GT FR); d_P 0.663 / 0.660, ICP cs 0.691 (GT SR).

## Descrittivi (`nocrop`; delta = nostro s1234 meno concorrente)

- **NoW metrical contro d_F (§9.2):**
  - d_F cal. senza crop: -0.017 [-0.064, +0.034] e -0.023 [-0.070, +0.029];
  - col crop: +0.173 [+0.121, +0.224] e +0.195 [+0.143, +0.247];
  - d_F non calibrata: +0.042 / +0.043 senza crop, +0.093 / +0.112 col crop.
- **ULRSSM:**
  - per coppia (FR): 0.479, delta +0.162 [+0.111, +0.209];
  - via template (SR): 0.472, delta +0.191 [+0.138, +0.241];
  - letture non dichiarate: `pair_sr` 0.354, `tpl_fr` 0.576.
- **GW:** FR 0.414 (delta +0.227 [+0.166, +0.285]), SR 0.258.
- **Percettivi:**
  - LPIPS: FR 0.596 (delta +0.045 [-0.002, +0.093]), SR 0.487;
  - DINOv2: 0.240 / 0.256; CLIP: 0.199 / 0.225; DreamSim: 0.218 / 0.250 (FR / SR).
- **Fit vB (i256-a10-08):**
  - GNM: FR 0.583, SR 0.709, composizione 0.645;
  - FLAME 2023: FR 0.583, SR 0.688, composizione 0.643;
  - delta sulla composizione: -0.003 (GNM) e -0.002 (FLAME); d_P contro GNM SR -0.046 [-0.089, -0.005].
- **ArcFace (x180, i256-a10-09):** FR 0.331, SR 0.348; delta +0.310.
- **Spettrali (FR):** ShapeDNA 0.114, HKS 0.084, WKS 0.061.
- **Altri (FR):** varifold 0.501 (delta +0.140), taglia stimata 0.505, Stirling 0.398.
- **C3M e123:** d_F cal. 0.680, d_P 0.677.
- **REALY:** non applicabile; il controllo del frame dà una mediana di 4,74 mm contro la soglia di 2 mm (`protocolli/realy_na.json`), riportato senza ripieghi.
- **Ripieghi del detector:** 7 su 1.200, tutte `noisy` (0,7% del gruppo, sotto il 2%).

## Deviazioni e note di esecuzione

(a) Gli shard `run:k/8` dei protocolli (job 1075553-1075560) sono girati su a768-l40s-05/06, i256-a10-06 e a256-t4-04.
- Il protocollo e FROZEN.md vincolano ad a256-t4-04 solo rilevatore e render (job 1075525, su quel nodo).
- Il passo `run` calcola le distanze di NoW e Stirling da landmark già fissati.
- I valori dipendono dal tipo di CPU solo nell'aritmetica. In sviluppo gli scarti sono sub-mm: ≤1,7e-4 mm nel controllo dei port di NoW (`32c3839`), limite di `pr_noise.check` 1e-4 mm.
- Effetto sugli Spearman trascurabile; non si ricalcola.

(b) Il job 1075536 (`embed`, nv-ai-04) è FAILED con codice 127.
- Singularity non ha avviato il container ("unknown userid"), dopo le verifiche della §11 e senza scrivere uscite.
- Rilanciato identico come 1075603, completato.
- È un guasto d'infrastruttura, non un crash del codice congelato: la §12 non si applica.

(c) `aau/ava256/ava_common.py` compare nei record dei moduli (importato da `ava_freeze.py` con `freeze`) ma non nella tabella §11.1, perché il tracciamento di collaudo non passa da `freeze`. Il suo sha (`dfd6067b…`) coincide col blob a `8a6e184`.

Verifica indipendente (verifier, 14 ottobre): PASS su commit e moduli (0 scarti su 2.902 voci), nodi, completezza, Holm e coerenza interna.
