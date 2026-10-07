# Protocollo dichiarato prima dei numeri: quasi-duplicati fra test e training (punto 9 del rebuttal)

Dichiarato il 2026-10-07 22:05 CEST, prima di calcolare qualunque distanza test-training. Noto in anticipo:
la guardia della valanga (`aau/data_scale/PLAN.md`): soglia 0.0055 = meta' del vicino piu' prossimo minimo di
ICT-5000 in vertex-mean-L2 maxabs; nessuna identita' nuova sotto soglia, vicino minimo 0.0107.

**Domanda (Z1mX).** I soggetti di test sono identita' distinte da quelle di training, o ci sono quasi-copie?

**Split.** (1) NeurIPS: BFM, 400 training / 100 test (`bfm_only` di `aau/runs/ws2_cross3dmm/splits.json`,
seme 1234; il test e' il set held-out di WS1). (2) Congiunto `x3dmm_joint_bfm_ict_s1234_1019532`: `joint` dello
stesso file, held-out controllati contro `aau/data_scale/heldout_frozen.json` (BFM 108, ICT 992). Il vicino
si cerca solo dentro lo stesso 3DMM (template diversi).

**Metrica primaria:** vertex-mean-L2 fra le `original` normalizzate maxabs, la metrica della guardia, cosi' la
soglia 0.0055 vale nelle sue unita'. **Secondaria:** la GT di training del run (normalizzata al massimo), con
la soglia ricavata dalla stessa regola (meta' del vicino piu' prossimo minimo nel pool del 3DMM).

**Misure.** Per ogni soggetto di test il NN verso il training; come termine di paragone il NN training ->
training (lascia-uno-fuori, pool di taglia uguale), il NN test -> test e la distribuzione di tutte le distanze
test-test. Codice: `aau/near_dup/near_duplicates.py`, un job CPU.

**Lettura fissata ora.** "Nessun quasi-duplicato" se nessun test ha NN verso il training <= soglia, in
entrambe le metriche. Se la distribuzione NN test -> training e' allineata a quella training -> training
(mediane entro il 10%), il test e' "distinto come due identita' di training fra loro". Ogni test sotto soglia
e' elencato per nome con il suo vicino.

---

# Risultati (job 1060589, srun CPU, 2026-10-07)

| split | 3DMM | metrica | train / test | soglia (regola) | test sotto soglia | NN test->train: min / p5 / mediana | NN train->train (lascia-uno-fuori): min / p5 / mediana | NN test->test: min / mediana | distanze test-test: p1 / mediana | min NN test->train / soglia |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neurips | BFM | vertex_mean_l2_maxabs | 400 / 100 | 0.0055 (valanga 0.0055) | 0 | 0.0213 / 0.0225 / 0.0256 | 0.0195 / 0.0213 / 0.0261 | 0.0230 / 0.0273 | 0.0269 / 0.0454 | 3.9x |
| neurips | BFM | gt_training (normalized_matrix_distances.npz) | 400 / 100 | 0.0654 (meta' NN minimo del pool) | 0 | 0.1562 / 0.1632 / 0.1928 | 0.1308 / 0.1629 / 0.1963 | 0.1734 / 0.2078 | 0.2076 / 0.3355 | 2.4x |
| joint | BFM | vertex_mean_l2_maxabs | 392 / 108 | 0.0055 (valanga 0.0055) | 0 | 0.0201 / 0.0229 / 0.0259 | 0.0195 / 0.0218 / 0.0260 | 0.0213 / 0.0289 | 0.0290 / 0.0478 | 3.6x |
| joint | BFM | gt_training (gt_matrix.npz) | 392 / 108 | 0.0654 (meta' NN minimo del pool) | 0 | 0.1308 / 0.1681 / 0.1963 | 0.1442 / 0.1631 / 0.1944 | 0.1619 / 0.2203 | 0.2172 / 0.3560 | 2.0x |
| joint | ICT | vertex_mean_l2_maxabs | 4008 / 992 | 0.0055 (valanga 0.0055) | 0 | 0.0115 / 0.0137 / 0.0164 | 0.0111 / 0.0139 / 0.0166 | 0.0126 / 0.0178 | 0.0204 / 0.0400 | 2.1x |
| joint | ICT | gt_training (gt_matrix.npz) | 4008 / 992 | 0.0322 (meta' NN minimo del pool) | 0 | 0.0670 / 0.0799 / 0.0957 | 0.0644 / 0.0808 / 0.0963 | 0.0732 / 0.1035 | 0.1189 / 0.2330 | 2.1x |

Controlli:
- neurips / bfm: Spearman fra vertex-mean-L2 maxabs e GT di training su 124750 coppie = 0.783
- joint / bfm: Spearman fra vertex-mean-L2 maxabs e GT di training su 124750 coppie = 0.783
- joint / ict: Spearman fra vertex-mean-L2 maxabs e GT di training su 12497500 coppie = 1.000

# Giudizio (scritto dopo i numeri)

**Nessun quasi-duplicato** in nessuno dei due split, con nessuna delle due metriche: 0 soggetti di test su
100 (NeurIPS), 0 su 108 BFM e 0 su 992 ICT (congiunto) hanno il vicino di training sotto soglia.

- Metrica della valanga (vertex-mean-L2 maxabs, soglia 0.0055): il test piu' vicino al training sta a 0.0213
  (NeurIPS, 3.9 volte la soglia), 0.0201 (congiunto BFM, 3.6x), 0.0115 (congiunto ICT, 2.1x).
- Il test e' distinto dal training quanto due identita' di training lo sono fra loro: mediana NN test -> training
  0.0256 contro 0.0261 training -> training (NeurIPS BFM), 0.0259 contro 0.0260 (congiunto BFM), 0.0164 contro
  0.0166 (congiunto ICT); anche il minimo e il 5 percentile coincidono. La lettura fissata ("mediane entro il
  10%") e' soddisfatta in tutti e sei i casi.
- Con la GT di training (unita' diverse, soglia = meta' NN minimo del pool): stesso esito, minimo a 2.0-2.4 volte
  la soglia.
- Le distanze test-test stanno sopra il NN verso il training (mediana 0.045 contro 0.026 su BFM): e' atteso,
  sono distanze fra coppie qualsiasi, non fra vicini; il confronto giusto per il NN e' la riga training -> training.
- Nota: su BFM la vertex-mean-L2 maxabs e la GT del paper non sono la stessa misura (Spearman 0.783 sulle coppie);
  su ICT la GT di training e' la stessa misura (Spearman 1.000). Le conclusioni non cambiano con nessuna delle due.

Frase per il rebuttal: "No test identity is a near-duplicate of a training identity: the nearest training
neighbour of every test face is at least 3.6x (BFM) and 2.1x (ICT) the duplicate threshold, and test-to-train
nearest-neighbour distances match train-to-train ones (median 0.0256 vs 0.0261 on the NeurIPS BFM split)."
