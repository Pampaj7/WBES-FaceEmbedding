# E1, emendamento 2: C3F-UGT con la GT unificata TARATA (9 ottobre 2026, prima di ogni numero delle celle nuove)

Si aggiunge a `protocol.md`, `protocol_amendment.md` e `nota_tecnica_gradvec.md`, che restano invariati.

**Motivo (critic).** I margini della loss sono in unita' della GT (`--rank_margin 0.05` ecc.), e la GT unificata
ha un'altra scala. Senza taratura, C3F-UGT contro C3F mescolerebbe il contenuto della GT con la sua scala.

**Stato al momento della scrittura:**
- C3F-UGT (1061850) non era ancora partito; e' stato tenuto fermo (`scontrol hold`) fino alla verifica.
- Nessuna cella nuova ha ancora un numero di valutazione.
- C2F, C3F e C2F-GNM sono in training dalle 19:15 dell'8 ottobre. Usano la GT maxabs e non cambiano.

**Taratura** (`aau/evidence/e1_factorial/e1_calib_ugt.py`, job 1062071):
- Per ogni dominio d, f_d = mediana della GT maxabs del run / mediana della GT unificata, sul blocco
  intra-dominio. Mediane esatte sulle coppie i < j, tutti i soggetti del dominio.
- Le coppie fra domini (che il trainer a batch monodominio non legge) sono scalate con sqrt(f_d1 f_d2).
- Fattori:

  | dominio | mediana maxabs | mediana unificata | f_d |
  | --- | --- | --- | --- |
  | BFM | 0.3452 | 0.2808 | 1.2294 |
  | ICT | 0.2311 | 0.2531 | 0.9134 |
  | GNM | 0.3001 | 0.2931 | 1.0241 |

- File: `datasets/UNIFIED_GT/train/gt_unified_bfm_ict_gnm_calib.npz` (+ `.json`, massimo 0.967).
- Controllo col loader del trainer (`check_calib.json`, ok):
  - names e indici di train, held-out ed eval online identici alla GT del run;
  - nessun valore non finito, diagonale 0, simmetria;
  - rapporto tarata / unificata = f entro 1e-7;
  - mediane per dominio di nuovo uguali alla maxabs.

**Celle:**
- **C3F-UGT** usa la GT TARATA. E' la cella della domanda dichiarata in `protocol_amendment.md`, sezione 4,
  che resta invariata (C3F-UGT - ICP+Chamfer, GT unificata di HIFI3D). La taratura cambia per dominio solo la
  scala, non l'ordine delle distanze: lo Spearman di valutazione non ne dipende.
- **C3F-UGT non tarata** (`c3fugtraw`, GT unificata originale): aggiunta in coda, a bassa priorita'.
  C3F-UGT - C3F-UGT non tarata e' l'effetto della scala; descrittivo, nessuna regola.
