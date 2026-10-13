# Interazione tardiva sulle feature per vertice: regole (scritte PRIMA dei numeri)

Domanda: il dettaglio che il nostro encoder perde contro NoW metrical (GT FR) e REALY (GT SR) si recupera confrontando
le feature PER VERTICE (interazione tardiva, stile ColBERT) invece del solo vettore dopo il pooling? Nessun
riaddestramento. Codice `aau/late_interaction/` (`li_common.py`, `li_tokens.py`, `li_dist.py`, `li_eval.py`,
`li_cost.py`, `li.sbatch`).

Questo file e il codice sono nel commit delle regole, prima di qualunque numero con la GT. Prima di scriverle ho fatto
solo due collaudi tecnici su 20-40 mesh HELD-OUT dei generatori di training (nessuna GT letta, nessun dominio di test;
A10, job 1072557 e 1072947): tempi e convergenza, che hanno fissato `N_SUB` e il solutore di Sinkhorn (sez. 3).

## 1. Perimetro

- Solo domini di sviluppo: **HIFI3D** e **FaceScape dev**, senza crop (le 5 topologie `original`, `remesh`, `down8k`,
  `up60k`, `noisy`). Niente Ava-256: nessun file di Ava-256 letto, i modelli congelati per Ava solo letti (sha256
  controllato prima della forward).
- **FaceVerse: fuori da questo giro.** Il calcolo sarebbe economico, ma la vista con espressioni usa operatori e mesh
  con le facce invertite (`_flip_embed`) e un pool del pilota da mettere in scena a parte; con le L40S piene la
  rimando. Se si aggiunge, sara' solo descrittiva, con la variante gia' dichiarata e senza cambiare la scelta.
- Modelli: factorized **s1234** e **s2345**, passo 21.096 (epoca 72, pesi EMA), i checkpoint e gli sha256 della tabella
  della sez. 7.1 del protocollo di Ava-256 (`li_common.SEEDS`).
- GPU di produzione: L40S (`--gres=gpu:l40s:1`). I due collaudi su A10 non producono numeri usati.

## 2. Token: dove si leggono, pesi, normalizzazione

- **Feature per vertice** f_i (256 dimensioni) = l'uscita di `vertex_bottleneck`, prima del pooling:
  `v3_work/trainer/model_v3.py:92-93` (`EncoderV3.forward`: `Z = self.encoder(...)`, `Z = self.vertex_bottleneck(Z)`).
  Il pooling e' `model_v3.py:98` (`Z_global = self.pool(Z, mass, w)`), cioe' `FactorizedEncoderV3.pool`
  (`v3_work/trainer/factorized_v3.py:58-72`: media pesata con w e max, poi `pool_proj` -> u e `size_head` -> s).
- **Pesi** a_i = i pesi dello stesso pooling (`area_v3.pool_weights`, modo `smooth`, `model_v3.py:97`: aree della
  geometria passa-basso), normalizzati a somma 1. I vertici con peso 0 non sono token.
- **Lettura**: la stessa forward degli embedding della matrice (`eval_v3.install` + gli argomenti di `d3_embed.sbatch`,
  come `cc.py ours`), senza rumore, TF32 acceso nella forward come in sviluppo; un pre-hook legge l'ingresso X (frame
  globale) e un involucro di `model.pool` legge Z e w. Nulla cambia nel modello: [s, u] della stessa forward si
  confronta con gli embedding della matrice (`run.json`, `check_Z_vs_matrice`; A100 contro L40S, scarto atteso ~1e-3).
- **Normalizzazione: NESSUNA (feature grezze), costo c(f, g) = ||f - g||_2 (non al quadrato).** Motivo: (i) l'encoder
  e' stato addestrato con un pooling sulle feature grezze (media pesata per area e max, poi una proiezione lineare): la
  componente media di u e' lineare nei token grezzi e, con il costo euclideo, W1(token A, token B) >= || media pesata A
  - media pesata B || (Jensen), quindi l'interazione tardiva raffina il confronto del pooling nella sua stessa
  geometria; la normalizzazione L2 (coseno, ColBERT) introdurrebbe una geometria mai vista in training; (ii) la norma
  per vertice porta informazione che il pooling usa (il max per canale; con l'ingresso a scala globale anche la
  taglia), e la normalizzazione la butterebbe. Chamfer e OT usano lo stesso c (stesse unita').
- Calcolo delle distanze in float32 SENZA TF32 (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`, controllato), feature centrate
  sulla media dell'insieme (c invariato). Controlli: Chamfer contro float64 su CPU, Sinkhorn contro POT.

## 3. Varianti (3, fissate qui)

| variante | token | accoppiamento |
|---|---|---|
| `chamfer` | tutti i vertici | Chamfer simmetrico pesato: d = 1/2 [sum_i a_i min_j c(f_i, g_j) + sum_j b_j min_i c(f_i, g_j)] |
| `chamfer_sub` | sottocampione FPS | lo stesso Chamfer, masse = aree di Voronoi (isola l'effetto della risoluzione da quello dell'accoppiamento) |
| `sinkhorn` | sottocampione FPS | OT entropico, masse = aree di Voronoi, valore = <P, C> del piano entropico (senza il termine d'entropia) |

- **Sottocampione** deterministico, `N_SUB` = **1000** vertici per mesh: farthest point sampling su X (frame globale)
  dal vertice piu' vicino al baricentro pesato; massa di un campione = somma dei pesi a_i dei vertici piu' vicini a lui
  (Voronoi in X), a somma 1. 1000 e' l'estremo basso di "1-2k", scelto per il costo: nel collaudo sugli held-out
  (A10) Sinkhorn nel dominio dei logaritmi costava 73 ms per coppia a n = 1000 (circa 2,5 h per vista e seme).
- **Epsilon fisso** = 0,05 x s0, con s0 = mediana di c sui sottocampioni di 200 coppie a caso (`rng(0)`) dei due
  piloti, per seme (una scala delle feature, nessuna GT). Il valore di s0 e di epsilon e' scritto in `run.json` prima
  di ogni distanza.
- **Sinkhorn**: Sinkhorn-Knopp sul nucleo exp(-C / epsilon), a blocchi di 128 coppie su GPU (l'iterazione di
  `ot.sinkhorn`, metodo `sinkhorn`); arresto quando l'errore L1 sul marginale delle righe e' < 1e-5 (controllo ogni 10
  iterazioni) o a 3000 iterazioni (coppia tenuta e contata come non convergente); dove il nucleo va in underflow,
  ripiego nel dominio dei logaritmi (contato). Controllo contro `ot.sinkhorn(..., method="sinkhorn_log")` di POT
  0.9.5 in float64 su CPU (2 coppie per pilota e seme: POT in float64 costa minuti per coppia). Nel collaudo sugli
  held-out: scarto relativo da POT 2-6e-6, Chamfer contro float64 3e-6, 480 iterazioni mediane, 11,7 ms per coppia a
  blocchi su A10.
- Coppie: pilota e viste valutate = tutte le coppie di mesh di soggetti diversi (stessa topologia compresa); NaN sulle
  coppie dello stesso soggetto (non sono righe della matrice).

## 4. Letture e calibrazione (FR post hoc)

- `sr` = d_LI, con la GT SR.
- `fr` = **ricomposizione POST HOC** d = sqrt((S_i - S_j)^2 + S_i S_j (c' d_LI)^2), con la GT FR. S = exp(s) della testa
  di taglia dello stesso seme (stessa forward dei token). c' = mediana(d_P GT-SR) / mediana(d_LI) sugli held-out dei
  generatori di training: 12 soggetti per generatore (BFM, ICT, GNM; `rng(1234)` su `fact_calib.heldout_subjects`) x
  le 5 etichette, coppie dello stesso generatore con soggetti ed etichette diversi (`fact_calib.calib_one`), GT-SR di
  training `c3f/gt_sr.npz` x `dP_per_unit`. Un c' per seme e variante. Nessun dato dei domini di test entra in c'.
  Controllo: la stessa formula col vettore (d_P) su queste coppie contro il c ufficiale (`calibration.csv`).

## 5. Scelta della variante sul pilota (soggetti NON valutati)

- Pilota: i primi 12 soggetti NON valutati di `blmm.template_subjects` per vista (gli stessi del pilota di GW e
  percettivi in `competitors_new/gw_percettivi`) x le 5 topologie senza crop, tutte le coppie di soggetti diversi;
  GT FR e SR di quei soggetti (`gp_eval.pilot_gt`, che si ferma se un soggetto e' fra i valutati). Mai i 100 valutati.
- **Punteggio** = media sui due semi e sulle due viste (HIFI3D, FaceScape) di (rho(fr, GT FR) + rho(sr, GT SR)) / 2.
- **Regola**: vince il punteggio massimo; entro 0,01 dal migliore vince la variante con la mediana dei ms per coppia
  ISOLATA piu' bassa (100 coppie del pilota per vista e seme, GPU, `run.json`). La stessa variante e' dichiarata per
  entrambi i semi (`declared` in `fm_plugin.json`); le altre si calcolano e si riportano come non dichiarate.
- Righe di riferimento del pilota (fuori dalla regola): d_F cal. e d_P dei vettori della stessa forward.

## 6. Valutazione (100 soggetti valutati per vista)

- Righe, **maschera comune**, seme e **1000 repliche per soggetto** della matrice (`fm_eval`: righe di `rows_view`,
  maschera ricostruita con le righe di `fm_eval.run_view`, seme = `seme` di `full_matrix/spearman.csv`), insiemi senza
  crop: `nocrop` (principale), `con_perturbazione` per ognuna delle 5 topologie, `stessa_topologia senza_crop`.
- Uscita in formato plug-in: `factorized_<seme>/<variante>/` con `<vista>.npz` (`D_fr`, `D_sr`) e `fm_plugin.json`
  (`declared` = variante scelta sul pilota).
- **Delta appaiati** (interazione tardiva - colonna, replica per replica; positivo = interazione tardiva davanti):
  contro d_F cal. e d_F non cal. (GT FR) e d_P (GT SR) dello STESSO seme; contro NoW metrical e REALY (landmark
  esatti, letture dichiarate della matrice), ICP + Chamfer in mm e il miglior concorrente della matrice per (dominio,
  GT, insieme), cioe' la colonna `colonna_b` della riga `migliore_vs_migliore` di `full_matrix/deltas.csv`.
- Controllo: i rho delle colonne della matrice ricalcolati qui contro `full_matrix/spearman.csv` (punto, IC e righe:
  scarto atteso 0).
- **Lettura della domanda (fissata qui)**, variante dichiarata, insieme `nocrop`, per dominio:
  - *il pooling perde dettaglio che i token conservano*: delta contro il vettore dello stesso seme (d_F cal. con FR,
    d_P con SR) con IC sopra 0 in **entrambi** i semi;
  - *il dettaglio si recupera rispetto ai protocolli*: in piu', delta contro NoW metrical (FR) e contro REALY (SR)
    con IC che contiene lo 0 o sta sopra, in entrambi i semi;
  - *nessun recupero*: delta contro il vettore con IC che contiene lo 0 o sotto lo 0, in almeno un seme.
  La ricomposizione FR resta post hoc anche se la lettura e' positiva.

## 7. Costo per confronto

Stesso nodo L40S, stesso job (`li.sbatch`, passo `cost`): campione di `aau/costo_confronto/cc.py sample` (HIFI3D,
coppie original-down8k, original-up60k, down8k-up60k); ICP + Chamfer in mm per coppia (`cc.py geo`, CPU a un thread) e
confronto fra vettori (`cc.py cmp`, CPU a un thread, a blocchi di tutte le coppie di N mesh); interazione tardiva per
coppia ISOLATA su GPU (`li_cost.py`), mediana [p25, p75] fra le coppie della mediana di 3 giri (riscaldamento escluso).
Per mesh: forward coi token e sottocampione; memoria dei token per mesh contro il vettore.
