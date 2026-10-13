# Protocollo: validazione esterna su FaMoS, riconoscimento dell'identita' su scansioni grezze

Scritto il 13 ottobre 2026 e committato da solo PRIMA di calcolare qualunque distanza o metrica. Codice: commit
`20f94b4` (`aau/esterna_famos_id/ei.py` sha256 `a163b09e11adc8edae44b2db0fd3a744177b76ce8bce8c7b6598406458a1a098`,
`ei.sbatch` sha256 `0f52a899f238bffe787da94ac87664b31ce06ddfc4181e1e395a0a074c0cb42f`). Fino a questo commit sono stati
calcolati solo la selezione (`selection.json`) e il pilota della preparazione su scansioni NON valutate
(`prep_pilot.json`, controlli di qualita' del rilevatore e del ritaglio). Non e' stata calcolata nessuna distanza.

## 1. Scopo

La critica di circolarita' riguarda la GT FR/SR, costruita da registrazioni. Qui il compito ha solo l'etichetta
"stessa persona": riconoscere l'identita' su scansioni reali e grezze. Non si usano GT geometriche (FR, SR, unificata),
ne' registrazioni FLAME, ne' i loro landmark.

## 2. Soggetti

Sono i 15 soggetti di FaMoS TEST (`aau/famos/split.json`: FaMoS_subject_079 ... 093). E' l'unica parte di FaMoS con
scansioni grezze su disco, e nessuno e' escluso. Controlli fatti su manifest e split:

| uso | soggetti FaMoS | effetto qui |
|---|---|---|
| training del nostro (C3F factorized s1234 e s2345: `c3f/spec.json`, `c3f/split.json`) | nessuno: REMESH/BFM, ICT, GNM | - |
| calibrazione c della d_F cal. (`fact_calib.py`) | nessuno: held-out sintetici BFM, ICT, GNM | - |
| ULRSSM `faces_xyz` (training `ulr_data.py`, scelta su validazione `ulrssm/selection.json`) | nessuno: generatori BFM 2019, ICT, GNM | - |
| GW (configurazione `gw/pilot_choice.json`) | nessuno: pilota su soggetti non valutati di HIFI3D, FaceScape e FaceVerse | - |
| ICP, NoW | nessun parametro appreso | - |
| D3 (teste), template del NICP FaMoS (`fm_compute famos-template`) | le 80 di FaMoS TRAIN | nessuna scansione di TRAIN e' in questo test; metodi non usati qui |
| scelte della GT di E12 (maxabs, scala di S: "tutte le persone FaMoS") | TRAIN e TEST | solo GT, che qui non entra |
| valutazioni descrittive (matrice, `eval_famos_v3`, `famos_eval` e108) | TEST | numeri visti in sviluppo, ma mai usati per scegliere un modello (PLAN_MASSIVE 22.x, BOARD_DIARY: "gli unici mai usati in selezione sono FaMoS TEST e NoW") |
| L_d = 95,05 mm e CS_ref = 64,55 mm del dominio `famos` (`baselines_mm/params.json`) | TEST (mediane sulle 30 mesh di galleria di test_view) | costanti globali di unita', uguali per tutte le coppie; dichiarato |

I 15 soggetti non sono mai entrati in training, selezione o calibrazione di nessun modello o metodo usato qui.

## 3. Scansioni

- **Pool**: le scansioni rimaste su disco sono quelle di `famos_test_view.py` (851): per sequenza 2 o 3 fotogrammi
  (galleria, il piu' lontano e il piu' vicino al neutro). Furono scelte allora con `expr_mm`, una misura d'espressione
  della registrazione, non con distanze d'identita'. Gli altri grezzi sono stati cancellati per la quota disco
  (`aau/runs/evidence/famos/NOTES.md`).
- **Sequenze**: delle 28 sequenze, 26 hanno scansioni per tutti i 15 soggetti (mancano mouth_side per 087 e
  head_rotation_up_down per 091). Si prendono K = 10 sequenze, le stesse per tutti:
  `rng(20261013).choice(26, 10, senza reinserimento)`. Sono bareteeth, blow_cheeks, disgust, fear, happiness, kissing,
  mouth_down, rolling_lips, surprise e wrinkle_nose.
- **Fotogramma**: per ogni (soggetto, sequenza) uno fra quelli su disco, `rng([20261013, numero del soggetto,
  crc32(sequenza)])`. Quindi a volte e' l'espressione piu' marcata, a volte quella vicina al neutro.
- **Totale**: 15 x 10 = **150 scansioni** e 11.175 coppie, di cui 675 della stessa persona e 10.500 di persone diverse.
  Elenco completo in `selection.json`.
- **Pilota della preparazione**: 2 altre sequenze comuni (cheeks_in, lips_back; `rng([20261013, 1])`), 30 scansioni,
  mai valutate. Serve solo a controllare rilevatore e ritaglio; su di esse non si calcola nessuna distanza.

## 4. Preparazione (identica per tutti i metodi)

1. Si legge la scansione grezza (obj, mm, frame di acquisizione: +y in alto, volto verso +z; verificato sulle
   registrazioni di 7 sequenze per soggetto, frontale entro circa 25 gradi).
2. Render frontale con `pr_noise.render_mesh`, invariata (camera sull'asse +z, 640 px, FILL 0,45). Si rilevano i 68
   landmark iBUG con insightface `1k3d68` / `det_10g` (`pr_noise_detect.py --dir`, buffalo_l, i pesi della matrice) e si
   retroproiettano col raggio della camera. **Regola `front_only`**, solo per le grezze, fissata sul pilota: se il primo
   colpo cade su un triangolo visto dal retro, il raggio e' entrato da un buco (canti, narici) e ha colpito l'interno
   della nuca. Lo si tratta come raggio a vuoto e si usa il pixel piu' vicino con un colpo visto di fronte. Il verso
   del retro si fissa per scansione dalla maggioranza dei pixel colpiti. Nel pilota, prima della regola, 089/cheeks_in
   aveva en_l a z = -136 mm e scala 0,35; dopo, il residuo dei 7 landmark sul template e' <= 6,3 mm e la scala sta
   fra 0,87 e 1,08 (test_view con i landmark della registrazione: residuo <= 8,6 mm, scala 0,80-1,11).
3. Patch di NoW con i 7 landmark rilevati (iBUG 36, 39, 42, 45, 33, 48, 54): `famos_test_view.canonical_patch`
   importata (similarita' sul template T7 di NoW, ritaglio `compute_mask`, pulizia, `bfs_orient`, decimazione a 5215
   triangoli). Poi si torna in mm dividendo per la scala della similarita', come `blmm.to_mm` per FaMoS. Il ritaglio e'
   la regola geometrica fissa di NoW, uguale per tutti. Il frame resta canonico (+y in alto, +z naso, u = 1, R = I).
4. Landmark dei protocolli NoW: si rilevano di nuovo sulla patch con la stessa catena della `lmk_det` della matrice
   (render, insightface, retroproiezione invariata, senza `front_only`).
5. **Fallimenti**: se il rilevatore non trova volti, si usa il ripiego di `pr_noise_detect` (riquadro dei pixel colpiti),
   contato. Se la preparazione di una scansione solleva un errore, quella scansione esce per TUTTI i metodi, ed e'
   dichiarato in `prep_eval.json` e in results.md. Non si esclude nient'altro: nessun filtro su residui o punteggi.

## 5. Metodi e letture

Distanze su tutte le 11.175 coppie i < j, le stesse per tutti. Implementazioni della matrice importate, mai riscritte.

| colonna | metodo | implementazione |
|---|---|---|
| `factorized_s1234\|form`, `factorized_s2345\|form` | nostro, d_F non cal. | epoch072 EMA; `ws3b_latent.encode` dentro `eval_v3.py` (`WBES_V3_FACTORIZED_OUT=full`, tabella di scala `build_scale_table --view-dir --domain famos`: area in mm^2 della patch; operatori areanorm k_eig 128), poi `fact_paired.distances` |
| `...\|form_cal` | nostro, d_F cal. | c di `factorized_calibration.csv` (0,4053 e 0,4054) |
| `...\|shape` | nostro, d_P | idem |
| `icp\|mm`, `icp\|cs` | ICP + Chamfer in mm e cs | `blmm_pairs.run_items` -> `blmm.pair_metrics` (passo fast, `rigid_icp_chamfer`); X = indice minore, seme `blmm.mesh_seed(nome X, nome Y)` (la regola di FaMoS in blmm_pairs); L_d e CS_ref del dominio famos; CS_i = `blmm.mesh_scalars` della patch |
| `now_metrical\|mm`, `now_original\|mm` | NoW metrical e originale, landmark del rilevatore | `pr_run.one_pair` (ritaglio `pm.crop` raggio 1,4 coi landmark rilevati, `--now-opt` 0), media dei due versi |
| `ulrssm\|pair_fr`, `ulrssm\|pair_sr` | ULRSSM `faces_xyz` per coppia, mm e cs | `ulr_eval._prep`, `ulr_eval.on_gpu`, `ulr_common.pair_rms` (mappe nei due versi) |
| `gw\|fr`, `gw\|sr` | GW `euclid\|128\|cgw2` (la variante dichiarata dal pilota), mm e cs | `gp_gw.run_prep`, `gp_gw.solve_pairs` |

GPU: L40S (`--gres=gpu:l40s:1`) per il nostro e ULRSSM. Il resto gira su CPU, partizione prioritized. Il NICP per
coppia, REALY, Stirling, i percettivi e le baseline di taglia non sono stati richiesti e non si calcolano.

**Controlli di riproduzione** (tolleranze fissate qui, prima di eseguirli), sulle 15 scansioni di galleria di FaMoS TEST
della matrice:
- retroproiezione e render contro la condizione det di HIFI3D (prime 8 mesh): 1e-9 mm, 0 pixel;
- ICP mm e cs, 12 righe scan peak x 15 di galleria contro `baselines_mm/famos`: 1e-6 mm;
- NoW, 12 coppie coi landmark esatti contro `competitors_new/now_*/famos.npz`: 1e-4 mm;
- ULRSSM, 105 coppie contro `faces_xyz/famos.npz`: 1e-2 mm (rumore della GPU);
- GW, 12 coppie contro `gw/euclid_128_cgw2/famos.npz`: 1e-5 mm;
- latenti del nostro sulle 15 di galleria contro quelli della matrice (`datasets/FAMOS/eval/embeddings`): 5e-3 (L40S
  contro la classe di GPU di allora; TF32).

Se un controllo esce di tolleranza, i numeri si riportano lo stesso, accanto al controllo e alla discrepanza.

## 6. Metriche

- Distanza non finita (metodo fallito sulla coppia) = +inf, cioe' il caso peggiore per quel metodo. Il conteggio e'
  riportato per colonna.
- **Retrieval**: ogni scansione e' la query, la galleria sono le altre 149. **rank-1** = 1 se la piu' vicina e' della
  stessa persona; a pari merito sul minimo conta la quota della stessa persona fra i pari. **AP** = media delle
  precisioni ai ranghi delle 9 scansioni della stessa persona; ordine per distanza, a pari distanza per indice di
  galleria. Si riportano rank-1 e mAP medi sulle query.
- **Verifica**: coppie della stessa persona contro coppie di persone diverse, punteggio = -distanza. **AUC** pesata
  (`famos_eval.weighted_auc`: i pari valgono 1/2). **EER**: si accetta se d <= t; FAR e FRR si calcolano su tutte le
  soglie distinte (piu' t = -inf), interpolando linearmente fra i due punti operativi dove FAR - FRR cambia segno.
- **IC 95%** percentile con 1000 repliche per soggetto, `famos_eval.bootstrap_counts(15, 1000, 1234)` (le repliche di
  FaMoS TEST della matrice); la replica 0 e' il punto. Retrieval: query pesate con i conteggi del loro soggetto,
  galleria fissa. Verifica: peso c_s per le coppie della stessa persona, c_a c_b per le coppie di persone diverse; le
  coppie fra copie dello stesso soggetto ricampionato non esistono.
- **Delta appaiati**: ciascuna delle 6 letture nostre contro ciascuna delle 8 dei concorrenti, per le 4 metriche, sulle
  stesse repliche. Il verso e' "positivo = il nostro meglio": nostro - concorrente per rank-1, mAP e AUC; concorrente -
  nostro per l'EER. **Principali** (dichiarati ora): la d_F non calibrata, la lettura preregistrata della matrice, per i
  due semi, contro ogni concorrente, per rank-1, mAP ed EER. Gli altri delta (d_F cal., d_P, AUC) sono riportati
  completi in `deltas.csv` e in results.md.

## 7. Lettura

Lettura descrittiva: nessun test confermativo, nessuna soglia di successo, nessun p-value. I risultati sfavorevoli si
riportano come gli altri e nella stessa tabella. Dopo aver visto i numeri non si cambiano metodo, lettura, insieme
delle scansioni o metrica. Correzioni di bug dopo questo commit: si riportano in results.md con commit e motivo.
Limiti noti, gia' ora: 15 persone, quindi IC larghi; un solo sistema di acquisizione; espressioni comprese (scelte a
caso fra il pool); `L_d` e `CS_ref` sono costanti stimate su questi 15 soggetti.

## 8. Uscite

`aau/runs/evidence/esterna_famos_id/`: `PROTOCOLLO.md`, `selection.json`, `spec_*.json`, `prep_pilot.json`,
`prep_eval.json`, `controls/*.json`, `metrics.csv`, `deltas.csv`, `summary.json`, `results.md`. Distanze, patch,
render e latenti in `datasets/FAMOS/esterna_id/` (fuori da git, licenza MPI). Nessuna mesh o scansione in git.
