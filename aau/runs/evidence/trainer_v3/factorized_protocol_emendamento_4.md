# factorized_protocol.md, emendamento 4 (10 ottobre 2026, 11:15, PRIMA di qualunque calcolo di questo emendamento)

Protocollo ed emendamenti 1-3 invariati salvo quanto segue. Stato: i numeri non calibrati dei bracci factorized,
factorized2, ctrlfr (C3F) e di C3M sono gia' noti (`factorized_results.md` del 10 ottobre, verdetto BLOCCANTE del
critic); dual (s1234, s2345) in training. Nessuno dei calcoli qui sotto (calibrazione, colonne calibrate, analisi della
taglia, composizioni, delta su FaMoS, regola dual) e' stato eseguito prima del commit di questo file.

Motivi (difetti accertati dal critic): (1) d_P dei bracci fattorizzati non e' in unita' assolute (la stress di u
normalizza per la media, la rank loss allarga il latente): su HIFI3D la mediana di sqrt(S_i S_j) d_P e' 4.4 mm nella GT
e 8.6-15.7 mm nei modelli, quindi d_F sovrappesa la forma; (2) il vantaggio di ctrlfr su HIFI3D FR coincide con quello
della taglia oracolo; (5) le etichette di verdetto usavano NICP maxabs di E12 (0.367), soglia obsoleta; (6) mancavano
taglia oracolo, composizioni e FaMoS nei delta appaiati.

## 1. Calibrazione della scala di d_P

- Per ogni modello e checkpoint: factorized s1234 e s2345 (21.096 passi), factorized2 s1234 e s2345 (21.096 passi),
  factorized C3M e123 ed e205; dual s1234 e s2345 (21.096 passi), su u. UN solo scalare c per checkpoint, con
  c x d_P,modello che approssima d_P della GT.
- d_P,modello = ||u_i - u_j|| x dp_per_unit del checkpoint (`eval_factorized.dp_from_ckpt`: dP_per_unit / gt_scale;
  dual: quelli della GT shape). d_P,GT = GT-SR di training di E12 (`datasets/CANONICAL_GT/train/gt_sr_bfm_ict_gnm`,
  letta dal ritaglio identico `ablations/c3f/gt_sr.npz`) x dP_per_unit: distanza di Procrustes fra pre-forme a
  centroid size 1, la stessa definizione della GT-SR di valutazione.
- **Dati: SOLO held-out sintetici del training**, mai i domini di test. Held-out = `heldout` dello split (identico in
  `aau/data_scale/split_scale_all.json`, C3M, e `ablations/c3f/split.json`, C3F: si verifica), per dominio
  (`common.domain_of`): bfm (108), ict (992), gnm (100). 100 soggetti per dominio: sorted(rng(1234).choice(soggetti
  ordinati del dominio, 100, senza ripetizione)), per gnm tutti. Mesh: le etichette down8k, noisy, original, remesh,
  up60k presenti nelle sorgenti della spec di C3M (niente crop, niente espressioni), geometria GREZZA (come
  `build_scale_table.py`: REMESH -> npz_data_topo_500, ICT -> ICT/topo, GNM -> membro del tar). Stessa pipeline dei
  domini di test: operatori `areanorm_operators.py` k_eig 128, tabella di scala area_mm2 = u_d^2 area(grezza) (frame di
  E12), `eval_v3.py` + `zs_embed.py` con `WBES_V3_FACTORIZED_OUT=full`, scenario clean, seme 1234.
- Coppie: mesh di soggetti DIVERSI dello STESSO dominio e di etichette diverse (come `nocrop_cross`), i tre domini
  insieme.
- **Primaria:** c = mediana(d_P,GT) / mediana(d_P,modello) sulle coppie. **Sensibilita':** minimi quadrati senza
  intercetta, c_LS = sum(d_GT d_mod) / sum(d_mod^2), sulle stesse coppie. Si riportano anche c per dominio
  (descrittivo).
- d_F calibrata = sqrt((S_i - S_j)^2 + S_i S_j (c d_P)^2) (stessa formula, S = exp(s) in mm). Spearman di d_P da sola
  non dipende da c. Si riporta c per ogni modello; d_F calibrata con c primaria e' la distanza "form" di riferimento
  dei bracci fattorizzati per tutto cio' che segue; d_F grezza resta riportata. Per dual, z_F si usa grezza (contro
  ctrlfr); c di u si riporta e non entra in z_F.

## 2. Analisi della taglia

HIFI3D `nocrop_cross`, GT FR, STESSE righe, maschera comune e repliche di `fact_paired.py` (righe e seme di
`aau/baselines_mm/blmm_eval.py`, 1000 repliche per soggetto). o = |delta log S| della taglia ORACOLO (`oracle_size`,
S di FR).
- **(a) Spearman parziale dato l'oracolo:** in ogni replica (righe ripetute col peso della replica) ranghi medi del
  metodo, della GT e di o; i ranghi del metodo e della GT si regrediscono (minimi quadrati con intercetta) su
  [1, rango(o), rango(o)^2]; Pearson dei due residui.
- **(b) Spearman nel quintile basso di o:** righe con o <= q20, q20 = 20esimo percentile di o sulle righe della
  maschera (fisso, non ricalcolato per replica); Spearman in ogni replica sulle righe ripetute del sottoinsieme.
- **(c) Spearman grezzo** contro due composizioni con la formula di d_F, d_P = (ICP + Chamfer in modo cs) / CS_ref
  (`baselines_mm/params.json`: distanza media a centroid size 1):
  - "oracolo taglia + ICP cs" (tetto): S = centroid size di FR (`<set>_centroid_size.npz`);
  - "taglia stimata + ICP cs" (concorrente equo non oracolo): S = centroid size robusta della mesh osservata
    (`blmm.mesh_scalars`, la stessa di `est_cs`).
  d_P di ICP cs NON e' calibrata (non esiste su held-out sintetici; calibrarla sui test sarebbe oracolo): limite
  dichiarato, ICP cs e' una media punto-punto dopo ICP, la GT un RMS senza rotazione per coppia.
- Delta appaiati (metodo - riferimento) sulle stesse repliche per (a), (b) e Spearman grezzo, contro: ICP + Chamfer
  in mm, NICP su template in mm, NICP per coppia cs, ICP + Chamfer cs, taglia oracolo, taglia stimata, le due
  composizioni.
- **Regola "forma oltre la taglia"** per un braccio (ctrlfr z, factorized e factorized2 d_F calibrata, dual z_F), per
  seme: (a) batte ICP + Chamfer in mm E NICP per coppia cs, estremo inferiore dell'IC del delta di (a) > 0 per
  entrambi; E (c) non inferiore a "taglia stimata + ICP cs": estremo inferiore dell'IC del delta grezzo > -0.03.
  Verdetto "si'" se vale in entrambi i semi, "no" se fallisce in entrambi, "discordante" altrimenti. (b) descrittiva.
  C3M (un seme) descrittivo.
- Ripetuta come analisi secondaria su dev FaceScape (`nocrop_cross`) e FaMoS TEST (righe di `blmm_eval.famos`,
  blocco scan gallery -> scan, seme 1234, 15 soggetti; FaMoS senza NICP su template). Secondaria: niente regola.

## 3. FaMoS TEST nei delta appaiati

Righe di `blmm_eval.famos`, blocco scan gallery -> scan, GT FR e SR, repliche `famos_eval.bootstrap_counts(15, 1000,
1234)`. Distanze dei bracci dai latenti gia' calcolati da `eval_famos_v3.py` (cache `datasets/FAMOS/eval/embeddings/
latent_<tag>full_test_view.npz`, controllata contro `graded.csv` del braccio). La riga `chamfer_full` della tabella
FaMoS dei bracci e' la Chamfer di `famos_eval.py` sulla patch, NON `mm_chamfer` delle baseline: si rinomina senza
ambiguita'.

## 4. Etichette di verdetto

Si confrontano con le baseline in mm sugli stessi dati (delta appaiato su HIFI3D FR, per seme): "sopra" se
l'estremo inferiore dell'IC del delta > 0, "sotto" se l'estremo superiore < 0, "pari" altrimenti, contro ICP + Chamfer
in mm e contro NICP su template in mm. NICP maxabs di E12 (0.367) e la frazione g non si usano piu'.

## 5. Regola di decisione dual (preregistrata)

dual si adotta solo se, in ENTRAMBI i semi (dual contro il braccio dello stesso seme, delta appaiati sulle stesse
righe e repliche, margine 0.03 sull'estremo inferiore dell'IC 95%):
1. HIFI3D FR, z_F grezza contro ctrlfr: estremo inferiore di (z_F - ctrlfr) > -0.03;
2. HIFI3D SR, u contro d_P di factorized: estremo inferiore > -0.03; e il parziale (a) di z_F contro il parziale (a)
   di factorized con d_F calibrata: estremo inferiore > -0.03;
3. dev FaceScape (`nocrop_cross`): FR, z_F contro d_F calibrata di factorized, e SR, u contro d_P di factorized:
   estremo inferiore > -0.03 per entrambe.
Altrimenti, o se l'esito non e' risolto (righe mancanti, semi incompleti), si sceglie factorized con d_F calibrata.
Interpretazione dichiarata: il parziale (a) di dual e' quello della sua distanza form (z_F). Finche' le righe dual
mancano la sezione stampa "in attesa".

## Controllo di non regressione

d_F NON calibrata di factorized s1234 (21.096 passi) su HIFI3D FR deve ridare 0.6425 (graduata di `fact_summary.py`)
e 0.6423 (righe della maschera comune di `fact_paired.py`), come nei file del 10 ottobre.
