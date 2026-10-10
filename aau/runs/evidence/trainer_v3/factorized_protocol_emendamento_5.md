# factorized_protocol.md, emendamento 5 (10 ottobre 2026, 14:30, PRIMA di qualunque calcolo di questo emendamento)

Protocollo ed emendamenti 1-4 invariati salvo quanto segue. Stato: i numeri dell'emendamento 4 sono noti
(`factorized_results.md` del commit f9c2b27: criterio "forma oltre la taglia" "no" per ctrlfr, factorized,
factorized2; dual in valutazione); verdetto del critic: RISERVE. Nessuno dei calcoli qui sotto (ICP e NICP per coppia
sugli held-out sintetici, k_ICP, k_NICP, composizioni calibrate, sezione esplorativa, c sul test) e' stato eseguito
prima del commit di questo file.

Motivo (difetto accertato dal critic): le composizioni dell'emendamento 4, sez. 2 (c) usano d_P = ICP cs / CS_ref
NON calibrata. Su HIFI3D (righe di `fact_paired.py`) la mediana e' 0.0378 contro 0.0805 della d_P GT: nella
composizione la forma pesa circa 2.1 volte meno che nella GT. E' lo stesso difetto corretto per il modello con c
(emendamento 4, sez. 1); il concorrente non oracolo del criterio era quindi sfavorito.

## 1. Calibrazione delle composizioni: k_ICP (primaria) e k_NICP

- **Dati e coppie: gli stessi di c** (emendamento 4, sez. 1): le 1500 mesh held-out sintetiche (100 soggetti per
  dominio bfm, ict, gnm; etichette down8k, noisy, original, remesh, up60k; geometria grezza) messe in scena da
  `fact_calib.py stage` (si verifica che i nomi coincidano con quelli di `datasets/V3_OPS_CACHE/heldout_calib/
  scale_table.npz`); coppie = mesh di soggetti diversi dello stesso dominio e di etichette diverse (297.000), d_P,GT =
  GT-SR di training x dP_per_unit. Mai i domini di test.
- **ICP + Chamfer in modo cs sugli held-out**: la pipeline di `aau/baselines_mm/blmm.pair_metrics` (modo `cs`, passo
  `fast`, 4096 punti coi semi della coppia, centro + ICP rigido punto-punto, p2p media). Coordinate come
  `blmm.work_coords` in modo cs: X = (V_mm - centro dei vertici) / L_d x CS_ref,d / CS_i, V_mm nel frame di E12 del
  dominio sintetico (`aau/runs/evidence/e12/frames.json`, bfm/ict/gnm), CS_i = centroid size robusta
  (`blmm.mesh_scalars`), L_d e CS_ref,d = mediana di maxabs e di cs sulle 100 original dei soggetti held-out del
  dominio (la definizione di `blmm_scalars.py`, applicata agli held-out). Coppia: X = la mesh di indice minore
  nell'ordine dei nomi, Y = l'altra, seme = indice della coppia nell'elenco ordinato. d_P,ICP = distanza x L_d /
  CS_ref,d, cioe' la distanza fra le due mesh a centroid size 1: la stessa grandezza di ICP cs / CS_ref sui test.
- **k_ICP = mediana(d_P,GT) / mediana(d_P,ICP)** su tutte le coppie (primaria); sensibilita' k_ICP,LS =
  sum(g m) / sum(m^2); k per dominio descrittivo. Coppie fallite (NaN) escluse e contate.
- **NICP per coppia in modo cs** (passo `nicp`, p2tri media, stesse coordinate): costa circa 4 s per coppia, quindi su
  un sottoinsieme fisso di 6000 coppie (`sorted(rng(1234).choice(297000, 6000, replace=False))` sugli indici
  dell'elenco ordinato). k_NICP con lo stesso metodo (mediane primaria, LS sensibilita') sul sottoinsieme; si riporta
  anche k_ICP sullo stesso sottoinsieme, per confronto.
- **Composizioni** (formula di d_F, S oracolo di FR o stimata `est_cs` come nell'emendamento 4):
  - "oracolo taglia + ICP cs cal.": d_P = k_ICP x ICP cs / CS_ref (tetto, oracolo);
  - "taglia stimata + ICP cs cal.": d_P = k_ICP x ICP cs / CS_ref (concorrente equo non oracolo, primario);
  - "taglia stimata + NICP cs cal.": d_P = k_NICP x NICP per coppia cs / CS_ref (seconda composizione non oracolo,
    DESCRITTIVA: non entra in alcuna regola).
  k_LS si riporta e non entra nelle composizioni. Le due composizioni non calibrate dell'emendamento 4 restano nei
  delta, rinominate "(em. 4, non cal.)", e non entrano in alcuna regola. Su FaMoS TEST le composizioni usano gli stessi
  k (FaMoS non ha NICP su template; NICP per coppia cs c'e').
- **Regola "forma oltre la taglia"** (emendamento 4, sez. 2): la condizione (c) diventa "estremo inferiore dell'IC del
  delta grezzo contro **taglia stimata + ICP cs cal.** > -0.03". Le condizioni su (a) e i verdetti per seme restano
  quelli. Se k_ICP non e' disponibile quando gira `fact_paired.py`, le composizioni calibrate mancano e il criterio e'
  "in attesa" (non si ricade su quella non calibrata).

## 2. Presentazione della tabella SR

Nella tabella "HIFI3D FR e SR" i concorrenti di SR sono NICP per coppia cs e ICP + Chamfer cs (invarianti alla
taglia, come SR): per GT SR le prime colonne sono queste due, poi quelle gia' presenti (ICP mm, NICP tpl mm, taglia,
composizioni). Solo presentazione: i delta esistono gia' nel CSV.

## 3. Analisi esplorativa, NON preregistrata (motivata dalla dipendenza di (a) da c)

Dopo i numeri dell'emendamento 4 il critic ha osservato che il parziale (a) della d_F composta cresce in modo
monotono con c (circa 0.17 a c = 0.2, 0.42 a 0.405, 0.565 a c = 1 per factorized s1234) e assorbe gli errori di
taglia S. Questa sezione e' **esplorativa, post hoc**: si riporta in una sezione separata dopo i risultati
preregistrati, che restano primi e invariati, e non cambia alcun verdetto.
- (i) **Parziale (a) della sola d_P**: HIFI3D, GT FR, righe e repliche di `fact_paired.py`; (a) di d_P del modello
  (factorized, factorized2 s1234 e s2345, C3M e123 ed e205: `shape`; dual: `u`) e delle baseline cs (ICP + Chamfer cs,
  NICP per coppia cs), con i delta appaiati contro ICP + Chamfer in mm, NICP per coppia cs, ICP + Chamfer cs.
- (ii) **Curva di sensibilita' a c**: HIFI3D, d_F(c) = sqrt((S_i - S_j)^2 + S_i S_j (c d_P)^2) per factorized s1234,
  factorized s2345, C3M e205, c in {0.2, 0.3, 0.405, 0.5, 0.75, 1}; per ogni c: (a), Spearman con FR, delta di (a)
  contro ICP mm e NICP cs, delta di Spearman contro "taglia stimata + ICP cs cal.", e la condizione del criterio
  calcolata a quel c (descrittiva). Uscita `factorized_explore.csv` e una tabella.
- (iii) **Errore di dominio della calibrazione** (informazione, non parametro): c_test = mediana(d_P,GT) /
  mediana(d_P,modello) sulle righe della maschera di HIFI3D e di dev FaceScape (d_P,GT = GT-SR di valutazione x
  dP_per_unit del suo json), per ogni modello; lo stesso per k_ICP e k_NICP (ICP cs / CS_ref, NICP cs / CS_ref).
  Confronto con c e k degli held-out. Non si usa per alcuna distanza.

## 4. Etichetta della regola dual

Se un esito della regola dual (emendamento 4, sez. 5) e' falso, l'esito stampato e' "no" (si sceglie factorized
calibrato) anche se altre righe mancano; "non risolto" solo se nessun esito e' falso e qualcuno manca. La decisione
non cambia.

## Controllo di non regressione

d_F grezza di factorized s1234 (21.096 passi) su HIFI3D FR: 0.6425 (graduata di `fact_summary.py`) e 0.6423 (maschera
comune di `fact_paired.py`); d_F calibrata (form_cal) di factorized s1234: 0.749 (come in f9c2b27).
