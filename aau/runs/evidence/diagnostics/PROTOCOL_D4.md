# Diagnostica D4: ridiscretizzazione canonica al test (protocollo, scritto PRIMA dei numeri)

Scritto l'11 ottobre 2026, mattina, dal coder, su richiesta del PI, PRIMA di ogni embedding delle mesh ridiscretizzate e
di ogni Spearman di D4. L'impronta sha256 di questo file sta nel messaggio del commit che lo introduce e in
`PROTOCOL_D4.sha256`. E' una DIAGNOSTICA post hoc rispetto ai risultati gia' pubblicati (nasce da un'osservazione di
D3): si legge come tale, con le regole sotto fissate prima dei suoi numeri.

## 0. Cosa era gia' visto prima di scrivere

- **Il gap di discretizzazione (D3, emendamento 2 POST HOC, `results_d3.md`; `baselines_param/conclusions.md`,
  emendamento 3 sez. 6).** d_P contro SR sulle righe con la stessa etichetta ai due lati sta sopra le righe incrociate:
  remesh-remesh - incrociate, factorized s1234 / s2345, FaceScape +0.053 [+0.041, +0.068] / +0.050, HIFI3D +0.020 /
  +0.019, FaceVerse neutra +0.012 / +0.011, FLAME +0.026 / +0.030. Livelli incrociati (s1234): FaceScape 0.747, HIFI3D
  0.622, FaceVerse neutra 0.333.
- **Un precedente (E3b, 8 ottobre, `aau/runs/evidence/e3/summary_e3b.md`).** Rimesh uniforme al test con un altro
  remesher (clustering di Voronoi stile ACVD, lato 0.035 raggi RMS) e un altro modello (e108, operatori non normalizzati
  per area): su HIFI3D Spearman nocrop -0.174 [-0.224, -0.127] contro le mesh come sono (GT maxabs), crop e noisy
  peggiorati, down8k recuperato. Bracci, GT e operatori di D4 sono diversi: il precedente non decide l'esito, ma dice
  che una ridiscretizzazione al test puo' anche costare.
- **Pilota geometrico di questa mattina (nessuna GT, nessun embedding; job 1068277-1068280).**
  - Lato mediano delle mesh del training in mm (held-out della calibrazione, `datasets/V3_OPS_CACHE/heldout_calib`, un
    file su tre, 500 mesh, le 5 etichette di BFM, ICT, GNM; lato della mesh ad area 1 x sqrt(area_mm2) della tabella):
    mediana 2.04 mm (p10 1.18, p90 3.91); mediane per dominio BFM 1.38, GNM 2.42, ICT 3.49.
  - Viste di test (10 soggetti valutati per vista, lato x u_d di `e12/frames.json`): original HIFI3D 1.73 mm (9.5k
    vertici), FaceScape 1.27 (12.6k), FaceVerse 1.25 (28.6k); down8k 2.1-3.6 mm, up60k 0.8-1.2 mm.
  - pymeshlab col lato 2.0 mm e i parametri della sez. 1 su 36 mesh (2 soggetti x 6 topologie x 3 viste): una
    componente e nessun vertice non referenziato in 36 / 36, uno spigolo non-manifold in 1 / 36 (FaceScape up60k),
    operatori areanorm k 128 calcolati in 36 / 36; 10-14k vertici sulle mesh pulite, 18-41k sulle noisy (il rumore
    gonfia l'area); lato mediano 2.03-2.08 mm (noisy 1.76-1.92); area dopo / prima 0.98-1.00; 1.6-9 s per mesh pulita e
    8-21 s per noisy, un thread.

## 1. Remesher canonico

- **Filtro**: `meshing_isotropic_explicit_remeshing` di pymeshlab 2025.7.post1 (remeshing isotropico esplicito di
  VCG / MeshLab, alla Botsch-Kobbelt: split, collapse, flip, rilassamento tangenziale, riproiezione sulla superficie
  d'ingresso). Venv `.venv_d4` (fuori da git, `aau/diagnostics/d4.sbatch` passo setup), lanciato con
  `aau/diagnostics/d4_run.sh`.
- **Non e' usato per costruire le viste del benchmark** ne' quelle del training: le topologie di test e di training
  vengono tutte da decimazione quadrica (`igl.qslim`, `v2_work/genict/mesh_ops.decimate_to`: remesh = 2 passi di
  smoothing umbrella + 0.7x quadrica, down8k = quadrica, up60k = suddivisione a punto medio + quadrica), rumore e banda
  di bordo (`make_crop`). Per questo non si usa "suddivisione + decimazione quadrica a un numero fisso di vertici": e'
  la ricetta di up60k e down8k.
- **Parametri** (scelti su criteri geometrici, sez. 0, prima dei numeri):
  - `targetlen` = **2.0 mm** in valore assoluto, convertito nelle unita' native della vista con u_d di
    `aau/runs/evidence/e12/frames.json` (HIFI3D 9.336 mm per unita', FaceScape 1, FaceVerse 161.81): vicino alla
    mediana del training (2.04 mm), dentro l'intervallo dei suoi domini, 10-14k vertici sulle mesh pulite (training
    3k-60k);
  - `iterations` = 10; `adaptive` = False; `featuredeg` = 30; `checksurfdist` = True con `maxsurfdist` = 1% della
    diagonale del box della mesh d'ingresso (default di MeshLab: l'unico parametro che dipende dall'ingresso);
    `splitflag`, `collapseflag`, `swapflag`, `smoothflag`, `reprojectflag` = True; `selectedonly` = False.
  - Dopo il filtro: se la mesh ha piu' di una componente si tiene la piu' grande (per facce) e si tolgono i vertici non
    referenziati (contato). Nient'altro: niente riparazione degli spigoli non-manifold (contati).
  - FaceVerse: facce invertite DOPO il remesh (F[:, ::-1], come `zs_stage.py --flip-faces` degli store).
- **Si applica a ogni mesh d'ingresso**, original compresa: le 6 topologie (original, remesh, down8k, noisy, up60k,
  crop) dei 100 soggetti valutati di ogni vista. Non usa etichette ne' GT.
- **Validita'** su tutte le 1.800 mesh: una componente, nessun vertice isolato, operatori calcolati (autovalori finiti,
  k = 128). Spigoli non-manifold, lato (mediana, CV), vertici, area dopo / prima per etichetta: riportati. Determinismo:
  10 mesh fisse (le prime 10 nell'ordine dei nomi di HIFI3D) ridiscretizzate due volte devono coincidere.
- **Se una mesh non e' valida**: dopo la pulizia sopra, se resta non valida diventa NaN e le sue righe escono da tutte
  le colonne (maschera comune, contata). Se in una vista non e' valido piu' dell'1% delle mesh, quella vista non entra
  nella regola (sez. 5), che allora chiede 2 domini su quelli restanti.

## 2. Pipeline (quella ufficiale, cambia solo l'ingresso)

- Viste: FaceScape dev (`datasets/DEV_FACESCAPE/eval_view`), HIFI3D (`datasets/HIFI3D/eval_view`), FaceVerse neutra
  (`datasets/FACEVERSE_ZS/eval_view`); soggetti e verso delle facce dai `subjects.json` degli store ufficiali
  (`d3_pools.POOLS`, importato in sola lettura): stessi soggetti e stesse 6 topologie della valutazione ufficiale.
  **Esclusi**: Ava-256 (test confermativo vergine), FaMoS TEST; FaMoS TRAIN e FLAME di D1 non fanno parte di D4.
- Tabella di scala: `v3_work/trainer/tools/build_scale_table.py --view-dir <mesh canoniche> --domain <dominio>` (area in
  mm^2 della mesh CANONICA; dominio quello della tabella ufficiale della vista).
- Operatori: `v2_work/potential/areanorm_operators.py --k-eig 128` (come gli store e bp_e3).
- Embedding: `eval_v3.py -- aau/zs3dmm/zs_embed.py` con gli argomenti degli store (`bp_e3.sbatch`:
  `WBES_V3_FACTORIZED_OUT=full`, `--checkpoint_selector best_by_clean`, `--pair_mode cross_topology --seed 1234`, ...),
  checkpoint e `--dist_npz` dello store della vista e del braccio.
- **Bracci**: factorized s1234 e factorized s2345 (primari; d_P = ||u|| x dp_per_unit, d_F calibrata con la c ufficiale
  di `factorized_calibration.csv`, `diag.arm_distances`); ctrlfr s1234 (secondario, ||z||).
- **Controllo della catena**: factorized s1234 sugli operatori dello store ufficiale di HIFI3D
  (`datasets/V3_OPS_CACHE/8e8f81d5f0204394`), nello stesso job, ridà lo store (scarto massimo <= 1e-3).

## 3. Righe, gruppi, repliche

- Righe di all_cross con GT FR e SR (`bp_paired_e2.all_cross_rows`, quelle dell'emendamento 3 dei concorrenti); per
  FaceVerse neutra le righe e le GT di FaceVerse (stessi soggetti e topologie).
- Gruppi:
  - **nocrop_cross (primario)**: righe senza crop ai due lati, topologie diverse (20 coppie ordinate); le chiavi devono
    coincidere con `fact_paired.rows_for` (per FaceVerse `mesh_pair_nocrop`);
  - **stessa topologia**: le stesse coppie di soggetti con la stessa etichetta ai due lati, 5 etichette senza crop (come
    la sez. 6 dell'emendamento 3); anche per etichetta (descrittivo);
  - **all_cross** e **righe col crop** (un lato crop): secondari.
- Spearman sulle righe del gruppo, pesato c_a c_b con i conteggi della replica (`diag.wspearman`, identico a righe
  ripetute). Bootstrap per soggetto, 1000 repliche, conteggi di `fact_paired.main` (`default_rng(seme)`) col seme del
  gruppo primario ufficiale della vista: **HIFI3D 990708, FaceScape 796786, FaceVerse 271049** (quelli di D3); le stesse
  repliche per tutti i gruppi e tutte le colonne (confronti appaiati).
- Maschera comune: righe con tutte le colonne finite (bracci originali e canonici, GT); righe tolte contate.
- Colonne: per braccio, ingresso originale (store ufficiali) e canonico. GT dichiarate: d_P con SR (primaria), d_F cal.
  con FR (vincolo), ctrlfr con FR (secondaria); tutte le combinazioni colonna x GT si salvano (descrittive).
- Delta appaiato = rho(canonico) - rho(originale), stesse righe e repliche; IC 95% percentile, P(delta <= 0).

## 4. Controlli

- K1: i bracci sull'ingresso originale su nocrop_cross ridanno i valori di D3 (`d3_spearman.csv`, righe `cross`, colonne
  `factorized_s1234|dP|shape` e `factorized_s2345|dP|shape` con SR): stesso numero di righe (FaceScape 88.725, HIFI3D
  98.224, FaceVerse neutra 99.000) e scarto <= 1e-6 sul punto. Se la maschera comune di D4 toglie righe in piu', si
  dichiara e il confronto si fa sulle righe comuni.
- K2: controllo della catena (sez. 2). K3: determinismo del remesher (sez. 1). K4: validita' (sez. 1).
- K5: chiavi e GT di nocrop_cross = `fact_paired.rows_for` (`bp_paired_e2.check_rows`).

## 5. Lettura preregistrata

Per ognuno dei due semi factorized (s1234, s2345), su nocrop_cross:

- **P(seme)** vale se (i) in almeno **2 domini su 3** delta(d_P, SR) >= **+0.02** con IC 95% sopra 0, e (ii) in
  **tutti e 3** i domini delta(d_F cal., FR) >= **-0.02** (stima puntuale).
- **UTILE** se P vale in entrambi i semi; **NON CONFERMATO** se in uno solo; **NO** se in nessuno.
- In piu' **DANNOSA** se in entrambi i semi almeno 2 domini su 3 hanno delta(d_P, SR) <= -0.02 con IC sotto 0.

Perche' queste soglie: +0.02 e' il gap misurato su HIFI3D (+0.020 / +0.019) e circa il 40% di quello di FaceScape
(+0.053 / +0.050); sopra la semi-ampiezza tipica degli IC dei delta appaiati di D3 (0.01-0.02), quindi risolvibile.
Un passo al test che costa un remesh per mesh deve comprare almeno quanto il gap piu' piccolo su due domini. La soglia
e' esigente: con i gap misurati, P chiede di chiudere quasi tutto il gap su HIFI3D (o di andare oltre il gap su FaceVerse,
+0.01) oltre a FaceScape; misura l'utilita' pratica, non il meccanismo. -0.02 su FR: lo stesso ordine, perche' il
guadagno di forma non si paghi con la forma-con-taglia. Due semi, come le letture di D3.

Descrittivo, accanto (non regole): ctrlfr s1234 (FR e SR); per dominio e braccio il **residuo** stessa topologia -
nocrop_cross prima e dopo la canonizzazione, con IC, e la loro differenza; la frazione recuperata delta(nocrop_cross) /
(residuo prima), riportata solo dove il residuo prima ha IC sopra 0; all_cross e righe col crop; stessa topologia per
etichetta.

## 6. Costo misurato per mesh

- Remesh: tempo per mesh di tutte le 1.800 (un thread per processo, processi in parallelo sul nodo), mediana e p90 per
  etichetta.
- Operatori: canonico contro originale su un campione fisso (i primi 3 soggetti valutati per id x 6 topologie x 3 viste
  = 54 coppie), stesso processo, un thread, alternati (originale, canonico) per non confondere la deriva del nodo.
- Embedding: tempo a parete del job GPU per le 600 mesh canoniche di HIFI3D contro le 600 originali del controllo della
  catena (stesso job, stessa GPU, factorized s1234).
- Confronto: (remesh + operatori canonici) contro operatori sull'originale, per mesh.

## 7. Risorse, tempi, pulizia

- CPU: partizione prioritized. GPU: A100 di nv-ai-04 (`aicentre-a100`, QoS unprivileged, `--requeue`, al massimo 2 GPU,
  `gpufree` prima), solo per gli embedding, da finire entro le 11:00 (poi le A100 servono alla valutazione
  dell'ablazione). Se alle 11:00 mancano embedding, il job si cancella e i bracci mancanti si dichiarano; la regola si
  valuta solo se entrambi i semi factorized sono completi sui 3 domini, altrimenti "non valutabile".
- Mesh canoniche in `datasets/DIAG_D4/<vista>/in`, operatori in `datasets/V3_OPS_CACHE/diag_d4/<vista>/ops` (fuori da
  git, cancellati a fine lavoro); embedding e statistiche per mesh in `aau/runs/evidence/diagnostics/d4/` (npz fuori da
  git); numeri in `d4_spearman.csv`, `d4_delta.csv`, `d4/controls.json`, `d4/remesh_stats.json`, `d4/cost.json`;
  risultati in `results_d4.md`.
- Codice nuovo in `aau/diagnostics/d4_*.py` (`d4_remesh.py`, `d4_stats.py`, `d4_cost.py`), `d4.sbatch`,
  `d4_embed.sbatch`, `d4_run.sh`; `fact_paired.py`, `fact_summary.py`, `diag.py`, `d1_stats.py`, `d3_*.py`,
  `eval_ablation*.py` e `v3_work/stream/` importati in sola lettura o non toccati.
