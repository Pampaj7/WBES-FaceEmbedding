# Training su "form" (GT-FR di E12): protocollo dichiarato prima dei numeri

Scritto il 9 ottobre 2026 alle 14:00 dal coder, su richiesta del PI, PRIMA di lanciare qualunque run di questo
protocollo e prima che esistano le GT di training di E12 (`datasets/CANONICAL_GT/train/`, in produzione: nessun
numero di training su FR/SR esiste). L'impronta sha256 sta in `factorized_protocol.sha256`; ogni modifica va in un
emendamento datato. Codice e definizioni: `aau/runs/evidence/trainer_v3/factorized.md`.

## Domanda dichiarata

**Un modello addestrato su form si avvicina a NICP con la GT FR?** Riferimento (E12, `aau/runs/evidence/e12/summary.md`
sez. 5, HIFI3D `nocrop_cross`, GT "F + rigida robusta" = FR): NICP P2Tri 0.367 [0.289, 0.449], ICP + Chamfer 0.325,
e108 (modello attuale, cieco alla taglia) 0.194 [0.110, 0.282], Chamfer eval 0.155.
Domanda secondaria: basta cambiare GT e ingresso (ctrl-FR) o serve la fattorizzazione (factorized)?

## Bracci

Tutti: trainer v3, `--area robust --area-robust smooth` (arearobust) + `--sampler balanced --domain-alpha 0` (bal),
ingresso a scala globale (`--input-norm global`, mm nel frame di GT-F di E12, L0 = 100 mm, tabelle di scala di
`tools/build_scale_table.py`), ricetta v1 del run su scala, EMA 0.999, forward sequenziale con `--fast-data`.

| run | dati | testa e GT | hardware |
|---|---|---|---|
| `factorized` s1234, s2345 | sottoinsieme C3F (split_c3f, store C3F ricostruito), T 21.096, S 293, eval a 10.548 e 21.096 | `--head factorized`: u su GT-SR grezza di E12 x kappa, s su log S_i (centroid size di E12, mm), `--scale-aug 0.8,1.25`, `--lambda-size 1` | 1 A100 ciascuno (unprivileged, requeue) |
| `ctrlfr` s1234, s2345 | come sopra | testa standard (z), GT-FR TARATA di E12 (mediane per dominio sulla maxabs) | 1 A100 ciascuno |
| `factorized` C3M | split del run su scala (65.600 identita', 64.400 di training), spec del run su scala (K 46), T 60.000, S 293 per rank | come factorized C3F | DDP su 6 L40S di a768-l40s-06 (QoS normal) |

- **kappa** (GT-SR adimensionale; i margini della loss v2, 0.05 e 0.02, sono in unita' di GT): la media geometrica dei
  fattori di taratura per dominio di E12 per SR (`gt_sr_bfm_ict_gnm_calib.json`, `factor`), UN valore per tutti i
  domini (cosi' d_P = ||u_i - u_j|| x dP_per_unit / kappa resta in unita' assolute). Si calcola dai json di E12
  prima del lancio e si scrive nella riga di lancio; nessun valore e' scelto guardando risultati.
- Il secondo seme cambia `--seed` e `block_seed` della spec (politica del run su scala), nient'altro.
- C3M e' un primo assaggio del run massivo: descrittivo, fuori dalla regola.

## Distanze del modello

- `factorized`: **d_F = sqrt((S_i - S_j)^2 + S_i S_j d_P^2)**, S = exp(s) (mm), d_P = ||u_i - u_j|| x dP_per_unit / kappa
  (corda fra pre-forme a centroid size unitaria; Dryden e Mardia, forma esatta). Riportate anche d_P contro SR e
  |delta s| contro FR.
- `ctrlfr`: ||z_i - z_j||.
Per coppia di mesh di soggetti diversi e topologie diverse senza crop, media per coppia di soggetti (come `nocrop_cross`
di E12), pesi EMA dell'ultimo checkpoint (21.096 passi).

## Valutazione

- **Primaria:** HIFI3D, 100 soggetti di `select_subjects(..., 1234)`, `nocrop_cross`, Spearman con **GT-FR**
  (`datasets/CANONICAL_GT/eval/hifi3d_fr.npz`), IC 95% bootstrap per soggetto (1000 repliche, seme 1234).
- Secondarie (riportate, fuori dalla regola): HIFI3D con GT-SR e maxabs; dev FaceScape (FR, SR, maxabs; rank-1 con
  espressioni); FaceVerse con espressioni (FR, SR; rank-1); FaMoS TEST (FR, SR) se la vista con operatori e le
  tabelle di scala sono pronte entro la fine dei training, altrimenti dichiarato mancante; accuratezza di s contro
  log S_i delle viste di eval.
- Confronti: NICP P2Tri, ICP + Chamfer e Chamfer eval con le stesse GT, dai csv di E12
  (`aau/runs/evidence/e12/methods_spearman.csv`); se le righe di E12 sono ricostruibili sugli stessi soggetti e
  coppie, differenze appaiate sulle stesse repliche.

## Regola (ultimo checkpoint EMA, ogni seme; il verdetto vale se i due semi concordano)

Sia g = (rho_modello - 0.194) / (0.367 - 0.194) la frazione del divario e108 -> NICP chiusa con GT-FR su HIFI3D.
1. **"raggiunge NICP"**: rho_modello >= 0.289 (estremo inferiore dell'IC di NICP) e IC del modello che contiene 0.367
   o sta sopra;
2. **"si avvicina a NICP"**: altrimenti, g >= 0.5 (rho >= 0.281) e estremo inferiore dell'IC del modello > 0.194;
3. **"non si avvicina"**: altrimenti.
Secondaria: factorized - ctrlfr (stesso seme) >= +0.03 in entrambi i semi -> "serve la fattorizzazione"; |delta| < 0.03
in entrambi -> "basta GT e ingresso"; altrimenti "non risolto". Se i semi discordano si riporta "discordante".

## Limiti dichiarati

- Due semi per braccio C3F: la variabilita' fra semi e' stimata solo grossolanamente.
- Store C3F ricostruito (lo store dei bracci dell'8 ottobre non esiste piu'): gli operatori delle mesh dei tar sono
  ricalcolati, segni degli autovettori possibilmente diversi; tutti i run C3F di questo protocollo leggono lo stesso
  store nuovo.
- Il rumore del training resta in unita' di L0 (0.05-2 mm), piu' piccolo in mm che in arearobust.
- BFM: le original REMESH sono allineate per similarita' una per una, la taglia BFM e' quasi costante.
- FR contiene la rigida robusta per identita' e il termine dei centroidi (d_FR^2 = ||c_i - c_j||^2 + d_F^2): d_F
  non lo rappresenta.
