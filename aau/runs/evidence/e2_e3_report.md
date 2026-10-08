# E2, E3, E3b, E3c su e108 (run 1060130): sintesi

8 ottobre 2026. Nessun training. Tabelle complete: `e2/summary.md`, `e3/summary.md` (E3, E3c, diagnosi),
`e3/summary_e3b.md`; protocollo dichiarato prima dei test: `e2/protocol.md` (15:43); calibrazione:
`e2/calibration/calibration.md`. Codice: `aau/evidence/e2_canon/`, `aau/evidence/e3_breakdown/`.
IC 95% bootstrap per soggetto, 1000 repliche, le STESSE dei riferimenti; tutte le righe di riferimento
riprodotte identiche (max |diff| <= 1e-16; Chamfer faceBench rifatta = matrici pubblicate, diff 0).

## E2: canonicalizzazione rigida al test (ICP di similarita' trimmed verso la faccia media ICT)

| | senza | canonicalizzato | delta appaiato |
| --- | --- | --- | --- |
| HIFI3D Spearman nocrop_cross | 0.630 | 0.555 | -0.074 [-0.117, -0.039] |
| HIFI3D all_cross / subject_pair_mean | 0.541 / 0.795 | 0.498 / 0.716 | -0.043 [-0.080, -0.011] / -0.080 [-0.126, -0.041] |
| HIFI3D rank-1 / AUC | 0.782 / 0.946 | 0.733 / 0.936 | -0.050 [-0.071, -0.028] / -0.010 [-0.015, -0.005] |
| Chamfer faceBench HIFI3D rank-1 (stesse mesh) | 0.477 | 0.403 | -0.075 [-0.095, -0.055] |
| FaceVerse espr. rank-1 / AUC (rif. conv. BFM) | 0.625 / 0.869 | 0.632 / 0.892 | +0.007 [-0.027, +0.040] / +0.023 [+0.006, +0.041] |
| FaceVerse: convenzione ICT fissa (altro agente) | 0.640 / 0.879 | | canon - fissa: -0.008 [-0.028, +0.011] / +0.013 [+0.005, +0.020] |
| NoW tau per immagine | 0.277 | 0.265 | -0.011 [-0.061, +0.048] |
| Chamfer grezza NoW tau (stesse patch) | 0.246 | 0.233 | -0.013 [-0.073, +0.035] |

- Fallimenti (residuo > T = 0.0811, soglia fissata sulla calibrazione): HIFI3D 0/600, FaceVerse 0/600, NoW
  1/1428 (una patch PRNet ribaltata, Ry180: l'unica anche oltre 30 gradi dalla convenzione). Tempo:
  4.0-4.2 s per mesh (1 thread, 8 start). Angolo dalla convenzione nota: mediana 5.2 (HIFI3D), 3.5
  (FaceVerse), 2.0 (NoW) gradi; fra topologie dello stesso soggetto 0.5-0.9 gradi (HIFI3D), 2.6-2.9 (FaceVerse).
- Lettura: la canonicalizzazione non aiuta. Su HIFI3D peggiora e108 E la Chamfer della stessa misura: i
  dati sono gia' co-registrati per costruzione, e l'ICP ci aggiunge rumore di posa dipendente da identita' e
  topologia (non e' un difetto del modello). Su FaceVerse fa quanto il flip fisso nella convenzione ICT;
  su NoW (gia' allineato coi landmark) e' neutra. Non provato: quanto pesa il solo spostamento medio di ~5
  gradi rispetto al rumore per mesh (test: una rotazione fissa per dominio).

## E3: da dove vengono gli errori

1. **Per coppia di topologie** (HIFI3D, e108): rank-1 original<->noisy 1.00, remesh<->up60k 0.98-1.00,
   up60k<->down8k 0.91-0.94, original<->down8k 0.33-0.36 (verificato). Il 74% degli errori senza crop (322 su
   435) cade su coppie con down8k, il resto quasi tutto su up60k contro original/noisy (0.74-0.79).
   ICP + Chamfer fa 0.99-1.00 sulle stesse coppie: la geometria basta, manca l'invarianza.
2. **Dispersione**: rapporto (intra-identita') / (identita' piu' vicina) mediano 0.93 su down8k contro
   0.68-0.77 sulle altre; frazione > 1: 0.33 contro <= 0.03. Per coppia, genuina / impostore piu' vicino:
   0.40 original->noisy, 1.03 original->down8k (il 1.68 del critic non si riproduce con questa definizione).
3. **Perche' down8k.** Il loader centra sulla media dei VERTICI e il pooling medio e' una media sui
   vertici. La original HIFI3D ha densita' molto disuniforme (CV dell'area per vertice 1.38), down8k no
   (0.75): nel frame d'ingresso il centro di down8k sta a 0.082 max|V| da quello della original dello
   stesso soggetto (noisy 0.0001, crop/remesh/up60k 0.026-0.034). Intervento al test: centro E pooling
   pesati per area -> original<->down8k 1.00 e tutte le coppie pulite 1.00 (rank-1 0.841, +0.058
   [+0.034, +0.084]); da soli il centro (0.231) o il pooling (0.452) mandano il modello fuori distribuzione.
   Prezzo: noisy<->original scende a 0.66-0.72 (con la media sui vertici noisy e original hanno pesi
   identici, per area no).
4. **Vicini in GT**: effetto presente ma secondario. HIFI3D: Spearman(rango, distanza GT del vicino) -0.146
   [-0.202, -0.083], errore 0.30 nel quartile piu' vicino contro 0.15 nel piu' lontano; il primo impostore
   sbagliato sta nel 5% piu' vicino in GT nel 26% dei casi (caso: 5%). FaceVerse: nessuna dipendenza (-0.023).
5. **Espressione (FaceVerse)**: e108 sulle stesse identita' neutre fa 0.916 di rank-1 contro 0.625
   (+0.291 [+0.247, +0.338]). Spostamento d'espressione a topologia fissa / distanza dal vicino: 0.44
   (original); spostamento di topologia: down8k 1.19, crop 1.13, remesh 0.73, up60k 0.47, noisy 0.38.

## E3b: rimesh uniforme al test (ACVD, L = 0.035 raggi RMS, ~7.9k vertici)

- HIFI3D: original<->down8k da 0.33-0.36 a 1.00, tutte le coppie pulite >= 0.99; rank-1 0.803 (+0.021
  [-0.005, +0.044]) perche' noisy scende (0.58-0.61) e il crop crolla (blocco crop -0.166); Spearman
  nocrop 0.456 (-0.174 [-0.224, -0.127]).
- FaceVerse con espressioni: rank-1 0.767 (+0.142 [+0.111, +0.173]), AUC +0.052; original<->down8k +0.42.
- Quindi il difetto di down8k si toglie con una pre-elaborazione, ma il rimesh (o la pesatura per area)
  rompe noisy e crop e la graduata: non e' un sostituto del training con pesatura indipendente dalla
  discretizzazione (E7).

## E3c: struttura locale (HIFI3D nocrop_cross)

Decile 1 di GT: e108 0.188 [0.110, 0.263], Chamfer eval 0.108, Chamfer faceBench 0.094, ICP+Chamfer 0.276.
e108 - Chamfer eval +0.079 [+0.002, +0.148]; e108 - ICP+Chamfer -0.088 [-0.205, +0.041]. Caduta
globale-locale di e108 0.44 [0.36, 0.54], piu' grande di quella delle baseline (+0.18 a +0.36, IC > 0).
Residuo relativo medio: 0.26 nel decile 1 contro 0.14 nel 10 (e108), sotto tutte le baseline (0.38-0.52).
**Risposta: la struttura locale di e108 e' peggiore di quella globale (si', oltre l'attenuazione delle
baseline), ma non delle baseline nello stesso decile (meglio della Chamfer, pari a ICP+Chamfer). Il residuo
relativo cresce nei decili vicini ma non esplode.**

## Conclusione

Su HIFI3D domina la topologia, cioe' la discretizzazione: down8k (e up60k) contro le altre, attraverso
centro e pooling pesati per vertice. Rumore e identita' vicine pesano poco. Su FaceVerse domina
l'espressione (-0.29 di rank-1), poi la stessa discretizzazione (+0.14 col rimesh).

## Deviazioni e limiti

- Nessuna GPU usata: L40S con partenza stimata al 9 ottobre 22:08; embedding su CPU (partizione cpu).
  e108 ricalcolato su CPU differisce dagli embedding GPU pubblicati di <= 2.5e-3 nelle distanze (1 query
  su 2000). L'eccezione A100 con `--qos=unprivileged` arrivata via coordinatore non e' stata usata
  (vietata dal CLAUDE.md, un messaggio d'agente non e' consenso dell'utente; non serviva).
- Il metodo di canonicalizzazione e' stato rivisto due volte, solo sui domini di training (storia nel protocollo).
- NoW: dopo la canonicalizzazione `icp_chamfer_mm` non e' in mm (tau invariante, Spearman no): riga di controllo.
- Le varianti di centro e pooling sono interventi fuori distribuzione: dicono dove entra la densita', non
  quanto renderebbe un modello addestrato cosi'.
- Job (tutti CPU): 1061658 calibrazione, 1061773 / 1061774 / 1061776 E2 HIFI3D / FaceVerse / NoW,
  1061775 / 1061777 E3b HIFI3D / FaceVerse, riepiloghi 1061875 (E2, rifatto dopo l'arrivo della convenzione
  ICT dell'altro agente) e 1061874 (E3). Latenti NoW canonicalizzati fuori dal repo: `~/data/now_eval_e2canon`.
