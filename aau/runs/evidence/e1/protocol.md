# E1: varieta' di 3DMM contro quantita' di identita' (protocollo, scritto PRIMA dei numeri)

8 ottobre 2026, prima di qualsiasi training o valutazione delle celle nuove. Piano: `paper/PLAN_MASSIVE.md`,
sezione 13. Codice in `aau/evidence/e1_factorial/`, run e risultati in `aau/runs/evidence/e1/`.
Questo file non si modifica dopo i numeri: `e1_summarize.py` lo copia in testa a `summary.md`.

## Regola di lettura (decisa dal PI, dichiarata qui prima dei numeri)

**La varieta' e' sostenuta se C3F - C2F e C3M - C2M sono entrambe > 0 sulla distanza graduata
`nocrop_cross` di HIFI3D, con IC 95% che esclude lo 0.**

Come la applico, senza margini di interpretazione dopo:
- metrica: Spearman fra distanza latente e GT `maxabs` di HIFI3D, mesh-pair, 20 coppie ordinate di
  topologie senza crop, clean (la colonna "HIFI3D mesh-pair senza crop" di `aau/runs/data_scale_ood/curve.md`);
- differenza appaiata sulle STESSE righe e sulle STESSE 1000 repliche bootstrap per soggetto (seme 1234);
  "IC che esclude lo 0" = estremo inferiore dell'IC 95% > 0;
- servono ENTRAMBE le differenze; se ne passa una sola, la varieta' NON e' sostenuta e si dice quale;
- si applica separatamente a 10.548 e a 21.096 passi. Lettura principale a 21.096 (fine di ogni run);
  10.548 accanto. Se i due numeri di passi non concordano, lo si scrive, senza fonderli.

Letture secondarie (mie, simmetriche alla primaria, solo descrittive):
- quantita': C2M - C2F e C3M - C3F sulla stessa metrica, con la stessa regola;
- G1 contro C3M e contro C3F: se G1 - C3M ha IC che contiene lo 0 o e' > 0, GNM da solo basta a
  spiegare il livello di C3M su HIFI3D;
- le stesse differenze sugli altri domini di test (sotto), senza regole di decisione.

## Celle

| cella | training | da dove |
| --- | --- | --- |
| C3M | BFM 392 + ICT 54.008 (4.008 ICT-5000 + 50.000 nuove) + GNM 10.000 | run su scala 1060130, checkpoint epoch036 / epoch072: nessun training nuovo |
| C2M | BFM 392 + ICT 54.008 | nuovo run |
| C2F | BFM 392 + ICT 5.401 (401 ICT-5000 + 5.000 nuove) | nuovo run |
| C3F | BFM 392 + ICT 4.557 (338 + 4.219) + GNM 844: stesso totale non-BFM di C2F (5.401), rapporto ICT/GNM 5.40 come C3M | nuovo run |
| G1 | GNM 10.000 | nuovo run |

- Stesso trainer (`v2_work/fastio/train_steps.py`, invariato), stessa ricetta v1 flag per flag, stessa
  spec dei dati, stessa GT, stesso held-out (1.200) ed eval online, seme 1234 (modello, epoche, blocchi).
- Sottoinsiemi stratificati per sorgente e insieme di mesh (ICT nuove: 6 topologie + 8 espressioni;
  GNM: 1 o 2 espressioni), seme 1234; le ICT di C3F sono un sottoinsieme di quelle di C2F
  (`make_e1_splits.py`, `aau/evidence/e1_factorial/subsets.json`).
- Passi: checkpoint a 10.548 e 21.096 passi (epoche 36 e 72 da 293 passi). Quote di passi per epoca:
  BFM 26 (8.87%); celle a due domini ICT 267; celle a tre domini ICT 225 + GNM 42 (come C3M); G1 GNM 293.
- Blocchi: K = 40 (C2M), 4 (C2F, C3F), 6 (G1); C3M ne ha 46. Un blocco deve contenere i soggetti di
  un'epoca per dominio. C2M ha T nominale 91.709 (E = 313) per cambiare blocco alle stesse epoche di C3M
  ed e' fermato dopo il checkpoint dell'epoca 72: come C3M, i suoi checkpoint sono istantanee di un run
  piu' lungo.

## Attenzione prima di leggere i numeri: la quantita' VISTA

C3M a 21.096 passi ha toccato 10 blocchi su 46: le identita' viste sono circa 14.300, non 64.400
(`design.md`, calcolato con le funzioni del trainer). Per costruzione C2M ne vede quasi altrettante
(97%). Le celle F vedono 3.093 (10.548 passi) e 5.793 (21.096) identita'. Il contrasto di quantita'
effettivo e' quindi circa 2.5x in identita' viste (e 8 contro 18 epoche di esposizione per identita'),
non 10x. La regola primaria (varieta') non ne dipende; le letture di quantita' vanno lette cosi'.

## Domini di test (stesso protocollo e stesse repliche di `aau/runs/data_scale_ood/curve.md`)

- HIFI3D (100 soggetti x 6 topologie, pipeline `aau/zs3dmm`, bracci `scale_<tag>`): Spearman con la GT
  `maxabs` su `nocrop_cross`, `all_cross`, `subject_pair_mean` (clean); riconoscimento (rank-1, AUC di
  verifica) sulle 5 topologie senza crop, dagli embedding.
- FaceVerse con espressioni (100 soggetti, convenzione BFM `_flip`): riconoscimento (rank-1, AUC).
- NoW validation: tau di Kendall per immagine sui 3 metodi pre-registrati (`aau/recon`).
- FLAME zero-shot (`aau/flame`, 100 soggetti x 6 topologie): Spearman `nocrop_cross`, `all_cross`,
  `subject_pair_mean`.
- IC 95% bootstrap per soggetto, 1000 repliche, seme 1234; differenze fra celle appaiate sulle stesse righe
  e sulle stesse repliche.

## Limiti dichiarati prima

- Un solo seme per cella: l'IC copre il campionamento dei soggetti di test, non la variabilita' da run a run
  del training (stimata in `aau/data_scale/PLAN.md` intorno a +-0.035 su un training BFM corto, metrica diversa).
  Differenze di pochi centesimi con IC che esclude lo 0 vanno lette con questo in mente.
- C3M e' un run gia' esistente; le altre celle sono nuove (stesso codice, nodi e tempi diversi).
