# mm_aug: moltiplicatori di dati (9 ottobre)

Codice in `v3_work/mm_aug/`:
- `aug.py`: l'API, la provenienza e le licenze;
- `transfer.py`: il trasporto e i controlli di validità;
- `selftest.py`: le prove;
- `stats.py`: le soglie, i controlli, lo spazio FR e i render.

Tutto si genera **al volo**: su disco ci sono solo i JSON e i PNG di questa cartella (1.9 MB). Non ho modificato `v3_work/mm/`, `stream/` né `trainer/`: li importo e basta (`sources.MMSource`, `FamosSource`, `targets.CanonTargets`).

## Cosa fa

`sample_view_spec(rng, cfg)` sceglie il tipo della vista secondo `cfg.p_kind` (default pure 0.4, hybrid 0.3, expr_transfer 0.15, rbf 0.15). Il template A è uno dei 3DMM di training dello stream: bfm2019, ict, gnm, flame2020.

| tipo | identità (neutra della GT) | vista |
|---|---|---|
| `pure` | z_A a code larghe (FaMoS: una persona TRAIN) | espressione nativa con p 0.6 |
| `hybrid` | mean_A + α Δ_A + β T_{B→A}(Δ_B), con (α, β) = (cos θ, sin θ) e θ ~ U(0, 90°) | espressione nativa con p 0.6 |
| `expr_transfer` | z_A pura | espressione di B ≠ A: FLAME 100, GNM 383, ICT 53, BFM 2019 100, oppure FaMoS (fotogramma − neutra, rigida robusta) |
| `rbf` | z_A più 1-3 bump (1-3 mm, raggio 10-30 mm) centrati nella regione | espressione nativa con p 0.6 |

La vista restituita contiene:
- `V` e `F`, nel frame canonico dello stream;
- `neutral_points` (1478, 3) e `gt_domain`, da passare a `CanonTargets`, oppure `gt_targets(spec, lib)` per avere FR, SR e S;
- `mm_factor`;
- `size_mask` (True se tutte le fonti d'identità hanno unità dichiarate; oggi lo è sempre);
- `checks` e `provenance`.

**Trasporto T_{B→A}.** Il campo di B sui punti della regione unificata va nel frame FLAME in mm veri (u_B R_B), poi sui vertici di A:
- **dentro la regione**: baricentrico, esatto sui punti;
- **fuori, nella banda geodetica di 40 mm**: raccordo biarmonico, che riempie anche occhi, bocca e narici;
- **oltre la banda**: zero.

La GT della neutra è esatta: mean_u + α id_u z_A + β g_B, calcolata sul template A.

**Validità.** Tre controlli contro il template:
- triangoli capovolti;
- triangoli degeneri (area sotto il 2%);
- auto-intersezioni nuove, cioè fuori dalla zona già intersecata nel template, dilatata di 2 anelli. Le labbra delle medie si compenetrano agli angoli della bocca: 121-182 coppie per template.

La soglia per template è il p99 dei puri dello stesso template (`thresholds.json`, 400 viste per template): un campione aumentato non deve essere peggiore di quelli nativi. Un campione scartato si riestrae. `cfg.validate="cheap"` (capovolti e degeneri) costa **0.046 s per vista** (misurato, 120 viste); `"full"` aggiunge circa 0.2-1.8 s per vista.

**Provenienza.** `provenance` contiene: il seme, il tipo, A, B, α, β e θ, tutti i coefficienti, i bump, la persona e il fotogramma FaMoS, il tentativo, la configurazione, le fonti, le licenze e `redistributable` (la più restrittiva vince). `rebuild_view_spec(prov)` ricostruisce la vista bit per bit, anche dopo un giro in JSON.

**Licenze** (`LICENSES`):
- ICT (MIT) e GNM (Apache-2.0) sono verificate sui file;
- FLAME e BFM 2019 sono marcate non ridistribuibili ma **non verificate nel repo**;
- FaMoS è MPI, dalle note.

## Verifiche (eseguite)

**`selftest.py`: 24/24 OK** (`selftest.json`). Copre:
- trasporto lineare (1.8e-13) e interpolante sui punti (0.0);
- determinismo e ricostruzione dalla provenienza;
- licenze;
- nessun modello dev o test caricato, e le fonti di test vengono rifiutate;
- nel trasferimento d'espressione, neutra e FR identiche a quelle di A (anche in `checks.json`: 150/150 per template).

**`checks.json`**: 150 viste grezze, cioè prima del rifiuto, per tipo e template.

| | fuori soglia (grezze) |
|---|---|
| puri | 0-2.7% |
| ibridi | 0-2.7% |
| rbf | 2.7-6.7% |
| trasferimenti d'espressione | 0.7-9.3% (ICT 0.7%, FLAME 9.3%) |

I campioni fuori soglia vengono riestratti.

**Continuità del raccordo.** Lo strain massimo sugli spigoli che attraversano il bordo del raccordo vale 1.4-5.2 negli ibridi, sempre ≤ quello dentro la regione (2.2-7.4). Il taglio netto senza raccordo arriva a 65-142.

**GT esatta contro la GT letta dalla mesh.** La mediana del massimo per campione è 0.04-0.50 mm, il caso peggiore 2.2 mm (ICT). Nei puri, per effetto della sola topologia di lavoro, è 0-0.75 mm.

**Render**: `hybrids_10.png` e `expr_transfer_10.png`. Non si vedono cuciture, e la bocca aperta di FaMoS si trasferisce. Attenzione: il renderer normalizza la taglia di ogni mesh.

## Spazio FR (GT di E12, mm): `fr_stats.json`, `subspace.json`, `fr_pca.png`

Campioni: 2000 puri e 2000 ibridi per template, 500 rbf per template, le 80 persone FaMoS.

- **I domini si sovrappongono.** Le medie distano 1.7-2.7 mm fra loro; un'identità dista circa 4 mm dalla propria media.
- **Gli ibridi non collassano su una media.** Mediana della distanza dalla media più vicina:

  | template | ibridi | puri |
  |---|---|---|
  | bfm2019 | 3.95 mm | 3.96 mm |
  | ict | 3.83 mm | 3.71 mm |
  | gnm | 4.02 mm | 4.17 mm |
  | flame2020 | 3.90 mm | 3.81 mm |

  La media più vicina è quella di A nel 45-65% dei casi, come per i puri (44-64%).
- **"Fuori" dai pool puri, solo di poco.** Rapporto fra la distanza dal vicino più prossimo nei pool puri (ibridi) e la stessa distanza fra puri (leave-one-out):

  | template | rapporto | IC 95% |
  |---|---|---|
  | bfm2019 | 0.98 | [0.97, 0.99] |
  | ict | 1.09 | [1.08, 1.10] |
  | gnm | 1.02 | [1.01, 1.03] |
  | flame2020 | 1.04 | [1.04, 1.05] |

- **Fuori dal sottospazio di A.** Residuo dopo la proiezione sulla base d'identità di A più i moti rigidi: per i puri è 0; per gli ibridi è il 6.5-13% della deformazione in RMS (mediana; 0.35-0.65 mm) e cresce con β, fino al 9-19% per β > 0.8. È novità reale, ma modesta: le basi da 100-300 modi coprono già molto.
- **Vicinanza ai domini di test.** Distanza minima di ogni campione dai pool di test (HIFI3D, FaceVerse, FaceScape, 500 identità ciascuno):
  - il p1 degli ibridi è ≥ al p1 dei puri in tutti e tre i pool;
  - il minimo assoluto degli ibridi supera quello dei puri di +0.23, +0.03 e +0.14 mm (rbf: +0.47, +0.21 e +0.32 mm);
  - la quota di ibridi sotto il p1 dei puri è 0.70% [0.54, 0.90], 1.08% [0.85, 1.29] e 0.70% [0.53, 0.89].

  **Nessun moltiplicatore si avvicina ai test più dei puri.**

## Limiti e assunzioni

- **Spostamenti in mm veri, senza normalizzare la taglia fra domini.**
- **Fuori dalla regione B non conta.** Orecchie, collo e nuca restano quelli di A, raccordati entro 40 mm.
- **Bocca a labbra unite.** Sui template con la bocca chiusa, le espressioni a bocca aperta stirano la commessura. È per questo che FLAME e GNM hanno lo strain più alto dentro la regione nei trasferimenti (fino a 34) e i rifiuti fino al 9%.
- **Esclusi:**
  - bfm3ddfa, bfm2019_face12/fullhead e flame2023: non sono domini dello stream;
  - FaMoS come fonte di identità per gli ibridi: c'è solo come dominio puro e come fonte di espressioni;
  - l'integrazione nel produttore: `stream/` è di un altro agente.
- **Validazione nello stream.** Ho scelto `validate="cheap"` come default. Le auto-intersezioni sono misurate offline (`checks.json`) e i tassi degli ibridi sono pari a quelli dei puri.
