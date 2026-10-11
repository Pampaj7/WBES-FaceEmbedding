# Ablazione 2x2 dello stream: emendamento 1 al protocollo (scritto PRIMA dei numeri)

Scritto l'11 ottobre 2026, mattina, dal coder, su decisione del PI dopo il critic sul protocollo (RISERVE; job
1068219-1068222, misure delle viste in `views_used`). Modifica `PROTOCOL.md` (sha256 e7df116b..., commit 27f4893) dove
indicato; il resto vale com'e'. **Le letture di riferimento sono quelle di questo emendamento** (`v3_work/stream/
eval_ablation_em1.py`, uscite in `em1/`); il riepilogo con le regole originali (`eval_ablation.py`, uscite in questa
directory) gira lo stesso, per trasparenza, ed e' superato da questo. L'impronta sha256 di questo file sta nel messaggio
del commit che lo introduce e in `PROTOCOL_emendamento_1.sha256`.

**Stato al momento della scrittura (nessun numero dell'ablazione), 06:50:** training A-D alle epoche 31-38 di 100, A'
alla 10, B' alla 6; nessun embedding delle famiglie di test, nessuno Spearman, nessun delta di nessuna cella. Esistono, da prove
dello script su checkpoint NON dell'ablazione: `eval_check.json` e le radici di prova in `aau/scratch/abl2x2_*`. La c
su `epoch010_ema` di A-D (job 1068232-1068235, sez. 6) e' calcolata dai soli held-out; i suoi valori non sono stati
letti prima di questo commit.

## 1. FLAME: generatore e risoluzione (critic, punto 1)

- **Fatto.** In B e D le viste di FLAME 2023 sono la patch nativa: 628-4.529 vertici per vista (down8k circa 630, remesh
  circa 1.258, original e noisy 1.787, crop poco sotto 1.787, up60k circa 4.525; `diagnostics/d1/gen.json`, insieme
  `flame2023`). La topologia di test piu' piccola, HIFI3D down8k, ha 3.310 vertici (dato del critic): cinque etichette
  su sei di FLAME stanno sotto. B - A confonde il generatore con la risoluzione.
- **Q1 riformulata:** "FLAME 2023 nativo (1.787 vertici) al posto di 1/4 dei passi" (B contro A, D contro C, a T uguale),
  non "un generatore in piu'".
- **Lettura descrittiva nuova:** E_F (e E_P, I) sulle righe `nocrop` con down8k su almeno un lato e sulle altre righe
  `nocrop`, con le stesse repliche (seme del gruppo `nocrop`), e la loro differenza replica per replica
  (`down8k_meno_altre`). Nessun verdetto.
- **Cella nuova B'** (`Bp`) = B con le viste di FLAME suddivise 1-a-4 coi punti medi (lineare, `igl.upsample`, come
  l'insieme `flame2023_s1` di D1), stessi semi (trainer 1234, produttori 20261009) e stessa configurazione di B:
  `STREAM_FLAME_SUBDIV=1` (opt-in, `producer.py --subdiv flame2023=1`, commit 8a25a3c). Viste di FLAME in B': original e
  noisy 6.986, crop circa 6.600, remesh circa 4.906, down8k circa 2.447, up60k circa 17.840 vertici: una etichetta su
  sei sotto 3.310.
  - Verifiche (`flame_subdiv/`): a opzione spenta, con le fonti di B (anche flame2023), shard identici byte per byte al
    commit 4e3114a delle celle in corso, ricetta uguale; run dir invariata (`test_partial_check.json`: base, adesso,
    copia congelata); ricetta c3m invariata (`recipe_c3m_check.json`); accesa: stesse identita', etichette,
    espressioni e semi del rumore, gruppi non FLAME identici, viste FLAME neutre = procedura di D1 (facce identiche,
    vertici entro 8e-8), rigenerazione 12/12 (`check.json`). Coi produttori veri le viste FLAME sono piu' care: 2.8
    viste/s con 12 processi nella prova, quindi B' avra' piu' riuso di B (misurato, sez. 3).
  - **Job 1068239** su a768-l40s-06 (il nodo di B), trainer dalle 06:20, copia congelata al commit 8a25a3c.
- **Letture con B':**
  - **R_F'** (la domanda "FLAME a risoluzione comparabile"): B' - A sulle 6 celle `nocrop`, soglia per cella degli
    effetti semplici (delta_S, sez. 2), regole della sez. 2a;
  - B' - B, descrittiva: effetto della risoluzione di FLAME a generatore uguale; B' anche su `all_cross`, righe col crop
    e down8k, descrittiva.

## 2. Regole e rumore (critic, punto 2)

**(a) Verdetti.** Per R_F, R_F' e R_P1 (6 celle, beneficio = E >= soglia della cella e IC_low > 0; danno = E <= -soglia
e IC_high < 0):
- **SI'**: beneficio in almeno 3 celle, in almeno 2 famiglie, e nessuna cella con danno (anche una sola: regola
  conservativa, invariata);
- **PARZIALE**: beneficio in almeno 2 celle (anche le due GT della stessa famiglia);
- **non confermato**: beneficio in una sola cella;
- **NO**: nessuna cella con beneficio (sez. 5: cosa vuol dire).
I danni: "danno confermato" con almeno 2 celle, "danno non confermato" con una. **R_P2**: "costa" con almeno 2 celle
con danno, "costo non confermato" con una, "non costa" con nessuna; le celle con IC_high < 0 sotto soglia si elencano.

**(c) Cella nuova A'** (`Ap`) = A col seme del trainer 2345 (`STREAM_SEED`); i semi dei produttori restano quelli di A
(20261009 + 1000 x nodo + 100000 x segmento), tutto il resto identico. A' misura il rumore di inizializzazione, ordine e
estrazioni del consumatore con la stessa legge dello stream, come la coppia di semi di C3F della tabella; non include
il campionamento delle identita' dei produttori (limite). **Job 1068228** su a768-l40s-05 (il nodo di A), trainer dalle
05:59, copia congelata al commit f9792e4 (codice del training identico a quello di A).
- **Soglia per cella**, per (insieme, gruppo, GT, distanza): delta_k = max(delta_k della tabella del protocollo,
  2.5 x m_k x |rho_A' - rho_A| / sqrt(2)), con m = 1 per un effetto principale (E_F, E_P), sqrt(2) per un effetto
  semplice o un contrasto fra due celle (B - A, B' - A, ...), 2 per l'interazione; tabella del gruppo: `nocrop` (e
  down8k, held-out, FLAME), `all_cross`, righe col crop (e crollo). Per le medie sulle famiglie: la tabella.
- **Se A' non e' pronta quando gira il riepilogo**, le letture escono "PROVVISORIO" con le sole soglie della tabella
  (`em1_provvisorio/`) e il riepilogo si riesegue quando A' (e B') sono pronte (`em1/`, job in coda, sez. 8).

## 3. Riuso (critic, punto 3)

- **Correzione della sez. 8 del protocollo:** il riuso NON e' "uguale per costruzione fra le celle a meno della velocita'
  dei produttori". Dipende anche dalla velocita' del trainer: a produzione uguale un trainer piu' veloce consuma di piu'
  e riusa di piu' (C: s/passo -12% rispetto ad A, riuso circa +20% all'avvio).
- **Criterio simmetrico:** |ln(va / vb)| > ln 1.2 (al posto di "piu' del 20%") su viste fresche/s, riuso a regime e
  **viste uniche per epoca** (nuova: `d_unique_views_used` dei due rank, media dalla seconda epoca), per le coppie B/A,
  D/C, C/A, D/B, B'/A, B'/B, A'/A.
- **Verso della distorsione** accanto a E_F, E_P e B' - A: r = media di ln(u_trattata / u_controllo) sulle coppie
  dell'effetto (E_F: B/A, D/C; E_P: C/A, D/B; B' - A), u = viste uniche per epoca. r > 0: le celle trattate vedono piu'
  viste distinte, distorsione a favore dell'effetto (possibile sovrastima); r < 0: contro (possibile sottostima);
  segnalata se |r| > ln 1.2. Descrittiva, nessuna correzione.

## 4. B, D e B' hanno imparato FLAME? (critic, punto 4)

- Embedding degli insiemi FLAME di D1 anche per B e D (`ablation_2x2_flame_embed.sbatch`, job 1068246 e 1068247, dopo
  le valutazioni 1068214 e 1068216, stesso checkpoint) e per A' e B' (dentro le loro valutazioni, `ABL_FLAME=1`).
- **Pre-regola:** se B - A su `flame2023_s1` con SR (d_P) non e' positivo con IC che esclude 0 (stima > 0 e IC_low > 0),
  **R_F e' "non interpretabile"** (il verdetto si scrive lo stesso, con l'etichetta). Lo stesso per **B' - A e R_F'**.
  D - C si riporta, descrittivo. Avvertenza: B e D hanno visto FLAME nativo e `flame2023_s1` e' suddiviso; l'insieme
  nativo `flame2023` si riporta accanto, descrittivo.

## 5. Formulazioni preregistrate (critic, punto 6)

- Le tre famiglie di test (FaceScape bilineare, AI-NEXT "HIFI3D", FaceVerse) sono 3DMM est-asiatici: "almeno 2 famiglie"
  non e' una replica indipendente.
- **NO** = "nessun beneficio >= soglia rilevato con un seme", non "nessun effetto".
- Un solo generatore (FLAME 2023) non risponde a "un generatore in piu' in generale".

## 6. c al 10% del run (critic, punto 7; descrittivo)

Il controllo di `PLAN_MASSIVE.md` sez. 22.5 (righe 595-602): `calib_ckpt.sbatch` su `epoch010_ema` (6.000 passi) di A-D
(job 1068232-1068235, completati) e di A' e B' (job 1068240 e 1068241, dalle 07:41), in `<cella>/eval/calib_e010`:
c, c_LS, c per dominio e rapporto massimo / minimo fra domini, al 10% e alla fine, accanto all'ordine del C3M (0.225-0.257).
Descrittivo: nessuna regola cambia.

## 7. Codice d'analisi congelato (critic, punto 5)

- **Commit fissato: 7a398c9c32715aaf007463bb37463af81d23438a** (`eval_ablation_em1.py` nella versione provata su checkpoint non dell'ablazione).
  sha256 dei file (anche in `analysis_freeze.json`):

  - `v3_work/stream/eval_ablation.py`: `61010ced9f502ceca2917ad79f0dd4c5ad304a621788f2deebd41d5c077b07c0`
  - `v3_work/stream/eval_ablation_em1.py`: `17a20092e192b54e939b2e3952f6314bcf6767e58543b3a8de483c60ff648baf`
  - `v3_work/trainer/tools/fact_paired.py`: `cb37bf32916237077069157d8c3ac79c2838b6a85fcbebdb639b8419b2fe709a`
  - `aau/baselines_mm/blmm_eval.py`: `7d439122deadd5c1502492c15ca463681984a2ca24fb8e2863d372c7fd12cda8`
  - `aau/diagnostics/diag.py`: `399882d0e81b4f1e7458bb0e8bd77777cf42ad8b8847b716d667a2a11f43af4d`
  - `aau/diagnostics/d1_stats.py`: `8d4b1985fdf2cc680a0d2cd7416ba611bb4e7b49084f3b6bfaf1c349184eb45c`
  - `v3_work/stream/slurm/ablation_2x2_summary_em1.sbatch`: `34db55b33054e09e13c9ac9cce54e4682f4d55fe280a32efc08a0428c7dbfa71`

- `ablation_2x2_summary_em1.sbatch`, fuori dal container, prima di ogni calcolo: per ogni file, sha256 dell'albero di
  lavoro = valore congelato = sha256 del blob al commit fissato, altrimenti esce con errore; commit, HEAD e stato git in
  `code_state.json`. Dentro, `eval_ablation_em1.py` ricontrolla i file e che ogni modulo importato sia quello
  (percorso e sha256) e scrive tutto in `controls.json`. Il PI ha chiesto al coder di D3 di non toccare `diag.py` e
  `d1_stats.py`.

## 8. Catena e rischio operativo (critic, punto 8)

| job | cosa | dipende da |
|---|---|---|
| 1068128 / 1068130 / 1068131 / 1068129 | training A / B / C / D | - |
| 1068228 / 1068239 | training A' / B' | - |
| 1068213 / 1068214 / 1068215 / 1068216 | valutazione A / B / C / D (A100) | il training della cella |
| 1068246 / 1068247 | solo FLAME di B / D (A100) | 1068214 / 1068216 |
| 1068248 / 1068249 | valutazione A' / B' con FLAME (A100) | 1068228 / 1068239 |
| 1068250 | riepilogo con le regole originali (`eval_ablation.py`), al posto di 1068217 (annullato) | valutazioni A-D, FLAME B e D |
| 1068251 | letture dell'emendamento, provvisorie (`em1_provvisorio/`) | valutazioni A-D, FLAME B e D |
| 1068252 | letture dell'emendamento, definitive (`em1/`) | tutto il sopra, A' e B'; dopo 1068251 (afterany) |

**Procedura se un training fallisce o va in TIMEOUT:** i job in `afterok` restano appesi (DependencyNeverSatisfied).
1. Si continua il run con la stessa riga (`CELL=<cella> sbatch --nodelist=<nodo> v3_work/stream/slurm/
   ablation_2x2.sbatch`: stesso `STREAM_OUT`, ripresa da `last.pth` con la copia congelata) fino a 60.000 passi.
2. Si annullano con `scancel` i job dipendenti di quella cella e quelli a valle (valutazione, solo FLAME, riepiloghi) e
   si risottomettono con le stesse righe e il job id nuovo.
Il coder controlla il primo completamento (D, atteso verso le 11:05) e l'avvio della sua valutazione.

## 9. Limiti in piu'

- A' cambia solo il seme del trainer (sez. 2c).
- A' e B' partono 1h45-2h dopo A-D, con tre job per nodo invece di due: piu' contesa, s/passo misurato.
- B' produce le viste FLAME piu' lentamente: piu' riuso di B (a sfavore di B'), misurato e scritto (sez. 3).
- Il controllo FLAME di B usa `flame2023_s1`, suddiviso, mentre B ha visto FLAME nativo (sez. 4).
