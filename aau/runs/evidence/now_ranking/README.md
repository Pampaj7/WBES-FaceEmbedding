# NoW ranking: dati, licenze e cosa c'e' qui

Regole in `REGOLE.md` (scritte prima dei numeri), risultati in `results.md`. Codice in `aau/recon/`: `tddfa_v3_run.py`,
`hrn_run.py`, `now_factorized.py`, `now_ranking.py`, `now_ranking_prep.sbatch`, `now_ranking_embed.sbatch`, e le
opzioni `--metrical` di `now_export.py` / `now_official.py`, `--metric` e `--recon-only` di `now_prepare_meshes.py`.

## Licenze e cosa NON e' nel repo

- **NoW** (validation: immagini, scansioni, landmark): licenza di ricerca non commerciale, nessuna redistribuzione. In
  `~/data/now`; ricostruzioni, patch, operatori, embedding ed export in `~/data/now_recon`, `~/data/now_eval_work`,
  `~/data/now_ranking_work`. Qui solo numeri per immagine e aggregati.
- **Basi derivate da BFM09** (riserva accettata dal PI il 14 ottobre, solo per valutazione di ricerca). 3DDFA_V2
  (`bfm_noneck_v3`, nel clone), 3DDFA-V3 (`face_model.npy`, Hugging Face `Zidu-Wang/3DDFA-V3`) e HRN
  (`BFM_model_front.mat`, cartella Google Drive del README di HRN) ridistribuiscono basi derivate dal Basel Face Model
  2009, la cui licenza richiede un modulo firmato con l'Universita' di Basilea. Noi non abbiamo firmato nulla:
  usiamo i pesi di inferenza e gli asset pubblicati dagli autori dei metodi, solo per questa valutazione. **Nessuna
  mesh ricostruita e nessuna base o asset entra in git** (`external/` e' ignorato); nessuna redistribuzione.
- **Codice dei metodi**: 3DDFA-V3 MIT (clone `e15385837`), HRN Apache-2.0 (clone `58833698`), 3DDFA_V2 MIT,
  SynergyNet MIT, PRNet MIT; cloni in `external/`, non modificati.
- **Esclusi**: MICA, DECA, EMOCA, SMIRK (FLAME 2020), Pixel3DMM (FLAME 2023 non Open), Deep3DFaceRecon (BFM09 solo
  col modulo). MICA e' stato ritirato anche dai risultati del 7 ottobre: `aau/runs/now_eval/NOTA_MICA.md`.
