# Competitori diretti per la verità geometrica (8 ottobre 2026)

Ricerca svolta da un agente Sonnet. Esistenza dei repository verificata via API GitHub/HF; nessun codice eseguito.

## Da integrare, in ordine
1. **Uni3D** (ICLR'24): `github.com/baaivision/Uni3D`, MIT, pesi su HF `BAAI/Uni3D`. Point cloud 10k, zero-shot.
2. **OpenShape** (NeurIPS'23): `github.com/Colin97/OpenShape_code`, Apache-2.0, pesi su HF `OpenShape/openshape-pointbert-vitg14-rgb` (OpenRAIL). Point cloud [B,6,10000].
3. **ShapeDNA / spettro di Laplace-Beltrami** (LaPy, MIT) e **HKS/WKS globali** (pyFM, MIT): mesh native, nessun peso. È il competitor più vicino al nostro asse.
4. **FLAME fitting** (`Rubikplayer/flame-fitting`): solo se arriva la licenza FLAME. Richiede landmark 3D; costo medio-alto.
5. **Cui et al.** (PointNet++ su BFM sintetico, `alfredtorres/3DFacePointCloudNet`): pesi non trovati, andrebbe riaddestrata.
6. **Point-MAE / Point-BERT**: secondari.

## Esclusi o non disponibili
- **PointFace, PointFaceFormer:** repository non trovati.
- **MICA:** lavora su immagini.
- **Led3D:** usa mappe di profondità 2.5D.
- **ULIP-2:** licenza dei pesi non chiara.
- **Metriche percettive apprese per mesh:** nessuna con codice trovata; da cercare in letteratura prima di dichiarare la categoria vuota.
