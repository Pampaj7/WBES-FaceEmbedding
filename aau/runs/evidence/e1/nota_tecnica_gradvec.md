# E1, nota tecnica: build_grad vettorizzato nel pre-pass (8 ottobre 2026, prima di ogni numero delle celle nuove)

Si aggiunge a `protocol.md` e a `protocol_amendment.md`, che restano invariati. Decisione del PI, presa dopo il
controllo di uguaglianza; nessun training di cella era ancora partito.

- **Cosa:** in TUTTE le celle (training su V100) il pre-pass degli operatori usa il `build_grad` vettorizzato di
  E9 al posto di `diffusion_net.geometry.build_grad`.
  - La copia e' congelata in `aau/evidence/e1_factorial/gradvec_site/e1_grad_vec.py`; sha256 della sorgente E9
    alla copia: ecdb6756..
  - E' agganciata da `gradvec_site/sitecustomize.py` con `WBES_E1_GRADVEC=1`; diffusion-net, v2_work e
    aau/data_scale restano invariati.
  - Le valutazioni (A100) usano tutte il `build_grad` originale, compresa C3M L40S.
- **Differenze misurate** (`gradvec_check.json`; 57 mesh, tutte le topologie e le espressioni, 3 ICT nuove e 2
  GNM, nodo V100):
  - uguali tutti gli array tranne `gradX_values` e `gradY_values`;
  - 10 file e 19 array diversi, il 6.3e-5 degli elementi;
  - |differenza| massima 1.2e-10, relativa al massimo 3.7e-13. Sono voci vicine a zero; la causa e' l'ordine
    di somma.
  - Molto sotto il rumore numerico del training.
- **Guadagno:** pre-pass da 4.61 a 2.81 CPU-s per mesh, cioe' 1.64x.
- **Confondenti:** nessuno fra le celle, perche' tutte usano la stessa versione. C3M L40S (1060130, `build_grad`
  originale) e' solo un riferimento: nel confronto col rumore (C3M V100 - C3M L40S) entra anche questa
  differenza, oltre all'hardware.
- **Lo smoke V100** (1061840) gira con il `build_grad` originale. Confronta solo la loss dell'epoca 1 con L40S:
  gli operatori coincidono a meno di 1.2e-10, quindi il confronto non ne risente.
