## Conclusioni (9 ottobre 2026, sui numeri delle sezioni sotto)

**1. Il json e' per dominio, ma la sua scala non e' solo un cambio di unita'.**
- Ricalcolato dal template medio di ogni dominio: scarto massimo 1e-15.
- `s = u * k`, con `k` fra 0.965 (ICT) e 1.037 (BFM): GT-F usa solo `u`.
- **Unita' confermate dall'IPD** (60.8-67.9 mm): FLAME m, BFM µm, ICT cm, GNM m, FaceScape mm, Multiface mm.
- **Unita' ignote:** HIFI3D 9.34 mm per unita' (vicino al cm, -6.6%); FaceVerse 161.8 mm per unita' (arbitraria).

**2. Dimensione del volto.**
- **CV plausibile** (4-7%): HIFI3D (centroid size 4.9%, IPD 5.6%), FLAME, ICT, GNM, Multiface, FaMoS (4.0%).
- **CV troppo basso:** FaceVerse 1.8%, FaceScape 1.6%, BFM REMESH 1.9%. Questi modelli, o questi dati, comprimono la taglia.

**3. Su HIFI3D la GT-F pura misura soprattutto dove sta il volto nel frame del modello.**
- Il centroide della regione si sposta da un'identita' all'altra di 5.5 mm (mediana; p95 15.9 mm). In FaceVerse e FaceScape lo spostamento e' di 0.8-1.0 mm.
- Le distanze F hanno mediana 10.1 mm, che scende a 5.4 dopo una rigida.
- **F contro maxabs:** 0.165 [0.069, 0.262]. **F contro unificata:** 0.167. **F contro "solo taglia":** 0.734.
- **F-rig-LS riproduce i numeri del critic:** 0.408 contro maxabs e 0.564 contro l'unificata sul pool (critic: 0.41 e 0.57). La sua "GT in mm grezzi" era quindi allineata rigidamente.
- **S, come dichiarata (scala attorno a un punto fisso), non toglie la traslazione:** su HIFI3D S contro F vale 0.990. E' un limite della definizione; non l'ho cambiata dopo i numeri.

**4. Arbitro (FaMoS, 95 persone, 2639 primi fotogrammi). La regola sceglie F; con la regola stretta l'esito e' lo stesso.**
- **AUC di F:** 0.9928 [0.9891, 0.9957]. F supera S (+0.0027 [+0.0013, +0.0043]), EDM (+0.0015 [+0.0002, +0.0027]) ed EDM-s (+0.0079 [+0.0051, +0.0110]).
- **Riserve:**
  - le differenze sono piccole, vicino al tetto dell'AUC;
  - fuori concorso, la rigida LS fa meglio della robusta (+0.0019 [+0.0009, +0.0031]) e la maxabs fa meglio di F (+0.0020), ma usa una regione piu' grande (maschera `face` di FLAME);
  - su dati reali F contiene la rigida per identita'. L'arbitro convalida "forma metrica dopo una rigida", non la posizione nel frame del generatore, che sulle scansioni non si osserva.
- **Un solo scalare identifica bene:** "solo taglia" ha AUC 0.939.
- **Multiface** satura (AUC 1.0): non ha riprese neutre ripetute.

**5. Metodi, HIFI3D nocrop_cross (primario).**
- **Con F pura** tutti i metodi stanno fra 0.005 e 0.119:
  - e108 0.068 [-0.004, 0.150], Chamfer eval 0.060;
  - fra e108, Chamfer e ICP/NICP nessuna differenza significativa (e108 - Chamfer eval +0.009 [-0.038, +0.065]; NICP - e108 +0.051 [-0.023, +0.121]).
- **Con F + rigida robusta:**
  - NICP P2Tri 0.367, ICP + Chamfer 0.325, NICP su template 0.260, e108 0.194, Chamfer eval 0.155;
  - NICP - e108 +0.174 [+0.092, +0.253];
  - e108 - Chamfer eval +0.039 [-0.021, +0.098].
- **I modelli (e036, e072, e108) sono in testa solo con la maxabs** (e036 0.677, e108 0.630). Batte ancora Chamfer eval con l'unificata (+0.071 [+0.015, +0.123]) e con EDM-s (+0.055 [+0.001, +0.108]); con F, S, EDM e le rigide no.
- **FaceScape dev:** e108 batte Chamfer eval con ogni GT (con F +0.119 [+0.076, +0.159]), ma non la Chamfer intera (-0.006).
- **FaceVerse:** e108 - Chamfer eval non e' significativo con nessuna GT (con F -0.017 [-0.073, +0.041]).

**Deviazioni:**
- Il nodo `cpu` era in drain: ho usato `prioritized` senza GPU.
- La maxabs di FaMoS, la scala di S e la scelta di tutte le persone FaMoS sono dichiarate nell'Emendamento 1.
