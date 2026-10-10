## Conclusioni (10 ottobre 2026, sui numeri delle sezioni sotto)

**1. Riproduzione.** Con la normalizzazione maxabs le righe pubblicate tornano identiche.
- 204 numeri di E12 (punto e IC, GT maxabs e F_rig_rob): scarto massimo 8e-17.
- Pipeline ricalcolata contro le matrici pubblicate: Chamfer bit per bit; ICP entro 1e-15; NICP entro 1.5e-6 (unita' maxabs, solo coppie con up60k; FaceVerse entro 5e-11); template identico; riconoscimento entro 1e-16.
- FR contro F_rig_rob di E12: entro 1.3e-6 (FR e' salvata in float32).

**2. HIFI3D, nocrop_cross, GT FR: con la normalizzazione coerente le baseline geometriche salgono e superano e108 (0.194).**
- ICP + Chamfer: da 0.325 a 0.643 [0.572, 0.703], delta +0.318 [+0.220, +0.411].
- NICP su template: da 0.260 a 0.614. NICP per coppia: da 0.367 a 0.495, delta +0.128 [+0.063, +0.197]. Chamfer: da 0.138 a 0.411.
- ICP in mm - e108: +0.449 [+0.357, +0.539].
- **Taglia.** Oracolo 0.736 [0.658, 0.802]. Stimata dalla mesh osservata (centroid size robusta, NON oracolo): 0.600 [0.498, 0.690], sopra ogni metodo non metrico. Su HIFI3D la maggior parte del segnale di FR e' la taglia: con SR l'oracolo scende a 0.074 e la stimata a 0.023.

**3. GT SR: vincono le baseline del modo cs.**
- HIFI3D: NICP per coppia 0.595 [0.546, 0.648], ICP 0.581; e108 0.279; cs NICP - e108 +0.316 [+0.249, +0.386].
- Il modo mm perde con SR: ICP -0.129 contro maxabs; il template mm scende a 0.083, perche' riportato con la rigida misura la taglia.

**4. FaceVerse e FaceScape: effetti piu' piccoli, coerenti con la poca variazione di taglia (CV 1.8% e 1.6%).**
- FaceVerse, FR: ICP mm 0.337 contro maxabs 0.201 (+0.136 [+0.042, +0.226]); e108 0.184; oracolo della taglia 0.209.
- FaceScape dev, FR: ICP mm 0.467 contro 0.398 (+0.069 [+0.039, +0.093]); NICP su template mm 0.544; e108 0.333; oracolo della taglia 0.464.
- FaceScape, taglia stimata: 0.206. Con taglie cosi' compresse la stima non basta, ed e108 la batte (-0.128 [-0.232, -0.028]).

**5. FaMoS TEST (15 persone, IC larghi).**
- scan gallery -> scan, FR: ICP mm 0.739 [0.362, 0.914], NICP mm 0.719, taglia stimata 0.715, oracolo 0.787.
- Riconoscimento scan peak -> scan, rank-1: ICP mm 0.909 contro maxabs 0.813.

**6. Riconoscimento (nocrop): la normalizzazione cambia poco.**
- HIFI3D ICP 0.996 -> 1.000 (gia' saturo); FaceVerse ICP 0.918 -> 0.958; FaceScape con espressioni 0.419 -> 0.444.
- La sola taglia stimata identifica poco: rank-1 0.13 su HIFI3D, con AUC 0.91.

**Avvertenze.**
- **Chamfer non centrata: solo diagnostica, non una baseline.** Sulla posizione assoluta nel frame del 3DMM fa rank-1 1.000 su HIFI3D e FaceScape neutra: le topologie di un soggetto condividono quella posizione, che sulle scansioni reali non si osserva. La Chamfer "in mm" di riferimento e' quella centrata: faceBench senza la scala.
- **NICP di faceBench fallisce ("Factor is exactly singular") in due casi, uguale nei tre modi.**
  - Sorgente up60k: su HIFI3D 199 coppie per coppia di topologie (776 NaN in nocrop, come le righe pubblicate); su FaceScape circa 10.275 NaN su 99.000.
  - Sorgente una patch di registrazione FaMoS: falliscono tutte.
  - Le coppie fallite sono escluse dagli Spearman e contano +inf nel riconoscimento; i blocchi FaMoS con una reg come sorgente sono omessi.
- **NICP non e' invariante alla scala.** Lavora in unita' L_d, una costante per dominio (il max|coordinata| mediano); le distanze si moltiplicano poi per L_d.
- **Semi.** `subject_pair_mean_nocrop` usa il seme 1234 dei bracci (in E12 quel gruppo non c'e'); gli altri gruppi usano i semi di E12.
