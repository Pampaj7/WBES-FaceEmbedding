# Protocolli di valutazione 3D del volto: che allineamento fanno fra predetta e GT

Ricognizione del 12 ottobre 2026. Nulla di questo codice e' stato eseguito: solo letto.
Cloni (depth 1) in (cloni locali non versionati)
(now_evaluation, REALY, MICA, facescape, h3ds, DECA, emoca, smirk, fg2018). Testo dei paper estratto da PDF arXiv in `.../protocolli/papers/`.
Rete del cluster: GitHub e arXiv raggiungibili. DBLP non ha risposto (bibtex presi dai README dei repo).

Legenda fonte: [CODICE] = letto nel sorgente ufficiale; [PAPER] = letto solo nel paper; [API] = dedotto dalla documentazione/sorgente di una libreria (trimesh), non dal repo del benchmark.

## Tabella

| Benchmark | Allineamento | Scala stimata per coppia? | Non rigido? | Regione GT | Distanza finale / unita' |
|---|---|---|---|---|---|
| NoW originale (non-metrical) | Procrustes su 7 landmark (similarita', con scala) come init, poi ICP scan-to-mesh con scala libera `s`. Per coppia (pred -> scan). [CODICE] | SI (due volte: Procrustes e ICP) | NO | Cerchio sulla scansione: centro = naso-basso + 0.3*(ponte-naso-basso), raggio = 0.7*(dist. occhi esterni + dist. naso) [CODICE+PAPER] | scan-to-mesh (distanza di ogni vertice scan al punto piu' vicino sulla superficie predetta), mm; mediana/media/std [CODICE] |
| NoW "metrical" (MICA, nel repo now_evaluation) | Procrustes a 7 landmark calcolato con scala ma la scala NON viene applicata (si usa il fattore d'unita' m/cm/mm -> mm); ICP solo [R,t]. [CODICE] | NO | NO | identica a NoW | identica, mm [CODICE] |
| REALY | gICP: similarita' su 7 keypoint (scala stimata) -> poi per ognuna delle 4 regioni (naso, bocca, fronte, guancia) un ICP rigido separato (no scala) con pesi sui keypoint -> poi nICP non rigido che deforma la regione della SCAN verso la predetta per trovare le corrispondenze. [CODICE] | SI (solo nel passo globale gICP, sui 7 keypoint) | SI, ma sul lato GT e solo per costruire le corrispondenze (l'errore e' misurato contro i vertici originali della scan) | 4 regioni pre-ritagliate fornite dal benchmark (`REALY_scan_region`), scansioni gia' riscalate/riallineate a un template | media delle distanze fra punti corrispondenti ("NMSE", in realta' media delle norme), moltiplicata per `metrical_scale.txt` per tornare in mm [CODICE] |
| FaceScape (wild e lab) | Nessun landmark/ICP/Procrustes. La predetta e' proiettata nell'immagine con la camera del metodo (nessuna rotazione fittata). wild: solo traslazione in Z ottimizzata su profondita' (L1). lab: scala uniforme attorno al centro camera `mean_z/(mean_z-delta_z)` per matchare la profondita' GT. [CODICE; il paper dice "solo spostamento in profondita'"] | wild: NO. lab: SI, 1 grado di liberta' fittato sulla profondita' GT (equivalente a una scala/profondita') | NO | Proiezione cilindrica di GT e pred, intersezione delle due maschere (Completeness Rate riportato a parte) | Chamfer bidirezionale (media di A->B e B->A) fra le mesh cilindriche, unita' ~mm (GT normalizzata a distanza pupille standard 62.85 mm), piu' MNE e CR [CODICE] |
| Stirling/ESRC (Feng et al. 2018, FG) | Procrustes su 7 landmark con scala, nessun ICP. Per coppia. [CODICE fg2018 + PAPER] | SI | NO | Cerchio come NoW ma raggio = 1.2*(occhi esterni + naso)/2, ~80 mm [CODICE+PAPER] | distanza vertice-GT -> superficie predetta (pymesh), RMSE in mm [CODICE+PAPER] |
| H3DS | Opzionale: Procrustes sui 6 landmark (`reflection=False`) -> ICP trimesh sempre per coppia. Il paper dice "rigid ICP" ma il codice chiama `trimesh.registration.procrustes/icp` senza `scale=False`, e il default di trimesh e' `scale=True` [CODICE + API] | SI (per default trimesh, in contraddizione col paper) | NO | `face_sphere`: sfera di 95 mm centrata sulla punta del naso GT (ICP e chamfer solo li'); `face` (volto+orecchie+collo) per l'ICP della testa intera | Chamfer unidirezionale GT -> pred con KD-tree sui VERTICI della predetta (non sulla superficie), mm [CODICE] |
| DECA | Nessun codice di valutazione 3D proprio: esporta mesh + 7 landmark e rimanda a now_evaluation (commento in `decalib/utils/trainer.py:245`). Quindi NoW non-metrical. Il paper dice solo "rigidly aligning". Anche il benchmark di Feng: "protocollo simile a NoW" [CODICE-rimando + PAPER] | SI (ereditato da NoW originale) | NO | come NoW / Feng | come NoW / Feng |
| EMOCA | `now_benchmark.py` chiama `metric_computation` di now_evaluation (linea 41 e 264): NoW non-metrical. Il paper non riporta numeri NoW in modo che io abbia trovato; dice solo che i benchmark d'identita' esistono [30,75]. [CODICE] | SI (ereditato) | NO | come NoW | come NoW |
| SMIRK | Nel repo nessun codice di valutazione 3D. Il paper (appendice B, Tab. 6) riporta errori per-vertice su MultiFace con "ScanToMesh [73]" (= Sanyal et al. 2019) ma NON specifica l'allineamento. Nel testo principale dice che gli errori geometrici sono dominati dalla forma d'identita' e poco informativi. [PAPER] | NON SPECIFICATO | NON SPECIFICATO | MultiFace, set test EXP ROM07, 5 soggetti, 6324 espressioni | media/mediana/max ScanToMesh, mm |

## Giudizio in una riga per benchmark: "allinea per coppia con scala/non rigido"

- NoW originale: **SI scala** (Procrustes + ICP con `s` libero), no non rigido.
- NoW metrical (MICA): **NO** (solo rigido; scala fissata dall'unita').
- REALY: **SI scala** (gICP) **e SI non rigido** (nICP) ma sul lato GT e solo per le corrispondenze; in piu' 4 ICP rigidi indipendenti per regione (allineamento a tratti).
- FaceScape: **quasi NO** (wild: solo Z; lab: 1 DoF di scala/profondita' attorno alla camera fittato sulla profondita' GT).
- Stirling/ESRC: **SI scala** (Procrustes a 7 landmark), no non rigido, no ICP.
- H3DS: **SI scala** per default di trimesh (il paper dice "rigid"), no non rigido.
- DECA / EMOCA: **SI scala**, perche' usano NoW originale.
- SMIRK: **non determinabile** (nessun codice, nessuna specifica nel paper).

## Dettaglio con file:riga (tutto [CODICE] salvo dove indicato)

Percorso base: `.../scratchpad/protocolli/`

### NoW (`now_evaluation/`)
- `scan2mesh_computations.py:126` Procrustes a 7 landmark, `scaling=True, reflection='best'`.
- `scan2mesh_computations.py:129` applica `tform['scale']*(R.T v) + t` a tutta la mesh.
- `scan2mesh_computations.py:29` e `:53` ICP scan-to-mesh con `s = ch.ones(1)` variabile; `:53` `fun={'dist': s2m, 's_reg': 100*(ch.abs(s)-s)}` (il termine impedisce solo scala negativa); `:132-133` applica `s*(R v)+t`.
- `scan2mesh_computations.py:56-72` maschera: centro `:66`, raggio `:71` `1.4*(outer_eye_dist + nose_dist)/2` (= 0.7 x (...), coerente col paper; la riga 70 con 1.2 e' commentata).
- `scan2mesh_computations.py:74-93` ritaglio; `:169-171` errore = `ScanToMesh(..., signed=False, normalize=False).r` sui vertici della scan ritagliata.
- Variante metrical `scan2mesh_computations_metrical.py`: `:117` `scale_factor = get_unit_factor(...)` (m->1000 ecc., le scansioni NoW sono in mm); `:122` Procrustes ancora con `scaling=True` ma `:125` usa `scale_factor` al posto di `tform['scale']`; `:53` ICP `x0=[r, t]` senza `s`.
- Osservazione mia (dedotta dalla lettura, non verificata eseguendo): in `:125` la traslazione `tform['translation']` e' calcolata con la scala stimata `b` ma viene usata con il fattore d'unita'; l'ICP rigido successivo (`:53`) corregge la traslazione, quindi non dovrebbe influire sul risultato finale, ma l'inizializzazione e' leggermente inconsistente.
- Selezione: `compute_error.py:114-124` ramo `metrical_eval`; flag CLI `--metrical_evaluation` `compute_error.py:455`; `--predicted_mesh_unit` `:461`. Default NON metrical.
- Mediana/media: `compute_error.py` ~`:391-398` (scrive `.meanmedian`).
- Paper [PAPER] (RingNet/NoW, Sez. 4): "rigidly aligned (rotation, translation, and scaling) ... using a set of corresponding landmarks ... We further perform a rigid alignment based on the scan-to-mesh distance"; raggio "0.7 x (outer eye dist + nose dist)". Si noti che il paper usa la parola "rigid" includendo la scala.
- README di now_evaluation, linea 15: stessa dicitura "rigidly aligned (rotation, translation, and scaling)".

### MICA (`MICA/`) e NoW metrical [PAPER + CODICE]
- Il repo MICA non contiene il codice di valutazione: `README.md:61` rimanda a now_evaluation. `testing/now/template.sh:34` lancia `python compute_error.py ${NOW} ${PREDICTED} true`; `micalib/tester.py:166-167` salva le mesh x1000 (m -> mm); `:271` landmark x1000.
- Per Stirling il template usa uno script non pubblicato `feng_error.py` (commentato, `testing/stirling/template.sh:~40`): non verificabile dal repo.
- Paper (Sez. 1 e Sez. 6.1) [PAPER]: "aligned to the scan via Procrustes analysis, including an optimal scaling factor. This scaling factor favors the estimation methods that are not metrical"; "the mean error evaluated on the NoW Challenge for the average FLAME mesh (Table 2) drops from 1.92mm to 1.53mm because of the applied scale optimization. This is an improvement of around 20% which has nothing to do with the reconstruction quality"; protocollo metrical: "purely rigid alignment, i.e., without scale optimization"; errore = distanza euclidea vertice-scan -> punto piu' vicino sulla superficie. Sul benchmark di Feng (Tab. 4): "alignment protocol that only relies on reference landmarks and, thus, is very noisy".
- **Attenzione per la novita' del vostro paper**: l'argomento "la scala per coppia nasconde errore d'identita'" e' gia' fatto da MICA (con numero: 1.92 -> 1.53 mm sulla faccia media FLAME). Non e' coperto in MICA (ne' altrove, per quanto letto) il caso del non rigido per coppia ne' un test dell'effetto sulla separabilita' d'identita'.

### REALY (`REALY/`)
- `utils/gICP.py:47` `fit_icp_RT(..., with_scale=True)` su 7 keypoint (`utils/util.py:51` indici `[36,39,42,45,33,48,54]`); `:49` applica `vertices*scale` poi `R.T, t`.
- `utils/util.py:121` `scale = sum|t1| / sum|R s1|` se `with_scale` (scala stimata come rapporto di norme L1, non Umeyama).
- `utils/rICP.py:82` `fit_icp_RT(..., with_scale=False)`: ICP regionale rigido; `:77` corrispondenze = punto piu' vicino sulla SUPERFICIE predetta per ogni vertice della regione scan; keypoint ripetuti come pesi `:65-73`; `max_iteration=100` `:39`; chiamato per 4 regioni `:151-167` -> quattro allineamenti rigidi diversi per la stessa predetta.
- `utils/eval.py:78-103` nICP (`nICP_with_keypoints` / `nICP_without_keypoints` per la guancia) che deforma la regione della SCAN verso le corrispondenze sulla predetta; `:110` nuove corrispondenze; `:114-` errore = media della norma `||corr_aggiornata_su_pred - vertice_scan_ORIGINALE||`. Dedotto da me: poiche' e' la distanza fra punti corrispondenti scelti via scan deformata e non il punto piu' vicino, e' >= della distanza scan-to-mesh.
- `main.py:259` `error * metrical_scale[subject_id - 1]`; `main.py:114` carica `data/metrical_scale.txt` (valori ~6.5-9.9 per soggetto): riporta in mm; `main.py:155-186` pipeline gICP -> rICP; `main.py:230-262` bICP + errore per regione.
- Paper [PAPER]: abstract "typically relies on a rigid shape alignment ... aligning two shapes with different reference points can largely affect the evaluation results"; Sez. 3/4: processo standard = "(1) first rescale and align ... based on some sparse keypoints or applying ICP"; "rICP (region-aware ICP)" e "bICP (non-rigid and bidirectional ICP)"; "NMSE error is then transformed back to the physical scale of the raw scan in millimeters (each ground-truth in our benchmark is rigidly transformed from raw scan)".
- Nota: REALY e' il paper che mostra che il riferimento d'allineamento cambia i risultati (qualitativamente vicino al vostro argomento ma sul lato rigido/regionale, non sull'identita').

### FaceScape (`facescape/benchmark/code/`)
- `evaluator.py:36` `std_pupil_dist = (64.0+61.7)/2`; wild `:316-320` `gt_world_mesh.vertices *= pupil_scale; pred_align_mesh.vertices *= pupil_scale` (stesso fattore, derivato dalla GT, applicato a entrambe -> le dimensioni relative non cambiano).
- `evaluator.py:96-131` `optimize_Z` (SLSQP su L1 della differenza di profondita', un solo scalare); wild `:329` `pred.vertices[:,2] -= delta_z/ortho_scale*pupil_scale` (traslazione pura).
- lab `evaluator.py:424` `pred *= std_pupil_dist`, `:434-435` `pred.vertices *= mean_z/(mean_z-delta_z)`: scala uniforme attorno all'origine della camera. Dedotto da me: preserva la proiezione sull'immagine e cambia la dimensione 3D, quindi e' un grado di liberta' di scala/profondita' fittato sulla GT, per coppia.
- Allineamento X,Y e pose: nei loader `pred_loader.py` con la camera predetta dal metodo; es. DECA `pred_loader.py:137-168` (usa `cam` e `tform` del metodo; nessuna rotazione fittata). Nel caso prospettico (lab) i loader normalizzano anche per l'estensione in y della pred (`pred_loader.py:127-131`, `:176-178`), poi sovrascritto dal fit di profondita'.
- `mesh_util.py:23-39` `compute_chamfer`: media su concatenazione di A->B e B->A, con distanza punto-superficie (`aabb_tree.nearest`).
- `evaluator.py:341` Completeness Rate (frazione di raggi che colpiscono entrambe); `:356` MNE.
- Paper [PAPER] Sez. 6.2: "we do not use ICP or 7-point alignment to register the predicted mesh to the ground-truth mesh ... but only align them by moving the predicted mesh along the depth direction. We believe our alignment tells more because ICP and 7-point alignment changes the pose of the predicted mesh". Il README del benchmark dice anche che, a differenza di NoW, include pose/espressioni variabili. Conferma esplicita che gli autori considerano ICP/7 punti un'alterazione della predizione.

### Stirling/ESRC (`fg2018/` = patrikhuber/fg2018-competition, e `papers/feng.txt`)
- `compute_vertices_to_mesh_distances.py:25-28` Procrustes a 7 punti, `scaling=True, reflection='best'`, commento "The ground truth scan is in mm, so by aligning the prediction to the ground truth, we get meaningful units"; applicazione `:30-36`.
- Maschera `:40-53`, raggio `:46` `1.2 * (outer_eye_dist + nose_dist) / 2`; distanze vertice GT -> superficie predetta (pymesh) `:56-60`.
- Paper [PAPER]: "rigidly aligned (by translation, rotation, and scaling). Scaling is compensated for because participants' resulting meshes might be in a different coordinate system, whereas the ground truth scans are in the unit of millimetres"; "Typically, the resulting radius is around 80 mm"; RMSE in mm.
- Il codice di MICA per Stirling (`feng_error.py`) non e' nel repo MICA.

### H3DS (`h3ds/`)
- `h3ds/dataset.py:366-418` `evaluate_scene`; `:402-403` `perform_alignment` (coarse, solo se si passano i landmark); `:406-407` `perform_icp(mesh_gt, mesh_pred, region_gt)` e applica l'inversa alla pred; `:410-411` ritaglio `mesh_gt.cut(region_gt)` se c'e' `region_id`; `:413-416` chamfer.
- `h3ds/numeric.py:44-46` `trimesh.registration.procrustes(points_s, points_t, reflection=False)`; `:61-62` `trimesh.registration.icp(points_source, points_target, **icp_args)`; nessuno dei due passa `scale=False`.
- [API] In trimesh (letto sia tag 3.9.0 sia main): `procrustes(..., scale=True, ...)` di default e `icp(..., **kwargs)` inoltra i kwargs a `procrustes` (`registration.py`); `h3ds/requirements.txt` fissa `trimesh~=4.6.13`. Quindi, per default, coarse e ICP stimano scala. Non verificato eseguendo.
- `h3ds/numeric.py:74-79` `unidirectional_chamfer_distance` = KD-tree sui vertici di `target` (distanza vertice-vertice, non punto-superficie); `dataset.py:413` usato come GT -> pred.
- `dataset.py:338-345` regioni: `face` (volto, orecchie, collo), `face_sphere` (sfera 95 mm centrata sulla punta del naso), `nose`. `examples/evaluate.py:123-130` commento: nella `face_sphere` "a more fine alignment is performed".
- Unita': mm (`examples/evaluate.py:109,134`).
- Paper H3D-Net [PAPER] Sez. sperimentale: "roughly aligned ... using manually annotated landmarks, and then refined with rigid ICP [4]. Then, we compute the unidirectional Chamfer distance from the predicted reconstruction to the ground truth. All the distances are computed in millimeters"; sfera 95 mm come in [28,47]. Il README del repo avverte che i numeri del paper differiscono dal codice per "minor implementation changes in the alignment process and in the cutting of the regions".
- Rilevanza: H3DS e' ricostruzione multi-vista/few-shot di testa, non single-image su FLAME; utile solo come esempio a parte.

### DECA / EMOCA / SMIRK
- DECA: `DECA/decalib/utils/trainer.py:245` (e `DECA/decalib/trainer.py:358`): "then please run main.py in https://github.com/soubhiksanyal/now_evaluation". Esporta anche 7 landmark (`trainer.py:~241`). Paper [PAPER] Sez. 6 (NoW e Feng et al. benchmark): "after rigidly aligning scans and reconstructions" e "A protocol similar to NoW is used" per Feng; nessun cenno alla scala nel paper, che invece e' presente nel codice NoW. Nel paper la tabella Feng/NoW e' dunque non-metrical.
- EMOCA: `emoca/gdl_apps/EMOCA/evaluation/now_benchmark.py:38-41` aggiunge `now_evaluation` al path e importa `from main import metric_computation`; `:264` lo chiama; `:183` seleziona 7 landmark dai 51 (`landmark_51[[19,22,25,28,16,31,37]]`). `main.py` e' la versione precedente di now_evaluation (HEAD attuale ha `compute_error.py` e NON ha `main.py`): lo stesso algoritmo e' in `scan2mesh_computations.py`, ma la versione esatta usata da EMOCA non e' verificata. Il paper EMOCA [PAPER] rimanda a benchmark d'identita' [30,75] e dice che l'espressione non ha un benchmark equivalente.
- SMIRK: nessun file di valutazione 3D (`smirk/` contiene solo demo, train, src, utils; l'unico riferimento a MultiFace e' `src/renderer/renderer.py:254`). Paper [PAPER], Sez. 4.2 e Appendice B Tab. 6: errori per-vertice (mm) su MultiFace (DECA 1.40, EMOCAv1 1.45, EMOCAv2 1.43, SMIRK 1.28 medio) "ScanToMesh [73]" ma senza descrivere l'allineamento. Sez. 4.2 afferma che "geometric errors tend to be dominated by the identity face shape".

## Bibtex

Presi dai README dei repo ufficiali, salvo dove indicato. Non ho potuto interrogare DBLP, quindi pagine/volume non confermati dove assenti.

```bibtex
% NoW (README di now_evaluation, testuale)
@inproceedings{RingNet:CVPR:2019,
  title = {Learning to Regress 3D Face Shape and Expression from an Image without 3D Supervision},
  author = {Sanyal, Soubhik and Bolkart, Timo and Feng, Haiwen and Black, Michael},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  month = jun,
  year = {2019}
}

% REALY (README di REALY, testuale; arXiv 2203.09729)
@inproceedings{REALY,
  title = {REALY: Rethinking the Evaluation of 3D Face Reconstruction},
  author = {Chai, Zenghao and Zhang, Haoxian and Ren, Jing and Kang, Di and Xu, Zhengzhuo and Zhe, Xuefei and Yuan, Chun and Bao, Linchao},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2022}
}

% MICA. Il README del repo usa @proceedings con "journal", che e' malformato; sotto la versione corretta
% (stessi campi, tipo e booktitle sistemati; arXiv 2204.06607).
@inproceedings{zielonka22mica,
  author = {Zielonka, Wojciech and Bolkart, Timo and Thies, Justus},
  title = {Towards Metrical Reconstruction of Human Faces},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2022}
}

% FaceScape benchmark (README di facescape, testuale; arXiv 2111.01082)
@article{zhu2023facescape,
  title = {FaceScape: 3D Facial Dataset and Benchmark for Single-View 3D Face Reconstruction},
  author = {Zhu, Hao and Yang, Haotian and Guo, Longwei and Zhang, Yidi and Wang, Yanru and Huang, Mingkai and Wu, Menghua and Shen, Qiu and Yang, Ruigang and Cao, Xun},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year = {2023}
}
@inproceedings{yang2020facescape,
  author = {Yang, Haotian and Zhu, Hao and Wang, Yanru and Huang, Mingkai and Shen, Qiu and Yang, Ruigang and Cao, Xun},
  title = {FaceScape: A Large-Scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020},
  pages = {601--610}
}

% Stirling/ESRC. Nessun bibtex ufficiale trovato (il repo fg2018 non ne ha). COMPOSTO DA ME dai metadati arXiv 1803.05536
% (titolo, autori verificati) e dalla dicitura del PDF ("2018 13th IEEE Conference on Automatic Face & Gesture Recognition",
% "extended version"). Pagine/DOI NON verificati: controllare prima di usare.
@inproceedings{feng2018evaluation,
  title = {Evaluation of Dense 3D Reconstruction from 2D Face Images in the Wild},
  author = {Feng, Zhen-Hua and Huber, Patrik and Kittler, Josef and Hancock, Peter J. B. and Wu, Xiao-Jun and Zhao, Qijun and Koppen, Paul and R{\"a}tsch, Matthias},
  booktitle = {IEEE International Conference on Automatic Face \& Gesture Recognition (FG)},
  year = {2018},
  note = {arXiv:1803.05536}
}

% H3DS / H3D-Net (README di h3ds, testuale)
@inproceedings{ramon2021h3d,
  title = {H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction},
  author = {Ramon, Eduard and Triginer, Gil and Escur, Janna and Pumarola, Albert and Garcia, Jaime and Giro-i-Nieto, Xavier and Moreno-Noguer, Francesc},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages = {5620--5629},
  year = {2021}
}

% DECA. Il README del repo usa @inproceedings con "journal" e "number = 8"; arXiv riporta come journal_ref
% "ACM Transactions on Graphics (ToG), Vol. 40, No. 4, Article 88, August 2021". Versione corretta (DOI dall'URL del README):
@article{DECA:Siggraph2021,
  title = {Learning an Animatable Detailed {3D} Face Model from In-The-Wild Images},
  author = {Feng, Yao and Feng, Haiwen and Black, Michael J. and Bolkart, Timo},
  journal = {ACM Transactions on Graphics (Proc. SIGGRAPH)},
  volume = {40},
  number = {4},
  articleno = {88},
  year = {2021},
  doi = {10.1145/3450626.3459936}
}

% EMOCA (README di emoca, testuale)
@inproceedings{EMOCA:CVPR:2021,
  title = {{EMOCA}: {E}motion Driven Monocular Face Capture and Animation},
  author = {Danecek, Radek and Black, Michael J. and Bolkart, Timo},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {20311--20322},
  year = {2022}
}

% SMIRK (readme di smirk, testuale; il titolo del README omette "SMIRK:", quello arXiv e' "SMIRK: 3D Facial Expressions through Analysis-by-Neural-Synthesis")
@inproceedings{SMIRK:CVPR:2024,
  title = {3D Facial Expressions through Analysis-by-Neural-Synthesis},
  author = {Retsinas, George and Filntisis, Panagiotis P. and Danecek, Radek and Abrevaya, Victoria F. and Roussos, Anastasios and Bolkart, Timo and Maragos, Petros},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024}
}
```

## Cosa NON ho verificato
- Nessun codice eseguito. Le affermazioni su default di trimesh sono dalla lettura del sorgente trimesh (3.9.0 e main), non dall'esecuzione in H3DS.
- Il codice di MICA per Stirling (`feng_error.py`) non e' pubblico; il comportamento di MICA su Stirling e' solo dal paper.
- La versione di `main.py` di now_evaluation usata da EMOCA non e' nel clone HEAD (che e' stato rifattorizzato in `compute_error.py`); assumo lo stesso algoritmo.
- Per SMIRK/MultiFace l'allineamento non e' determinabile da repo ne' da paper.
- Le citazioni bibtex di Feng et al. 2018 sono composte da me; vedi nota.
