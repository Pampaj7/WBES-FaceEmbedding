# Encoder 3D agnostico distillato dal riconoscimento 2D: piano

Scritto il 6 ottobre 2026. Il piano parte solo se passa il test ArcFace-render su FaceVerse (`aau/runs/arcface_render_zs/`).

## Tesi
Un encoder di mesh agnostico alla discretizzazione (DiffusionNet) che produce direttamente un embedding d'identità. La conoscenza d'identità viene distillata da un modello di riconoscimento facciale 2D (ArcFace) applicato a render della sola geometria. Il risultato è una metrica d'identità 3D che:
- lavora su mesh di qualunque connettività, risoluzione e qualità;
- non richiede render, allineamento né registrazione al test, quindi è veloce e il retrieval è O(1) per mesh;
- porta la conoscenza d'identità del mondo reale, non quella di un 3DMM sintetico;
- generalizza fuori dominio.

## Insegnante
- ArcFace `buffalo_l/w600k_r50` su render grigi ombreggiati, a 3 viste (yaw 0, ±30) e 512 px, con crop fisso calibrato. È la pipeline di `aau/multiface/ws3a_render.py`.
- Embedding: media delle viste, poi normalizzazione L2, 512 dimensioni.
- Alternativa da valutare: AdaFace.
- Licenza dei pesi da verificare (insightface: ricerca non commerciale).
- **L'insegnante vede sempre la versione "pulita" della mesh**, cioè la topologia `original`. Lo studente vede la versione perturbata: remesh, decimazione, rumore, upsampling ed eventualmente crop. Così lo studente impara l'invarianza alla discretizzazione, che l'insegnante non ha.

## Studente
- DiffusionNet con gli stessi operatori della ricetta attuale, l'head su un embedding a 512 dimensioni e normalizzazione L2.
- Ingresso: xyz più eventualmente HKS. Il frame è un'ablazione: canonico, augmentation di rotazione, oppure solo HKS.

## Loss
1. **Distillazione puntuale:** coseno fra l'embedding dello studente (mesh perturbata) e quello dell'insegnante (mesh pulita).
2. **Distillazione relazionale:** preservare la matrice di similarità dell'insegnante dentro il batch. Conta per il retrieval e per il ranking graduato.
3. **Opzionale, quando ci sono etichette:** contrastiva d'identità fra espressioni e topologie della stessa persona. Rinforza l'invarianza all'espressione.

## Dati di training
L'insegnante non richiede etichette, quindi qualunque mesh facciale va bene.
- BFM (REMESH), ICT-5000 e la valanga ICT (50.000 identità con 8 espressioni).
- GNM Head (identità reali, con espressioni).
- **Domini di test esclusi dal training (leave-one-out):** FaceVerse, HIFI3D e Multiface reale. Anche FLAME, se arriva la licenza.
- Perturbazioni: le 6 topologie esistenti più le espressioni.

## Valutazione
- **Protocollo primario:** riconoscimento (rank-1, mAP, AUC di verifica) senza crop, con il crop a parte. È lo stesso protocollo del benchmark di identità sotto espressione.
- **Domini fuori training:** FaceVerse, HIFI3D, GNM (nel leave-one-out che lo esclude) e Multiface reale.
- **Baseline:**
  - NICP, oggi il riferimento (0.959 su FaceVerse);
  - ICP + Chamfer;
  - Chamfer;
  - l'insegnante stesso (ArcFace-render);
  - la metrica appresa da sintetico (congiunto).
- **Obiettivi:**
  - (a) battere Chamfer e il congiunto fuori dominio;
  - (b) eguagliare o battere l'insegnante sulle coppie cross-topologia e con rumore. L'insegnante soffre su noisy (0.932 su Multiface);
  - (c) avvicinarsi a NICP con un costo di retrieval molto più basso.
- Il tempo per confronto va misurato e riportato.

## Passi
1. **Gate:** il test ArcFace-render su FaceVerse e HIFI3D, contro NICP.
2. **Generatore di etichette dell'insegnante a scala:** si misura il costo del render per mesh e si parte da un sottoinsieme, per esempio 20k mesh.
3. **Trainer di distillazione:** si riusa il loader di DiffusionNet e gli operatori calcolati su /tmp.
4. **Pilota:** BFM + ICT-5000, test su FaceVerse e HIFI3D. Se batte Chamfer e il congiunto, si scala con la valanga e GNM.
5. **Ablazioni:** perdita puntuale contro relazionale, frame/HKS, numero di viste dell'insegnante, quantità di dati.
6. **Critic su ogni risultato**, con il protocollo dichiarato prima dei numeri.

## Rischi
- **L'insegnante su render potrebbe non scalare a 100+ identità.** Il gate serve a scoprirlo.
- **Lo studente eredita i bias dell'insegnante:** dipendenza dall'illuminazione e dalla vista. Va mitigato con più viste e luci diverse in training.
- **Costo del render a scala.** Va misurato prima di generare.
- **Licenza ArcFace.**
