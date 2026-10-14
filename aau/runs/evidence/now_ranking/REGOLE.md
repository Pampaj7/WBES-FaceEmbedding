# NoW validation: classifica dei metodi di ricostruzione con NoW e con la nostra metrica — regole (scritte PRIMA dei numeri)

Domanda (revisore YJz1, applicazione dichiarata della metrica): sulle ricostruzioni reali di NoW validation, la nostra
d_P ordina i metodi di ricostruzione 3D da immagine come il benchmark NoW ufficiale? Decisioni del PI del 14 ottobre
(fase 2): metodi, riserva BFM09, ruolo descrittivo di NoW metrical e d_F sui metodi in spazio immagine, MICA fuori.

Stato al commit di questo file:
- **noti**: i numeri di `aau/runs/now_eval` (7 ottobre: NoW ufficiale, Chamfer, ICP + Chamfer, latente congiunto,
  ArcFace su 3DDFA_V2, SynergyNet, PRNet e MICA), compresi i tau per immagine di Chamfer grezza (+0.246) e
  ICP + Chamfer (+0.612) sui tre metodi senza MICA;
- **calcolati ma non letti per questa analisi**: gli u dei factorized s1234 / s2345 sui tre metodi
  (`trainer_v3/ablations/c3f_eval/now/v3factorized{,s2345}e072`, passo `now` di `eval_body.sh`), mai riportati
  (`factorized.md`: "NoW escluso"). Qui si ricalcolano con la stessa catena e si usano solo i nuovi;
- **inesistenti**: ricostruzioni di 3DDFA-V3 e HRN, NoW metrical su NoW reale, d_F su NoW.

## 1. Metodi

- **3DDFA_V2, SynergyNet, PRNet**: le mesh esistenti (`~/data/now_recon`, 352 / 352), runner `aau/recon/*_run.py`.
- **3DDFA-V3** (Wang et al., CVPR 2024; codice MIT, clone `e15385837`; asset da Hugging Face
  `Zidu-Wang/3DDFA-V3`, nessuna registrazione): backbone resnet50 e detector del demo (retinaface + landmark). La
  geometria e' la catena di `face_model.forward()` fino a `v3d`, senza i render (che servono solo alle immagini del
  demo); nessuna riga del clone cambiata.
- **HRN**: entra solo se pytorch3d si compila entro 1 h (decisione del PI). La decisione si prende sul tempo di
  compilazione, prima di qualunque metrica su HRN; se esce, lo si dichiara.
- **Esclusi**: MICA, DECA, EMOCA, SMIRK (FLAME 2020); Pixel3DMM (FLAME 2023 non Open, e preelaborazione con MICA);
  Deep3DFaceRecon (BFM09 solo col modulo di Basilea).
- **Riserva BFM09**: 3DDFA_V2, 3DDFA-V3 (e HRN) usano basi derivate da BFM09 ridistribuite dagli autori. Uso solo
  per valutazione di ricerca, coi pesi di inferenza pubblicati; nessuna mesh ricostruita e nessuna base in git
  (`README.md` qui accanto).
- Immagini senza ricostruzione: mancanti, non imputate. Il tau per immagine usa le immagini ricostruite da TUTTI i
  metodi dell'insieme; i conteggi vanno nei risultati.

## 2. Criteri (ogni ricostruzione contro la scansione del suo soggetto)

- **NoW originale**: `external/now_evaluation/compute_error.py` ufficiale (commit `7bd1498`), mesh intera e 7
  landmark (iBUG 36, 39, 42, 45, 33, 48, 54) come in `now_eval`; punteggio dell'immagine = mediana delle distanze
  per vertice della scansione ritagliata (`now_median`).
- **NoW metrical**: lo stesso codice con `metrical_eval=True` e `predicted_mesh_unit='mm'` (Procrustes senza scala,
  ICP rigido). Mesh in mm:
  - 3DDFA-V3 (e HRN): dalle unita' del modello. Deep3DFaceRecon `util/load_mats.py` divide le basi BFM09 per 1e5
    ("unify the scale to decimeter"): 1 unita' = 100 mm. Controllo PRIMA dei numeri: distanza ex-ex della forma
    media fra 80 e 110 mm; se fallisce, ci si ferma e lo si riporta;
  - 3DDFA_V2, SynergyNet, PRNet (spazio immagine): la mesh portata alla taglia del template T7 (similarita' dai
    SUOI landmark, `now_common.to_canonical`). **Descrittivo** (decisione del PI): la scala la sceglie il
    protocollo, non il metodo.
- **d_P (nostra, primaria)**: factorized **s1234** (replica **s2345**), `epoch072_ema.pth`, i checkpoint della
  calibrazione (`factorized_calibration.csv`). Patch NoW alla taglia del template, con la catena di `now_eval`
  (`now_prepare_meshes.py`, operatori ad area unitaria k_eig 128) e di `eval_body.sh` (`eval_v3.py`, tabelle di
  scala con `unit_source` "patch T7"). d_P = c x dp_per_unit x ||u_rec - u_scan||; c e dp_per_unit sono costanti e
  non cambiano nessun ordine.
- **d_F**: sqrt((S_r - S_s)^2 + S_r S_s d_P^2), S = exp(s) (`factorized_v3.form_distance`, d_P calibrata con c).
  Sulle patch alla taglia del template per tutti i metodi (S quasi costante: descrittiva). **d_F metrica** solo per
  3DDFA-V3 (e HRN): patch con allineamento RIGIDO al template (Procrustes senza scala sui 7 landmark), scansione e
  ricostruzione in mm, stesso ritaglio NoW; descrittiva.
- **Termini di paragone**: Chamfer grezza (`chamfer_raw`, `now_geometric.py`) = comparatore primario; ICP + Chamfer
  in mm (`icp_chamfer_mm`) = secondario (e' quasi la misura di NoW: tetto atteso, non un risultato).

## 3. Statistica primaria

- **tau per immagine**: per ogni immagine, Kendall tau-a fra l'ordine dei K metodi secondo d_P e secondo NoW
  originale (coppie pari = 0); media sulle immagini.
- **IC 95%** percentile, bootstrap per soggetto: 20 soggetti, 1000 repliche, seme 1234; peso dell'immagine = conteggio
  del suo soggetto nella replica (`now_summarize.concordance`). Le stesse repliche servono tutte le righe.
- **Delta** = tau(d_P) - tau(Chamfer grezza), IC appaiato sulle stesse repliche.
- Insieme dei metodi: tutti quelli inclusi in sez. 1. Sensibilita' (descrittiva): i tre metodi del 7 ottobre.

## 4. Esito (per seme; l'esito finale e' quello comune a s1234 e s2345, altrimenti NON CONCLUSIVO)

- **INFORMATIVO POSITIVO**: IC di tau(d_P) tutto sopra 0 **e** IC del delta non tutto sotto 0.
- **NEGATIVO**: IC di tau(d_P) tutto sotto 0, **oppure** IC del delta tutto sotto 0.
- **NON CONCLUSIVO**: ogni altro caso.

Con 4-5 metodi e 20 soggetti gli IC saranno larghi. La classifica globale ha K(K-1)/2 coppie (6 o 10) e resta
descrittiva: nessun verdetto sull'ordine dei metodi in se'.

## 5. Descrittivi (nessun verdetto)

- Metro di paragone: tau per immagine fra NoW metrical e NoW originale, e fra ICP + Chamfer e NoW originale.
- Classifica per criterio: media dei punteggi per immagine (NoW anche la mediana di tutte le distanze concatenate,
  come il benchmark), IC bootstrap, P(primo); tau globale fra le classifiche.
- tau per immagine dentro ciascuna sfida (neutral, expressions, occlusions, selfie).
- d_F (taglia del template) e d_P contro NoW originale e contro NoW metrical.
- d_F metrica contro NoW metrical dove le unita' sono note: Spearman fra le immagini dentro il metodo, IC per
  soggetto; tau per immagine solo se i metodi metrici sono almeno due.
- Spearman dentro ciascun metodo fra d_P e NoW originale.

## 6. MICA

MICA produce la mesh con FLAME 2020, che non usiamo: esce da ogni risultato riportato, compresi quelli del 7 ottobre.
Nota in `aau/runs/now_eval/NOTA_MICA.md`; il riassunto di `now_eval` si rigenera sui tre metodi.

## 7. Uscite

`results.md` e i csv per immagine e per metodo in questa cartella; mesh, patch, operatori ed embedding fuori dal repo
(`~/data/now_recon`, `~/data/now_eval_work`, `~/data/now_ranking_work`). Ogni scostamento da queste regole va in
`results.md`, sezione "Deviazioni", con la data.
