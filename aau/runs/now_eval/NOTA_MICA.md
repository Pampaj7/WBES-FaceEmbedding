# MICA ritirato dai risultati di NoW (14 ottobre)

MICA (`aau/recon/mica_run.py`) produce la mesh con il modello FLAME 2020 (`generic_model.pkl`), che il progetto non
usa: per i dati e i modelli 3D sono ammessi solo FLAME 2023 Open e i modelli con licenza libera. Decisione del PI del
14 ottobre (`aau/runs/evidence/now_ranking/REGOLE.md`, sez. 6): MICA esce da ogni risultato riportato, compresi quelli
del 7 ottobre in questa cartella.

- `summary.md` e i csv di questa cartella sono rigenerati con `now_summarize.py --methods 3ddfa_v2,synergynet,prnet`:
  nessun numero con MICA. Il protocollo (`protocol.md`, scritto prima dei numeri) resta com'era, e la deviazione 1
  ("MICA aggiunto") e' superata da questa nota.
- Le conclusioni del 7 ottobre che dipendevano da MICA cadono: il tau per immagine del latente congiunto -0.236 (con
  MICA), la penalita' di tassellazione di MICA, l'AUC d'identita' 0.887 delle ricostruzioni MICA. Sui tre metodi
  pre-registrati valgono i numeri rigenerati.
- I file per metodo di MICA (`*_mica*.csv|json`, mai versionati) e le mesh in `~/data/now_recon/mica` restano fuori dal
  repo e non vanno citati. Le corse del passo `now` di `eval_body.sh`
  (`trainer_v3/ablations/c3f_eval/now/*`) contengono righe di MICA: non sono state riportate e non vanno citate.
- Il confronto fra metodi con la nostra metrica e' in `aau/runs/evidence/now_ranking/` (senza MICA).
