# Ablazione k_eig 64 / 128 / 256: protocollo dichiarato prima dei numeri FR/SR

Scritto il 10 ottobre 2026 dal coder su richiesta del PI, PRIMA di calcolare qualunque metrica FR/SR dei bracci
k. L'impronta sha256 di questo file sta nel messaggio del commit che lo aggiunge; ogni modifica successiva va in
un emendamento datato, separato, senza toccare questo file.

Cosa era gia' visibile quando l'ho scritto: i secondi per passo di k64 e k128 (0.82 e 0.99 s/passo, A100) e le
righe finali dei log di training di robal_k128 (eval online BFM: sp_clean 0.769, aucR 0.972 a 21.096 passi). Non
ho aperto le metriche delle vecchie eval di k64 e k128 (GT maxabs, `c3f_eval/*/scale_v3robal_k*`), ne' alcun
numero con GT FR o SR di questi bracci.

## Domanda

Quanti autovettori dell'operatore DiffusionNet (k_eig) usare nel run massivo: 64, 128 (lo stato attuale) o 256.

## Bracci (trainer v3, ricetta robal = arearobust + bal, seme 1234)

| braccio | job | operatori |
|---|---|---|
| robal_k128 | training 1065293 | store C3F di nv-ai-04 (k 128, pre-pass CPU di `prepass_v3.py`) |
| robal_k64 | training 1065295 | stesso store, primi 64 autovettori (`v3_work/stream/ktrunc.py`) |
| robal_k256 | da lanciare | store C3F a k 256: stessa geometria, stesso pre-pass (`ops_areanorm`), k_eig 256 |

Uguali in tutti: dati, split, blocchi e ordine della cella C3F, T = 21.096 passi, S = 293, checkpoint EMA alle
epoche 36 e 72, A100 di nv-ai-04, `--resume auto` con `--requeue`. Il modello non ha parametri che dipendono da k.

## Valutazione

Embedding dei checkpoint EMA `epoch072_ema.pth` (primario) ed `epoch036_ema.pth` (solo descrittivo), con gli
operatori delle viste di eval allo stesso k del training (k64: i primi 64 di k128; k256: calcolati a k 256).
Metriche e righe di `v3_work/trainer/tools/fact_paired.py` (importato, non modificato), calcolate da
`v3_work/trainer/tools/k_ablation_summary.py`:

- domini e gruppi: HIFI3D `nocrop_cross`, dev FaceScape `nocrop_cross` (vista neutra), FaceVerse con espressioni
  `mesh_pair_nocrop`; righe, GT e seme di `fact_paired.rows_for`;
- misura: Spearman graduato fra ||z_i - z_j|| e la GT, con GT **FR**, **SR** e **maxabs**;
- maschera comune per dominio (righe con tutte le distanze finite, colonne di `fact_paired` comprese), cosi' le
  righe sono quelle di `factorized_paired.csv`;
- IC 95% percentile con 1.000 repliche bootstrap per soggetto, seme del dominio come in `fact_paired.main`;
- delta appaiati sulle stesse repliche: k64 - k128, k256 - k128 e k256 - k64.

## Regola di adozione (checkpoint a 21.096 passi)

Si parte dal k piu' piccolo, k* = 64, e si sale in ordine (128, poi 256). Un k piu' grande sostituisce k* solo se
valgono TUTTE:

1. **Guadagno:** su HIFI3D, lo Spearman con GT FR **oppure** con GT SR sale di almeno **0.03** (delta k - k*,
   stima puntuale) e l'**estremo inferiore dell'IC 95% appaiato e' > 0**;
2. **Nessuna perdita altrove:** in nessuna delle altre celle (HIFI3D, dev FaceScape, FaceVerse x FR, SR, maxabs)
   il delta k - k* scende sotto **-0.03** (stima puntuale);
3. **Costo sostenibile per il run massivo:**
   - tempo: s/passo del training di k (media delle epoche 2-72 nel `train.log`, stesso hardware) al massimo
     **1.25 x** quello di k*;
   - disco: lo store degli operatori del run massivo a k (byte per mesh dall'indice dello store C3F a k, per il
     numero di mesh del run massivo) sta sul disco locale del nodo di training lasciando almeno il 10% libero, e
     non toglie spazio alla regola dei 300 GB liberi su CephFS. Se al momento della decisione il volume del run
     massivo non e' fissato, il costo e' "non dimostrato" e il criterio 3 non e' soddisfatto.

Altrimenti si tiene k*. Il checkpoint a 10.548 passi e il resto sono descrittivi, non entrano nella regola.

## Limiti dichiarati

- **Un solo seme (1234).** Gli IC coprono il campionamento dei soggetti di eval, non la variabilita' del training
  fra semi: una differenza di 0.03 puo' essere dello stesso ordine di quella fra due semi della stessa ricetta.
  Un esito positivo e' un'indicazione da confermare con un secondo seme prima del run massivo, non una prova.
- k64 e' una troncatura di k128 e k128 coincide coi primi 128 di k256 entro la tolleranza di ARPACK (verificato
  a campione in `bench_k256.json`, misura in corso mentre scrivo): i bracci differiscono solo per k, non per il
  calcolo degli autovettori. Se la verifica fallisce, lo si dichiara nei risultati.
- La testa e' quella standard (z unico, GT maxabs nel training): FR e SR misurano quanto lo stesso z ordina le
  coppie secondo le due GT, non una testa fattorizzata.
