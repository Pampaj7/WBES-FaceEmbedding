# NoW validation: classifica dei metodi con NoW e con la nostra metrica — risultati

Regole: `REGOLE.md` (commit `f1bf27e`, prima dei numeri). Licenze e percorsi: `README.md`. Tutte le tabelle complete
sono in `numbers.md` (scritto da `aau/recon/now_ranking.py`), i csv in questa cartella e in `tables/`.

## Esito

**INFORMATIVO POSITIVO in entrambi i semi** (REGOLE sez. 4). Su 5 metodi e 352 immagini (20 soggetti, nessuna immagine
mancante), d_P ordina i metodi immagine per immagine come NoW originale, e meglio della Chamfer grezza:

| | s1234 | s2345 |
|---|---|---|
| tau per immagine, d_P contro NoW originale | **+0.430 [+0.343, +0.517]** | **+0.420 [+0.332, +0.504]** |
| delta contro Chamfer grezza (tau +0.206 [+0.069, +0.318]) | +0.224 [+0.068, +0.401] | +0.214 [+0.053, +0.383] |
| delta contro ICP + Chamfer in mm (tau +0.595 [+0.520, +0.675]) | -0.165 [-0.231, -0.086] | -0.176 [-0.231, -0.109] |

IC 95% bootstrap per soggetto (1000 repliche, seme 1234). ICP + Chamfer allinea alla scansione e misura in mm come
NoW: e' il tetto atteso (REGOLE sez. 2), e d_P resta sotto. Metro di paragone (descrittivo): fra i due benchmark
ufficiali, NoW metrical contro NoW originale, il tau per immagine e' +0.388 [+0.258, +0.514]. Ma per tre metodi su
cinque NoW metrical vede una scala scelta dal protocollo, quindi questo valore non e' un tetto.

Sensibilita' (descrittiva), sui tre metodi del 7 ottobre: d_P +0.419 [+0.361, +0.479] (s1234) e +0.350 [+0.263, +0.438]
(s2345). Chamfer grezza +0.246 e ICP + Chamfer +0.612 riproducono i numeri del 7 ottobre. Delta contro Chamfer: +0.172
[+0.063, +0.286] (s1234), +0.104 [-0.028, +0.229] (s2345, IC con lo 0).

## Classifiche dei metodi (descrittive)

Punteggio: NoW = mediana di tutte le distanze concatenate, come il benchmark; le altre misure = media per immagine.
Piccolo = meglio. Fra parentesi la probabilita' bootstrap di essere primo.

| criterio | 1 | 2 | 3 | 4 | 5 | tau con NoW orig. |
|---|---|---|---|---|---|---|
| NoW originale (mm) | SynergyNet 1.350 (0.77) | 3DDFA_V2 1.390 | HRN 1.508 | 3DDFA-V3 1.521 | PRNet 1.563 | — |
| NoW metrical (mm) | SynergyNet 1.550 (0.87) | 3DDFA_V2 1.592 | PRNet 1.803 | HRN 1.910 | 3DDFA-V3 4.145 | +0.60 |
| d_P s1234 | SynergyNet 0.060 (1.00) | 3DDFA_V2 0.065 | PRNet 0.068 | 3DDFA-V3 0.071 | HRN 0.076 | +0.40 |
| d_P s2345 | SynergyNet 0.062 (1.00) | 3DDFA_V2 0.067 | PRNet 0.069 | 3DDFA-V3 0.076 | HRN 0.078 | +0.40 |
| Chamfer grezza | 3DDFA-V3 0.042 (1.00) | SynergyNet 0.047 | HRN 0.048 | 3DDFA_V2 0.049 | PRNet 0.050 | +0.20 |
| ICP + Chamfer (mm) | SynergyNet 2.79 (0.99) | 3DDFA_V2 2.89 | 3DDFA-V3 3.16 | PRNet 3.32 | HRN 3.34 | +0.60 |

d_P mette primi gli stessi due metodi di NoW (SynergyNet, 3DDFA_V2) e li separa in modo netto. Fra NoW originale e
d_P si invertono solo gli ultimi tre: NoW mette PRNet ultimo, d_P mette HRN ultimo. Sui tre metodi di coda NoW
originale ha IC sovrapposti (1.36-1.77 mm). Con 10 coppie di metodi il tau globale non e' un test.

## Altri descrittivi (`numbers.md`)

- **Per sfida**: tau per immagine di d_P s1234 fra +0.36 (occlusioni) e +0.47 (espressioni, selfie). Tutti gli IC sono
  sopra 0, mentre la Chamfer grezza ha IC con lo 0 su occlusioni e selfie.
- **Dentro il metodo** (Spearman fra immagini, d_P s1234 contro NoW originale): 3DDFA_V2 +0.65, SynergyNet +0.74,
  PRNet +0.74, 3DDFA-V3 +0.69, HRN +0.51. Chamfer grezza: +0.42..+0.59; ICP + Chamfer: +0.83..+0.87.
- **Contro NoW metrical**: d_P +0.31 / +0.33, Chamfer grezza -0.13, ICP + Chamfer +0.39.
- **d_F alla taglia del template**: tau con NoW originale +0.410 (s1234) e +0.384 (s2345). Dista poco da d_P perche'
  S e' quasi costante (patch alla taglia del template).
- **Metodi con unita' note** (3DDFA-V3, HRN; patch con allineamento rigido, mm veri):
  - d_F metrica contro NoW metrical: tau per immagine +0.82 / +0.81, che con 2 metodi e' l'accordo sul migliore dei
    due; Spearman dentro il metodo +0.92 (3DDFA-V3) e +0.51 (HRN);
  - d_P sulle stesse patch: segue NoW originale (+0.39 / +0.43) e non NoW metrical. E' la divisione attesa: d_F vede
    la taglia, d_P no;
  - 3DDFA-V3 in mm ha volti piccoli: raggio del ritaglio NoW 81 mm, contro 94 mm di HRN e 100 mm delle scansioni.
    Questo spiega i 4.1 mm di NoW metrical.

## Deviazioni (14 ottobre)

1. **GPU**: tutte le 48 L40S erano occupate. 3DDFA-V3 ha ricostruito su una V100 (nv-ai-03, job 1079327); HRN e gli
   embedding sono girati su A10 (job 1079841, 1079823, 1079850): pytorch3d e nvdiffrast di HRN sono compilati per
   sm_86 / sm_89. E' solo inferenza.
2. **HRN ammesso**: pytorch3d compilato in 315 s (job 1078933, sotto l'ora). Per farlo girare sono serviti interventi
   da fuori al clone, documentati in `aau/recon/hrn_run.py`: rasterizzatore CUDA al posto di OpenGL,
   `LandmarksType._3D`, `POS` appiattito per numpy recente, facce della mesh alta ricomposte 0-based. Mesh = la
   "high mesh" del demo; i 7 landmark vengono dai vertici chiave della mesh media. Nessuna immagine senza volto.
3. **Detector**: 3DDFA-V3 e HRN usano il proprio (retinaface + 106 landmark del demo, primo volto). Gli altri tre usano
   FaceBoxes col box piu' grande, come il 7 ottobre.
4. **Riproduzione**: per i tre metodi vecchi, ||u_r - u_s|| ricalcolato coincide con la corsa precedente del passo
   `now` di `eval_body.sh` entro lo 0.5% relativo (max 3.4e-3 assoluto). Il confronto e' stato fatto valore per valore,
   senza leggere concordanze. Lo scarto viene dal tipo di GPU.
5. **Tabelle di scala** delle patch metriche: dominio `now` (u = 1 mm), lo stesso delle patch alla taglia del template.
   Il testo di `unit_source` ("patch T7") e' quello del dominio e non cambia nulla.
6. **MICA ritirato** anche dal 7 ottobre: `aau/runs/now_eval/NOTA_MICA.md`, riassunto rigenerato sui tre metodi.

## Come rifare

```
aau/submit.sh recon/tddfa_v3_setup_env.sbatch; aau/submit.sh recon/hrn_pytorch3d_build.sbatch; aau/submit.sh recon/hrn_setup_env.sbatch
WBES_NOW_METHODS=3ddfa_v3 aau/submit.sh recon/now_recon.sbatch; WBES_NOW_METHODS=hrn aau/submit.sh recon/now_recon.sbatch
aau/submit.sh recon/now_ranking_prep.sbatch
WBES_RANKING_NEW=hrn WBES_RANKING_METHODS=hrn WBES_RANKING_METRIC=hrn aau/submit.sh recon/now_ranking_prep.sbatch
WBES_RANKING_NEW="3ddfa_v3 hrn" WBES_RANKING_METRIC="3ddfa_v3 hrn" aau/submit.sh recon/now_ranking_embed.sbatch
aau/run.sh aau/recon/now_ranking.py
```
