# Protocollo dichiarato prima dei numeri: quasi-duplicati fra test e training (punto 9 del rebuttal)

Dichiarato il 2026-10-07 22:05 CEST, prima di calcolare qualunque distanza test-training. Noto in anticipo:
la guardia della valanga (`aau/data_scale/PLAN.md`): soglia 0.0055 = meta' del vicino piu' prossimo minimo di
ICT-5000 in vertex-mean-L2 maxabs; nessuna identita' nuova sotto soglia, vicino minimo 0.0107.

**Domanda (Z1mX).** I soggetti di test sono identita' distinte da quelle di training, o ci sono quasi-copie?

**Split.** (1) NeurIPS: BFM, 400 training / 100 test (`bfm_only` di `aau/runs/ws2_cross3dmm/splits.json`,
seme 1234; il test e' il set held-out di WS1). (2) Congiunto `x3dmm_joint_bfm_ict_s1234_1019532`: `joint` dello
stesso file, held-out controllati contro `aau/data_scale/heldout_frozen.json` (BFM 108, ICT 992). Il vicino
si cerca solo dentro lo stesso 3DMM (template diversi).

**Metrica primaria:** vertex-mean-L2 fra le `original` normalizzate maxabs, la metrica della guardia, cosi' la
soglia 0.0055 vale nelle sue unita'. **Secondaria:** la GT di training del run (normalizzata al massimo), con
la soglia ricavata dalla stessa regola (meta' del vicino piu' prossimo minimo nel pool del 3DMM).

**Misure.** Per ogni soggetto di test il NN verso il training; come termine di paragone il NN training ->
training (lascia-uno-fuori, pool di taglia uguale), il NN test -> test e la distribuzione di tutte le distanze
test-test. Codice: `aau/near_dup/near_duplicates.py`, un job CPU.

**Lettura fissata ora.** "Nessun quasi-duplicato" se nessun test ha NN verso il training <= soglia, in
entrambe le metriche. Se la distribuzione NN test -> training e' allineata a quella training -> training
(mediane entro il 10%), il test e' "distinto come due identita' di training fra loro". Ogni test sotto soglia
e' elencato per nome con il suo vicino.
