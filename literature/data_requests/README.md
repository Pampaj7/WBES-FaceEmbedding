# Richieste d'accesso ai dati — bozze pronte per la firma

Tre richieste, tre destinatari, tre latenze diverse. **Nessuna delle tre può essere firmata da
un dottorando**: due richiedono chi impegna legalmente l'ente, una il direttore dell'ufficio
ricerca.

| ente | cosa sblocca | chi firma | latenza attesa |
|---|---|---|---|
| Notre Dame CVRL | FRGC v2 + ND-2006 + **3D-TEC** + Collection D, un solo accordo | chi assume impegni legali per DTU | settimane |
| FaceScape (Nanjing) | 847 soggetti × 20 espressioni, topologia uniforme | mail istituzionale, licenza firmata | giorni |
| BU-3DFE (Binghamton) | 100 soggetti × 25, espressioni graduate | direttore dell'ufficio ricerca | settimane |

## Cosa sblocca ciascuno, in una riga

- **Notre Dame** è il solo che dà **variazione topologica reale**: ogni scan è un'acquisizione
  indipendente, quindi conteggio di vertici e connettività cambiano davvero, e l'etichetta è
  l'ID del soggetto — indipendente dalla geometria. È l'unica strada nota per rompere la
  circolarità. Dentro c'è anche **3D-TEC, 107 coppie di gemelli**: due persone con geometria
  quasi identica e identità diverse, cioè il test più affilato possibile contro un ground truth
  che confonde forma e identità.
- **FaceScape** dà l'**asse espressivo** (20 espressioni per soggetto), che separa «identità» da
  «configurazione». Topologia uniforme, quindi non aiuta sull'asse topologico.
- **BU-3DFE** dà espressioni a intensità graduata, utile come secondo dominio per il transfer.

## Prima di spedire

1. Sostituire i segnaposto in MAIUSCOLO: nome del firmatario, dipartimento, indirizzo, e il
   nome del supervisore.
2. Verificare che l'affermazione sull'uso previsto corrisponda a ciò che faremo davvero — le
   licenze vincolano l'uso, e dichiarare qualcosa di più ampio del necessario è un rischio
   inutile; dichiarare qualcosa di più stretto blocca il lavoro dopo.
3. **FLAME e BFM restano fuori da qualunque artefatto pubblico** (licenza). Vale già oggi, ma va
   ricordato quando si promette un rilascio nel testo della richiesta.
