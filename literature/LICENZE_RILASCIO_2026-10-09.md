# Licenze per il rilascio del dataset e dei pesi (9 ottobre 2026)

Ricerca di un agente Sonnet sulle pagine ufficiali. NON è un parere legale: prima della pubblicazione va rivista dall'ufficio legale o di trasferimento tecnologico di AAU.

| Fonte | Mesh | Distanze | Coefficienti/semi | Pesi |
|---|---|---|---|---|
| GNM Head (google/GNM, Apache-2.0) | sì (NOTICE) | sì | sì | sì |
| ICT-FaceKit Light (MIT) | sì | sì | sì | sì |
| FLAME 2023 **Open** (CC BY 4.0 con restrizioni d'uso: no porno o deepfake) | sì (credito) | sì | sì | sì |
| FLAME 2020 / 2023 standard | no | dubbio | solo ricetta | no |
| BFM 2019 (§3.3: niente derivati né distribuzione) | no | dubbio | solo ricetta | no |
| MPI: FaMoS, CoMA, D3DFACS, VOCASET, NoW | no | no o dubbio | no | no (ps-license@tue.mpg.de) |
| Florence 4D (CC BY 4.0 dichiarata, ma contiene identità CoMA) | non risolto | non risolto | non risolto | non risolto |
| FaceScape, HIFI3D, FaceVerse | no | dubbio | solo ricetta | dubbio |
| Multiface (CC BY-NC 4.0) | solo non commerciale | solo non commerciale | solo non commerciale | solo non commerciale |

## Strategia
1. **Nucleo aperto:** GNM + ICT Light + FLAME 2023 Open. Mesh, coefficienti, semi e matrici rilasciabili; avvisi Apache, MIT e CC BY 4.0; pesi rilasciabili con licenza CC BY 4.0.
2. **Solo ricetta e semi:** BFM 2019, FLAME 2020, MPI, FaceScape, HIFI3D, FaceVerse.
3. **Pesi addestrati su dati misti:** non vanno rilasciati senza un permesso scritto (MPI: ps-license@tue.mpg.de; Basilea/Unitectra: mail@unitectra.ch). In alternativa si rilascia un secondo modello addestrato SOLO sul nucleo aperto.
4. **Persone reali:** consenso e GDPR (dati biometrici) da verificare.
