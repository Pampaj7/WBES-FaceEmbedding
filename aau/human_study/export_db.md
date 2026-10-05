# Esportare le risposte della versione ospitata

La pagina `hosted/index.html` scrive un documento per partecipante in `responses/<codice>`
(piu' `progress/<codice>`, cancellato appena la scrittura finale va a buon fine): si leggono
con il tool `read_db` dell'orchestratore sulla collection `responses`, e ogni documento va
salvato come un file JSON dentro una directory, uno per documento.

Poi l'analisi e' quella di sempre, con `--from-dir` al posto di `--responses-dir`:

    aau/human_study/analyze.py --from-dir aau/human_study/responses_db

`--from-dir` accetta il corpo del documento nudo, imbustato sotto `data`/`document`/`body`,
o una lista di documenti in un file solo, e converte `trials`/`choice`/`control`/`shown_left`
nel formato dei JSON scaricati dalla versione offline: screening sugli attention check,
maggioranze, bootstrap e kappa restano gli stessi. `--responses-dir` continua a funzionare
com'era per i file consegnati a mano.
