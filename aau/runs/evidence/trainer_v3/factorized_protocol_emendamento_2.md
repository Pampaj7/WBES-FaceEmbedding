# factorized_protocol.md, emendamento 2 (9 ottobre 2026, 15:15, PRIMA di qualunque numero dei run del protocollo)

Protocollo (`514f10e1...`) ed emendamento 1 (`15482ead...`) invariati. Stato: i run C3F non sono partiti (store in
costruzione); il run C3M 1062932 e' stato fermato dopo lo staging del primo blocco, prima di qualunque riga d'epoca o
eval, e rilanciato con questo emendamento.

**Dropout 0 in tutti i run del protocollo** (`factorized` e `ctrlfr` C3F, `factorized` C3M): `--dropout 0.0` invece di
0.1 della ricetta. Motivo, dal run breve di stabilita' sui dati di prova (job 1062802, NON un run del protocollo):
sulle mesh BFM di TRAINING l'errore di s rispetto a log S_i e' -0.21 in modo eval e ~0 in modo train (-0.002 con e
senza rumore latente; `factorized/short_1062802/`, controllo `mode_check`): il dropout dell'encoder cambia le feature
in modo non lineare fra train ed eval, e uno scalare di taglia che deve valere all'1% non lo tollera. Applicato anche a
`ctrlfr` perche' il confronto factorized contro ctrlfr non dipenda dal dropout. Conseguenza dichiarata: rispetto a
e108 (dropout 0.1) i run differiscono anche nella regolarizzazione.
