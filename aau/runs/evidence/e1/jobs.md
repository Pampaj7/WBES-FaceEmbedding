
## Sottomissione 2026-10-08_15:37:16
- c2m: training 1061659 (--mem 430G, --time 12:00:00), eval 1061660 (afterok:1061659)
- c2f: training 1061661 (--mem 430G, --time 10:00:00), eval 1061662 (afterok:1061661)
- c3f: training 1061663 (--mem 410G, --time 10:00:00), eval 1061664 (afterok:1061663)
- g1: training 1061665 (--mem 270G, --time 10:00:00), eval 1061666 (afterok:1061665)
- c3m: eval FaceVerse + FLAME 1061667 (HIFI3D e NoW: quelli di aau/runs/data_scale_ood)
- riepilogo 1061668 (afterany sulle eval)

## 2026-10-08_15:46:29: eval e riepilogo risottomessi con --kill-on-invalid-dep=yes (se un training fallisce la sua eval si cancella e il riepilogo parte comunque); sostituiscono 1061660 1061662 1061664 1061666 1061668
- c2m: eval 1061700 (afterok:1061659)
- c2f: eval 1061701 (afterok:1061661)
- c3f: eval 1061702 (afterok:1061663)
- g1: eval 1061703 (afterok:1061665)
- riepilogo 1061704 (afterany:1061700:1061701:1061702:1061703:1061667)
