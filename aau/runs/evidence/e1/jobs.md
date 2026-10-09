
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

## Sottomissione 2026-10-08_17:25:47 (catena corretta: emendamento del protocollo)
- c3ml (C3M L40S, riferimento): eval 1061841 su A100, subito
- cancello smoke V100 1061842 (afterany:1061840): tutti i training ne dipendono
- c2f: training 1061843 (V100, --mem 320G, --time 30:00:00, 32 CPU, pre-pass 22), eval 1061844 (A100, afterok:1061843)
- c3f: training 1061845 (V100, --mem 320G, --time 30:00:00, 32 CPU, pre-pass 22), eval 1061846 (A100, afterok:1061845)
- c2fgnm: training 1061847 (V100, --mem 240G, --time 30:00:00, 32 CPU, pre-pass 22), eval 1061848 (A100, afterok:1061847)
- c3fugt: cancello GT unificata 1061849
- c3fugt: training 1061850 (V100, --mem 320G, --time 30:00:00, 32 CPU, pre-pass 22), eval 1061851 (A100, afterok:1061850)
- c2fs2: training 1061852 (V100, --mem 320G, --time 30:00:00, 32 CPU, pre-pass 22), eval 1061853 (A100, afterok:1061852)
- c3fs2: training 1061854 (V100, --mem 320G, --time 30:00:00, 32 CPU, pre-pass 22), eval 1061855 (A100, afterok:1061854)
- c3mv: training 1061856 (V100, --mem 320G, --time 60:00:00, 32 CPU, pre-pass 22), eval 1061857 (A100, afterok:1061856)
- c2m: training 1061858 (V100, --mem 320G, --time 60:00:00, 32 CPU, pre-pass 22), eval 1061859 (A100, afterok:1061858)
- c2f40: training 1061860 (V100, --mem 320G, --time 30:00:00, 32 CPU, pre-pass 22), eval 1061861 (A100, afterok:1061860)
- c3f40: training 1061862 (V100, --mem 320G, --time 30:00:00, 32 CPU, pre-pass 22), eval 1061863 (A100, afterok:1061862)
- g1: training 1061864 (V100, --mem 240G, --time 30:00:00, 32 CPU, pre-pass 22), eval 1061865 (A100, afterok:1061864)
- riepilogo 1061866 (afterany sulle eval)

## 2026-10-08_17:47:15: build_grad vettorizzato adottato in tutte le celle (nota_tecnica_gradvec.md); TimeLimit dei training abbassati con scontrol: F e G1 24 h, c3mv e c2m 40 h. Nessun job risottomesso.

## 2026-10-09_00:54:53: emendamento 2 (GT unificata tarata)
- c3fugt (1061850) tenuto fermo fino alla verifica della GT tarata (1062071, check_calib.json ok), poi rilasciato: usa gt_unified_bfm_ict_gnm_calib.npz
- c3fugtraw: training 1062087 (V100, 320G, 24 h, in fondo alla coda), eval 1062088 (A100, afterok)
- riepilogo 1062089 (sostituisce 1061866; afterany su tutte le eval, compresa 1062088)

## 2026-10-09_12:59:58: correzioni
- eval C2F 1061844 FAILED per guasto di nv-ai-04 ('unknown userid 264582' alle 11:06, FLAME e NoW); HIFI3D, dev FaceScape e FaceVerse erano completi e restano
- c2fs2/c3fs2 (1061852/1061854) fermati dal controllo della spec (block_seed 2345 contro 1234 del riferimento): controllo corretto
- eval con ritentativi (3, a 10 minuti); cancellati e risottomessi nell'ordine di priorita' c3f40, g1, c3fugtraw e il riepilogo
- c2f: eval 1062674 (stesso checkpoint, rifa' solo FLAME e NoW)
- c2fs2: training 1062675 (V100, 320G, 24 h), eval 1062676
- c3fs2: training 1062677 (V100, 320G, 24 h), eval 1062678
- c3f40: training 1062679 (V100, 320G, 24 h), eval 1062680
- g1: training 1062681 (V100, 240G, 24 h), eval 1062682
- c3fugtraw: training 1062683 (V100, 320G, 24 h), eval 1062684
- riepilogo 1062685 (afterany:1061841:1062674:1061846:1061848:1061851:1061857:1061859:1061861:1062676:1062678:1062680:1062682:1062684)
