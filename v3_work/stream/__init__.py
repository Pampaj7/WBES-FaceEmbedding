"""Pipeline P1 del training massivo (paper/PLAN_MASSIVE.md sez. 8 e 14.3): produttori CPU e consumatori GPU
sullo stesso nodo, con un buffer ad anello di shard in RAM.

    produttori  v3_work/stream/producer.py   P processi: identita' fresche da tutti i domini di training, viste
                                             (espressione + discretizzazione), operatori DiffusionNet, shard
    anello      v3_work/stream/ring.py       un file per shard in una directory su /tmp (tmpfs = RAM), FIFO a
                                             budget di byte; scrittura tmp + rename, letto in mmap
    consumatore v3_work/stream/consumer.py   per rank DDP: shard in mmap (una copia per nodo, la page cache),
                                             riuso di ogni vista fino a R volte con rotazione e scala nuove,
                                             GT del batch al volo dai vettori s_i che viaggiano con le viste
    trainer     v3_work/stream/train_stream.py  il trainer v3 (v3_work/trainer/train_v3.py) con ``--stream``

Pezzi gia' pronti che si importano senza modificarli: la libreria 3DMM ``v3_work/mm``, i generatori delle
discretizzazioni (``v2_work/genict/make_ict_topologies.py``, ``mesh_ops.py``), gli operatori del pre-pass
(``aau/data_scale/prepass_ops.py::ops_areanorm``) col ``build_grad`` vettorizzato di E9
(``v3_work/trainer/grad_vec.py``), il loader congelato (``GTReadyDatasetNPZ``), lo spazio della GT unificata
(``v3_work/unified_gt/shapes.Space``).

Dati derivati (mai in git, ``datasets/`` e' ignorata): ``datasets/STREAM/cache`` (topologie di lavoro
decimate) e ``datasets/STREAM/maps`` (la mappa unificata di BFM 2019, ``bfm2019_map.py``).
"""
