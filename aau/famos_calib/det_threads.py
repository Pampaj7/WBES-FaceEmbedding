#!/usr/bin/env python3
"""``pr_noise_detect.py`` con i thread di onnxruntime fissati (``ORT_THREADS``): senza, ogni sessione apre un thread per
core della macchina e piu' rilevatori in parallelo nello stesso job si soffocano a vicenda. Stessi modelli e immagini.

    aau/recon/run_recon.sh mica aau/famos_calib/det_threads.py aau/competitors/protocolli/pr_noise_detect.py --dir <d>
"""
import os
import runpy
import sys

import onnxruntime as ort

_init = ort.InferenceSession.__init__


def _init_threads(self, path, sess_options=None, *a, **k):
    so = sess_options or ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "4"))
    so.inter_op_num_threads = 1
    _init(self, path, so, *a, **k)


ort.InferenceSession.__init__ = _init_threads
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
