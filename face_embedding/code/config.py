import torch

CFG = {
    "bfm_dir": "/path/to/BFM_meshes",
    "flame_dir": "/path/to/FLAME_meshes",
    "cache_dir": "./op_cache",
    "k_eig": 128,
    "c_width": 128,
    "emb_dim": 64,
    "batch_size": 2,
    "lr": 1e-4,
    "max_epochs": 20,
    "triplet_margin": 0.2,
    "lambda_align": 1.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
