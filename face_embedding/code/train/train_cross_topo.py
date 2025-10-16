import torch
import numpy as np
from config import CFG
from dataset.mesh_dataset import MeshDataset
from dataset.sampler import CrossTopoPairedSampler
from dataset.operators_cache import OperatorCache
from models.backbone import DiffusionBackbone
from models.losses import triplet_loss


def train():
    device = CFG["device"]
    ds_bfm = MeshDataset(CFG["bfm_dir"])
    ds_flm = MeshDataset(CFG["flame_dir"])
    sampler = CrossTopoPairedSampler(
        ds_bfm, ds_flm, batch_size=CFG["batch_size"])
    op_cache = OperatorCache(CFG["cache_dir"])

    model = DiffusionBackbone(c_in=3, c_width=CFG["c_width"], c_out=CFG["c_width"],
                              emb_dim=CFG["emb_dim"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=CFG["lr"])

    for epoch in range(1, CFG["max_epochs"] + 1):
        running = []
        for pairs in sampler:
            optim.zero_grad()
            loss_epoch = 0
            for idx, (sid, pb, pf) in enumerate(pairs):
                (mb, Lb, evb, U_b, gxb, gyb), Vb, Fb = op_cache.get_with_geo(
                    "BFM", pb, CFG["k_eig"])
                (mf, Lf, evf, Uf, gxf, gyf), Vf, Ff = op_cache.get_with_geo(
                    "FLAME", pf, CFG["k_eig"])
                tensors = lambda *t: [x.to(device) for x in t]
                mb, Lb, evb, U_b, gxb, gyb, Vb, Fb = tensors(
                    mb, Lb, evb, U_b, gxb, gyb, Vb, Fb)
                mf, Lf, evf, Uf, gxf, gyf, Vf, Ff = tensors(
                    mf, Lf, evf, Uf, gxf, gyf, Vf, Ff)
                z_b = model(Vb, mb, Lb, evb, U_b, gxb, gyb, Fb)
                z_f = model(Vf, mf, Lf, evf, Uf, gxf, gyf, Ff)
                loss_align = torch.nn.functional.mse_loss(z_b, z_f)
                neg_idx = (idx + 1) % len(pairs)
                _, pb_neg, pf_neg = pairs[neg_idx]
                (mbn, Lbn, evbn, Ub_n, gxbn, gybn), Vbn, Fbn = op_cache.get_with_geo(
                    "BFM", pb_neg, CFG["k_eig"])
                (mfn, Lfn, evfn, Uf_n, gxfn, gyfn), Vfn, Ffn = op_cache.get_with_geo(
                    "FLAME", pf_neg, CFG["k_eig"])
                mbn, Lbn, evbn, Ub_n, gxbn, gybn, Vbn, Fbn = tensors(
                    mbn, Lbn, evbn, Ub_n, gxbn, gybn, Vbn, Fbn)
                mfn, Lfn, evfn, Uf_n, gxfn, gyfn, Vfn, Ffn = tensors(
                    mfn, Lfn, evfn, Uf_n, gxfn, gyfn, Vfn, Ffn)
                z_b_neg = model(Vbn, mbn, Lbn, evbn, Ub_n, gxbn, gybn, Fbn)
                z_f_neg = model(Vfn, mfn, Lfn, evfn, Uf_n, gxfn, gyfn, Ffn)
                loss_trip = 0.5 * \
                    (triplet_loss(z_b, z_f, z_b_neg) +
                     triplet_loss(z_f, z_b, z_f_neg))
                loss = CFG["lambda_align"] * loss_align + loss_trip
                loss_epoch += loss
            loss_epoch = loss_epoch / len(pairs)
            loss_epoch.backward()
            optim.step()
            running.append(loss_epoch.item())
        print(f"[Epoch {epoch:03d}] loss={np.mean(running):.4f}")
    torch.save(model.state_dict(), "checkpoints/diffusionnet_shared.pt")
    print("✅ Model saved to checkpoints/diffusionnet_shared.pt")


if __name__ == "__main__":
    train()
