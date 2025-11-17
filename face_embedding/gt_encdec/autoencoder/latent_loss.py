# latent_loss.py
# Loss functions for "latent-aware" Diffusion Autoencoder training
# Date: 2025-11

import torch
import torch.nn.functional as F


# ============================================================
# 1) Stress Loss (scale-invariant, robust)
# ============================================================
def stress_loss(Z: torch.Tensor, D: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compare pairwise Euclidean distances in latent space vs given target distances.
    Robust to scale: both distance matrices are normalized by their off-diagonal mean.
    Uses Smooth L1 to tame outliers and averages across pairs.
    """
    B = Z.size(0)
    if B < 2:
        return torch.tensor(0.0, device=Z.device, dtype=Z.dtype)

    # pairwise distances in latent space
    diff = Z[:, None, :] - Z[None, :, :]
    Dz = torch.sqrt((diff * diff).sum(dim=-1) + eps)

    # remove diagonal
    device = D.device
    eye = torch.eye(B, dtype=torch.bool, device=device)
    Dz = Dz.masked_fill(eye, 0.0)
    Dn = D.masked_fill(eye, 0.0)

    # normalize by off-diagonal mean
    off = ~eye
    mean_Dz = Dz[off].mean().clamp_min(eps)
    mean_Dn = Dn[off].mean().clamp_min(eps)
    Dz = Dz / mean_Dz
    Dn = Dn / mean_Dn

    # Smooth L1 on off-diagonal entries (mean reduction avoids gigantic sums)
    loss = F.smooth_l1_loss(Dz[off], Dn[off], reduction="mean")

    # keep the value finite in case of rare numerical spikes
    loss = torch.nan_to_num(loss, nan=0.0, posinf=10.0, neginf=0.0).clamp_max(10.0)
    return loss


# ============================================================
# 2) VarCov Loss (VICReg-like, numerically stable)
# ============================================================
def varcov_loss(z_global: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Combines variance and decorrelation constraints on global latents.
    - Uses mean reductions (not sums) to keep scale small.
    - Guards for small batch sizes (B<2) and degenerate std.
    Expected input: z_global shape (B, D).
    """
    if z_global.dim() != 2:
        raise ValueError("Invalid input shape: expected z_global (B, D).")

    B, D = z_global.shape
    if B < 2:
        # With a single sample, covariance is undefined; return 0 to avoid noise.
        return torch.tensor(0.0, device=z_global.device, dtype=z_global.dtype)

    # zero-mean
    z_centered = z_global - z_global.mean(dim=0, keepdim=True)

    # per-dimension std (unbiased=False is more stable for small B)
    var = z_centered.var(dim=0, unbiased=False)
    std = torch.sqrt(var + eps)

    # Encourage std >= 1 (VICReg-style); average so scale is independent of D
    var_loss = F.relu(1.0 - std).mean()

    # whiten by std, then compute covariance of normalized features
    z_norm = z_centered / (std.unsqueeze(0) + eps)
    cov = (z_norm.T @ z_norm) / (B - 1)  # (D, D)

    # off-diagonal entries only
    off_diag = cov - torch.diag(torch.diag(cov))
    # mean over all off-diagonal cells (not sum) => scale ~ O(1)
    cov_loss = (off_diag ** 2).mean()

    total = var_loss + cov_loss
    total = torch.nan_to_num(total, nan=0.0, posinf=10.0, neginf=0.0).clamp_max(10.0)
    return total


# ============================================================
# 3) Smooth Loss (Laplacian regularizer on per-vertex latent field)
# ============================================================
def smooth_loss(Z_field: torch.Tensor, L) -> torch.Tensor:
    """
    Penalizes high-frequency variations in the per-vertex latent field.
    Z_field: (n_verts, D)
    L: Laplacian operator (torch.sparse_coo_tensor or dense 2D tensor)
    Implementation details:
    - Works with sparse or dense L (coalesce if sparse).
    - Uses mean reductions across vertices and channels to avoid large sums.
    """
    if Z_field.dim() != 2:
        raise ValueError("Expected Z_field with shape (n_verts, D).")

    # Apply Laplacian
    if torch.is_tensor(L) and L.layout == torch.sparse_coo:
        # ensure coalesced for stable multiplication
        if not L.is_coalesced():
            L = L.coalesce()
        Lz = torch.sparse.mm(L, Z_field)  # (N, D)
    else:
        # treat as dense
        Lz = L @ Z_field  # (N, D)

    # Mean over all entries to keep scale small and resolution-invariant
    loss = (Lz ** 2).mean()
    loss = torch.nan_to_num(loss, nan=0.0, posinf=10.0, neginf=0.0).clamp_max(10.0)
    return loss


def triplet_loss(Z, D_gt, margin=0.2):
    """Triplet margin loss using GT distances as structure prior."""
    n = Z.shape[0]
    if n < 3:
        return torch.tensor(0.0, device=Z.device)
    dist_lat = torch.cdist(Z, Z, p=2)
    loss_accum = 0.0
    triplets = 0
    for i in range(n):
        pos_idx = torch.argmin(D_gt[i] + torch.eye(n, device=Z.device)[i] * 1e9)
        neg_idx = torch.argmax(D_gt[i])
        ap = dist_lat[i, pos_idx]
        an = dist_lat[i, neg_idx]
        loss = torch.clamp(ap - an + margin, min=0.0)
        loss_accum += loss
        triplets += 1
    return loss_accum / max(triplets, 1)


# ============================================================
# 4) Combined Latent Loss (optional aggregation)
# ============================================================
def latent_loss_combined(z_global, Z_field, D_orig_batch, L,
                         λ_rank=1.0, λ_varcov=1.0, λ_smooth=0.1) -> dict:
    """
    Combines all latent-space losses into a single dict for logging and backward pass.
    Uses safe guards and mean-based reductions internally.
    """
    if D_orig_batch is None or (isinstance(D_orig_batch, torch.Tensor) and D_orig_batch.numel() == 0):
        L_rank = torch.tensor(0.0, device=z_global.device, dtype=z_global.dtype)
    else:
        L_rank = stress_loss(z_global, D_orig_batch)

    L_varcov = varcov_loss(z_global)
    L_smooth = smooth_loss(Z_field, L) if Z_field is not None else torch.tensor(
        0.0, device=z_global.device, dtype=z_global.dtype
    )

    total = λ_rank * L_rank + λ_varcov * L_varcov + λ_smooth * L_smooth
    total = torch.nan_to_num(total, nan=0.0, posinf=10.0, neginf=0.0)

    return {
        "total": total,
        "rank": L_rank.detach(),
        "varcov": L_varcov.detach(),
        "smooth": L_smooth.detach(),
    }



def stress_loss_with_scale(Z: torch.Tensor, D: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Stress loss con penalty per scale mismatch."""
    B = Z.size(0)
    if B < 2:
        return torch.tensor(0.0, device=Z.device, dtype=Z.dtype)

    diff = Z[:, None, :] - Z[None, :, :]
    Dz = torch.sqrt((diff * diff).sum(dim=-1) + eps)

    device = D.device
    eye = torch.eye(B, dtype=torch.bool, device=device)
    Dz = Dz.masked_fill(eye, 0.0)
    Dn = D.masked_fill(eye, 0.0)

    off = ~eye
    mean_Dz = Dz[off].mean().clamp_min(eps)
    mean_Dn = Dn[off].mean().clamp_min(eps)
    
    Dz_norm = Dz / mean_Dz
    Dn_norm = Dn / mean_Dn

    shape_loss = F.smooth_l1_loss(Dz_norm[off], Dn_norm[off], reduction="mean")
    scale_ratio = mean_Dz / mean_Dn
    scale_penalty = (scale_ratio - 1.0) ** 2
    
    total = shape_loss + 0.1 * scale_penalty
    return torch.nan_to_num(total, nan=0.0, posinf=10.0).clamp_max(10.0)


def hard_negative_mining_loss(Z: torch.Tensor, D_gt: torch.Tensor, top_k: int = 20) -> torch.Tensor:
    """Focus on worst pairs - FIXED: normalize before MSE."""
    B = Z.size(0)
    if B < 2:
        return torch.tensor(0.0, device=Z.device)
    
    with torch.no_grad():
        diff = Z[:, None, :] - Z[None, :, :]
        D_lat = torch.sqrt((diff ** 2).sum(-1) + 1e-8)
        
        D_lat_n = (D_lat - D_lat.min()) / (D_lat.max() - D_lat.min() + 1e-8)
        D_gt_n = (D_gt - D_gt.min()) / (D_gt.max() - D_gt.min() + 1e-8)
        
        error = torch.abs(D_lat_n - D_gt_n)
        
        mask = ~torch.eye(B, dtype=torch.bool, device=Z.device)
        error_flat = error[mask]
        D_lat_flat = D_lat[mask]
        D_gt_flat = D_gt[mask]
        
        k = min(top_k, len(error_flat))
        hard_idx = torch.topk(error_flat, k=k).indices
    
    # FIX: normalize before computing loss
    D_lat_norm = (D_lat_flat - D_lat_flat.min()) / (D_lat_flat.max() - D_lat_flat.min() + 1e-8)
    D_gt_norm = (D_gt_flat - D_gt_flat.min()) / (D_gt_flat.max() - D_gt_flat.min() + 1e-8)
    
    loss = F.mse_loss(D_lat_norm[hard_idx], D_gt_norm[hard_idx])
    return loss


def distortion_regularizer(D_lat: torch.Tensor, D_gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Penalizza distorsioni non-uniformi - FIXED: clipping."""
    mask = ~torch.eye(D_lat.size(0), dtype=torch.bool, device=D_lat.device)
    ratio = D_lat[mask] / (D_gt[mask] + eps)
    ratio = ratio.clamp(0.01, 100.0)
    distortion = ratio.std()
    return distortion


def percentile_matching_loss(D_lat: torch.Tensor, D_gt: torch.Tensor, 
                             percentiles: list = [10, 25, 50, 75, 90]) -> torch.Tensor:
    """Forza percentili a matchare."""
    loss = torch.tensor(0.0, device=D_lat.device)
    
    mask = ~torch.eye(D_lat.size(0), dtype=torch.bool, device=D_lat.device)
    lat_flat = D_lat[mask]
    gt_flat = D_gt[mask]
    
    for p in percentiles:
        qt_lat = torch.quantile(lat_flat, p / 100.0)
        qt_gt = torch.quantile(gt_flat, p / 100.0)
        loss += (qt_lat - qt_gt) ** 2
    
    return loss / len(percentiles)


def multiscale_distance_loss(Z: torch.Tensor, D_gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Multi-scale loss: cosine (angoli) + Euclidean (magnitudine).
    Immune a curse of dimensionality perché usa distanza angolare.
    """
    B = Z.size(0)
    if B < 2:
        return torch.tensor(0.0, device=Z.device)
    
    # Cosine distance - cattura struttura angolare
    Z_norm = F.normalize(Z, p=2, dim=1)
    cos_sim = torch.mm(Z_norm, Z_norm.T)
    D_cos = 1.0 - cos_sim
    
    # Euclidean chord distance
    diff = Z_norm[:, None, :] - Z_norm[None, :, :]
    D_euc = torch.sqrt((diff ** 2).sum(-1) + eps)
    
    # Combination: 70% angular, 30% magnitude
    D_lat = 0.7 * D_cos + 0.3 * D_euc
    
    # Normalize
    mask = ~torch.eye(B, dtype=torch.bool, device=Z.device)
    D_lat_flat = D_lat[mask]
    D_gt_flat = D_gt[mask]
    
    D_lat_n = (D_lat_flat - D_lat_flat.min()) / (D_lat_flat.max() - D_lat_flat.min() + eps)
    D_gt_n = (D_gt_flat - D_gt_flat.min()) / (D_gt_flat.max() - D_gt_flat.min() + eps)
    
    return F.smooth_l1_loss(D_lat_n, D_gt_n)


def curriculum_distance_mask(D_gt: torch.Tensor, epoch: int, max_epochs: int = 15) -> torch.Tensor:
    """
    Curriculum learning: early epochs focus on nearby pairs.
    Progressively include distant pairs.
    """
    # Distance threshold grows from 0.3 to 1.0 over first 15 epochs
    max_dist = min(1.0, 0.3 + epoch * (0.7 / max_epochs))
    
    # Create mask for valid distances
    mask = D_gt <= max_dist
    # Always keep diagonal masked out
    mask = mask & ~torch.eye(D_gt.size(0), dtype=torch.bool, device=D_gt.device)
    
    return mask, max_dist

def scale_loss(Z: torch.Tensor, target_mean: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Penalizza se la distanza media off-diagonal in latent space
    è troppo lontana da target_mean.

    Non guarda la GT, serve solo a fissare una scala assoluta sensata.
    """
    B = Z.size(0)
    if B < 2:
        return Z.new_tensor(0.0)

    # pairwise distances
    diff = Z[:, None, :] - Z[None, :, :]
    Dz = torch.sqrt((diff * diff).sum(dim=-1) + eps)

    # off-diagonal mask
    eye = torch.eye(B, dtype=torch.bool, device=Z.device)
    off = ~eye

    mean_dist = Dz[off].mean().clamp_min(eps)

    # L2 su (mean_dist - target)
    loss = (mean_dist - target_mean).pow(2)
    return loss