import torch
import torch.nn as nn
import torch.nn.functional as F

"""
pairiwse distance da gt, e poi quelle ricostruite
Questa funzione di perdita è progettata per valutare la qualità della ricostruzione
di una mesh 3D in termini di coerenza geometrica locale e globale. Integra tre
termini distinti — L1, Normale e Laplaciano — ciascuno con un ruolo complementare:

**L1 Loss (Vertex-level error)**
    - Misura l'errore punto a punto tra i vertici ricostruiti e quelli ground-truth.
    - È sensibile alla scala e alla traslazione, e garantisce che la rete impari
      a ricostruire correttamente le posizioni 3D dei vertici nello spazio canonico.

**Normal Loss (Surface orientation consistency)**
    - Confronta le normali per vertice tra la mesh predetta e quella reale.
    - È calcolata come *1 - cos(θ)*, dove θ è l’angolo tra le due normali.
    - Penalizza deviazioni nella direzione delle superfici, quindi cattura
      la coerenza della forma indipendentemente da piccole traslazioni.

**Laplacian Loss (Cosine Similarity on curvature field)**
    - È la componente più “geometrica”: misura la similarità tra i campi di curvatura
      delle due superfici, calcolati come L·V, dove L è il Laplaciano discreto.
    - Invece del classico MSE su L(V), che è numericamente instabile, usa la
      *similarità coseno* per confrontare le direzioni delle curvature locali.
    - Questo la rende **invariante alla scala e alle rotazioni**, ma **sensibile
      solo alle differenze di forma e curvatura** tra le due mesh.
    - In pratica, verifica che la “forma locale” (prossimità, concavità, convessità)
      sia preservata, indipendentemente dalla posizione assoluta nello spazio.

**Sintesi concettuale**
    - L1 → garantisce accuratezza spaziale globale.
    - Normal → garantisce coerenza della superficie.
    - Laplacian Cosine → garantisce consistenza geometrica locale.
    - La combinazione dei tre produce una misura robusta e bilanciata
      di ricostruzione geometrica.

**Vantaggio della versione Cosine**
    - Evita instabilità numeriche (overflow) tipiche delle differenze MSE su L(V),
      dove piccole variazioni di scala producono errori enormi.
    - Mantiene stabilità del gradiente e allena la rete a rispettare
      la *direzione geometrica* delle variazioni locali piuttosto che la loro magnitudine.

**In sintesi**
    Questa loss non misura semplicemente “quanto” due superfici differiscono,
    ma *come* differiscono nella struttura geometrica. È quindi una misura
    di **somiglianza di forma** più che di distanza euclidea pura.
────────────────────────────────────────────────────────────────────────────
"""


class GeometricLoss(nn.Module):
    """
    Una loss geometrica ibrida per la ricostruzione di mesh. (Versione Pura)
    
    Logica v4 (Cosine Similarity):
    Usa la Similarità Coseno per il Laplaciano.
    
    🌟 FIX v4.1 (Logging):
    Corregge il dizionario 'loss_breakdown' per loggare la
    loss Coseno reale, non il vecchio valore MSE esplosivo.
    """
    def __init__(self, w_l1=1.0, w_normal=1.0, w_laplacian=0.1, device='cuda'):
        super().__init__()
        
        self.w_l1 = w_l1
        self.w_normal = w_normal
        self.w_laplacian = w_laplacian
        
        self.l1_loss = nn.L1Loss()
        self.device = device
        
        print(f"🧬 GeometricLoss Pura (con Cosine Laplacian) creata con pesi: L1={w_l1}, Normal={w_normal}, Laplacian={w_laplacian}")

    def compute_vertex_normals(self, verts, faces):
        """
        Calcola le normali per vertice usando PyTorch puro.
        """
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        
        e1 = v1 - v0 
        e2 = v2 - v0 
        
        face_normals = torch.cross(e1, e2, dim=1) 

        vertex_normals = torch.zeros_like(verts) 
        
        vertex_normals.scatter_add_(0, faces[:, 0].unsqueeze(1).expand(-1, 3), face_normals)
        vertex_normals.scatter_add_(0, faces[:, 1].unsqueeze(1).expand(-1, 3), face_normals)
        vertex_normals.scatter_add_(0, faces[:, 2].unsqueeze(1).expand(-1, 3), face_normals)

        vertex_normals = F.normalize(vertex_normals, p=2, dim=1, eps=1e-6)
        
        return vertex_normals

    def forward(self, V_rec, V_gt, faces, L_sparse):
        """
        Calcola la loss totale con Cosine Laplacian.
        """
        
        loss_l1 = torch.tensor(0.0, device=self.device)
        loss_normal = torch.tensor(0.0, device=self.device)
        loss_laplacian = torch.tensor(0.0, device=self.device)
        
        # --- 1. L1 Loss (Vertex Loss) ---
        if self.w_l1 > 0:
            loss_l1 = self.l1_loss(V_rec, V_gt)

        # --- 2. Normal Loss (Geometric) ---
        if self.w_normal > 0:
            normals_gt = self.compute_vertex_normals(V_gt, faces)
            normals_rec = self.compute_vertex_normals(V_rec, faces)
            
            cos_sim_normal = F.cosine_similarity(normals_rec, normals_gt, dim=1)
            cos_sim_normal = torch.nan_to_num(cos_sim_normal, nan=1.0)
            loss_normal = (1.0 - cos_sim_normal).mean()

        # --- 3. Laplacian Loss (con Similarità Coseno) ---
        
        delta_gt = torch.sparse.mm(L_sparse, V_gt)
        delta_rec = torch.sparse.mm(L_sparse, V_rec)
        
        if self.w_laplacian > 0 and torch.isfinite(delta_gt).all() and torch.isfinite(delta_rec).all():
            
            cos_sim_laplacian = F.cosine_similarity(delta_rec, delta_gt, dim=1)
            cos_sim_laplacian = torch.nan_to_num(cos_sim_laplacian, nan=1.0)
            
            # Questa è la loss Laplaciana REALE (un numero piccolo)
            loss_laplacian = (1.0 - cos_sim_laplacian).mean()
            
        # --- 4. Combinazione ---
        
        total_loss = (self.w_l1 * loss_l1) + \
                     (self.w_normal * loss_normal) + \
                     (self.w_laplacian * loss_laplacian) # Questa è la loss reale
                     
        loss_breakdown = {
            "loss_total": total_loss.item(),
            "loss_l1": loss_l1.item(),
            "loss_normal": loss_normal.item(),
            # Logghiamo la loss Coseno REALE, non l'MSE esplosiva
            "loss_laplacian": loss_laplacian.item(), 
        }

        return total_loss, loss_breakdown