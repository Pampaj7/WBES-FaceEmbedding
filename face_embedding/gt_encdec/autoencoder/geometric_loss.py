import torch
import torch.nn as nn
import torch.nn.functional as F

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
                     
        # 🌟 === CORREZIONE LOGGING === 🌟
        loss_breakdown = {
            "loss_total": total_loss.item(),
            "loss_l1": loss_l1.item(),
            "loss_normal": loss_normal.item(),
            # Logghiamo la loss Coseno REALE, non l'MSE esplosiva
            "loss_laplacian": loss_laplacian.item(), 
        }
        # 🌟 =============================

        return total_loss, loss_breakdown