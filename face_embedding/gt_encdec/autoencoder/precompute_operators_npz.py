import os
import sys
import torch
import igl
# 🌟 Import NumPy
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

# === Add DiffusionNet path ===
# Assicurati che questi path siano corretti per il tuo sistema
if "/equilibrium/lpampaloni/diffusion-net/src" not in sys.path:
    sys.path.append("/equilibrium/lpampaloni/diffusion-net/src")
if "/home/pampaj/diffusion-net/src" not in sys.path:
    sys.path.append("/home/pampaj/diffusion-net/src")
if "/seidenas/users/lpampaloni/diffusion-net/src" not in sys.path:
    sys.path.append("/seidenas/users/lpampaloni/diffusion-net/src")


try:
    from diffusion_net.geometry import compute_operators
except ImportError as e:
    print(f"Errore: Impossibile importare diffusion_net.geometry. Assicurati che il path sia corretto.")
    print(f"Errore specifico: {e}")
    sys.exit(1)

# === CONFIG ===
DATA_DIR = "../../../datasets/GT_ready/" # Dove sono gli .obj originali
NPZ_OUT_DIR = os.path.join(DATA_DIR, "npz_data") # Nuova cartella per gli .npz
K_EIG = 128
os.makedirs(NPZ_OUT_DIR, exist_ok=True)

# === Funzione per salvare tensori sparsi in formato COO ===
def sparse_to_coo_dict(sparse_tensor, base_name):
    """Converte un tensore sparso PyTorch in un dizionario di array NumPy COO."""
    if not sparse_tensor.is_sparse:
        # Se per qualche motivo non è sparso, salvalo come denso
        print(f"[WARN] Il tensore {base_name} non è sparso, salvataggio come denso.")
        return {base_name: sparse_tensor.detach().cpu().numpy()}

    coalesced = sparse_tensor.coalesce()
    indices = coalesced.indices().detach().cpu().numpy() # Shape [2, n_nonzero]
    values = coalesced.values().detach().cpu().numpy()   # Shape [n_nonzero]
    shape = coalesced.shape                             # Tuple (righe, colonne)
    return {
        f"{base_name}_indices": indices,
        f"{base_name}_values": values,
        f"{base_name}_shape": np.array(shape) # Salva shape come array
    }

# === FUNZIONE SINGOLA PER MESH ===
def process_mesh(obj_filename):
    print(f"Worker {os.getpid()}: Starting {obj_filename}")
    mesh_path = os.path.join(DATA_DIR, obj_filename)
    # 🌟 Path di output NPZ
    out_path = os.path.join(NPZ_OUT_DIR, obj_filename.replace(".obj", ".npz"))

    if os.path.exists(out_path):
        return "[skip]", obj_filename

    try:
        # 1. Carica mesh
        V_np, F_np = igl.read_triangle_mesh(mesh_path)
        if V_np.size == 0 or F_np.size == 0:
            return "[error]", f"{obj_filename}: Mesh vuota."
        
        # Converte in tensori per compute_operators
        V = torch.tensor(V_np, dtype=torch.float32)
        F = torch.tensor(F_np, dtype=torch.long)

        # 2. Calcola operatori
        ops = compute_operators(V, F, k_eig=K_EIG)
        # ops = (frames, mass, L, evals, evecs, gradX, gradY)

        # 3. Prepara i dati per il salvataggio
        data_to_save = {
            # Geometria base
            "verts": V_np, # Salviamo NumPy array originali
            "faces": F_np,

            # Operatori densi (convertiti in NumPy)
            "mass": ops[1].detach().cpu().numpy(),
            "evals": ops[3].detach().cpu().numpy(),
            "evecs": ops[4].detach().cpu().numpy(),
        }

        # 4. Aggiungi operatori sparsi in formato COO dict
        data_to_save.update(sparse_to_coo_dict(ops[2], "L"))      # Laplaciano
        data_to_save.update(sparse_to_coo_dict(ops[5], "gradX"))  # GradX
        data_to_save.update(sparse_to_coo_dict(ops[6], "gradY"))  # GradY

        # 5. Salva in formato NPZ compresso
        np.savez_compressed(out_path, **data_to_save)

        return "[ok]", obj_filename

    except Exception as e:
        import traceback
        # Stampa un errore più dettagliato in caso di fallimento
        # err_details = traceback.format_exc()
        return "[fail]", f"{obj_filename}: {str(e)}" # \n{err_details}"

# === MAIN (Parallel Processing) ===
if __name__ == "__main__":
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".obj")])

    # Limita opzionalmente per test
    # files = files[:100]
    
    #files = files[:5000]

    #files_subset = files[:5000] # Prende i primi 5000 (indici 0-4999)
    #files = files_subset[::-1]   # Inverte la lista (ora va da 4999 a 0)

    
    start_index = 1250
    end_index = 5000 # Lo slice [start:end] esclude 'end', quindi prendiamo fino a 4999
    if start_index >= len(files):
         print(f"[ERRORE] Start index {start_index} è fuori dai limiti ({len(files)} files)")
         files = [] # Lista vuota se non ci sono file in questo range
    else:
         files = files[start_index:min(end_index, len(files))] # Prende da 2500 a 4999
    
    
    """    end_index = 5000
    
    if end_index > 0:
         files_subset = files[:min(end_index, len(files))]
         files = files_subset[::-1] # Inverte la lista (ora va da 499 a 0)
    else:
         files = [] # Nessun file se end_index è 0 o negativo
    """
    
    n_cores = 1 
    print(f"🚀 Uso {n_cores} core CPU | k_eig={K_EIG}")
    print(f"📂 Input: {len(files)} mesh .obj da {DATA_DIR}")
    print(f"💾 Output: file .npz in {NPZ_OUT_DIR}")

    ok = skip = fail = 0
    results = []

    # Crea il pool di processi
    try:
        # Assicura che il metodo 'spawn' sia usato se possibile, più sicuro con PyTorch/CUDA
        current_start_method = mp.get_start_method(allow_none=True)
        if current_start_method != 'spawn':
             mp.set_start_method("spawn", force=True)
    except RuntimeError: # Può essere impostato solo una volta
        pass
    except ValueError: # Metodo non supportato su alcune piattaforme
         print("[WARN] Impossibile impostare start_method 'spawn'. Potrebbero verificarsi problemi.")
         pass


    with mp.Pool(n_cores) as pool:
        with tqdm(total=len(files), dynamic_ncols=True, desc="Precomputing NPZ data") as pbar:
            # Usa imap_unordered per ottenere risultati appena sono pronti
            for status, msg in pool.imap_unordered(process_mesh, files):
                results.append(f"{status} {msg}")
                if status == "[ok]": ok += 1
                elif status == "[skip]": skip += 1
                else: fail += 1
                pbar.set_postfix(ok=ok, skip=skip, fail=fail)
                pbar.update(1)

    # Salva log riassuntivo
    log_path = os.path.join(NPZ_OUT_DIR, "_precompute_log.txt")
    try:
        with open(log_path, "w") as fp:
            fp.write("\n".join(results))
    except IOError as e:
        print(f"[ERRORE] Impossibile scrivere il file di log {log_path}: {e}")


    print("\n✅ Pre-processing completato!")
    print(f"💾 Dati NPZ salvati in: {NPZ_OUT_DIR}")
    print(f"🧾 Log salvato in: {log_path}")
    print(f"📊 Riepilogo → OK: {ok} | Saltati: {skip} | Falliti: {fail}")

    if fail > 0:
        print("\n[ATTENZIONE] Ci sono stati errori durante il pre-processing.")
        print("Controlla il file di log per i dettagli sui file falliti.")