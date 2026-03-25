import torch

device = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

torch.backends.cudnn.benchmark = True

SAFETY_MB = 300
CHUNK_MB = 128

# ----------------------------
# 1. Trova size massima stabile
# ----------------------------
size = 8192

while True:
    try:
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)
        torch.cuda.synchronize()
        break
    except RuntimeError:
        size = int(size * 0.9)

print(f"Matrix size chosen: {size}x{size}")

# ----------------------------
# 2. Stress loop autoregolante
# ----------------------------
chunks = []
iteration = 0

while True:
    try:
        z = x @ y
        x = z @ y
        y = x @ z

        iteration += 1

        if iteration % 5 == 0:
            torch.cuda.synchronize()

            # riempi solo lo spazio realmente libero
            free_mem, _ = torch.cuda.mem_get_info()
            free_mb = free_mem / 1024**2

            if free_mb > (CHUNK_MB + SAFETY_MB):
                try:
                    num_floats = (CHUNK_MB * 1024**2) // 4
                    chunk = torch.empty(num_floats, dtype=torch.float32, device=device)
                    chunks.append(chunk)
                except RuntimeError:
                    pass

            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"Iter {iteration} | VRAM {allocated:.2f} GB")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("OOM during compute. Shrinking matrices.")

            # libera matrici
            for var in ["x", "y", "z"]:
                if var in locals():
                    del globals()[var]

            torch.cuda.empty_cache()

            size = int(size * 0.9)
            x = torch.randn(size, size, device=device)
            y = torch.randn(size, size, device=device)

        else:
            raise e