import numpy as np

# === LOAD ===
ix_23470_to_53490 = np.loadtxt("/home/pampalonil/data/utils/ix_23470_relative_to_53490.txt", dtype=int)
ix_53215_to_53490 = np.loadtxt("/home/pampalonil/data/utils/ix_53215_relative_to_53490.txt", dtype=int)

# === INVERT: 53490 → index in 53215 ===
lookup = {val: idx for idx, val in enumerate(ix_53215_to_53490)}

# === BUILD MAP: 23470 → 53215
ix_23470_to_53215 = []
for i in ix_23470_to_53490:
    if i in lookup:
        ix_23470_to_53215.append(lookup[i])
    else:
        raise ValueError(f"Index {i} not found in 53215->53490 map")

ix_23470_to_53215 = np.array(ix_23470_to_53215, dtype=int)
np.savetxt("ix_23470_relative_to_53215.txt", ix_23470_to_53215, fmt="%d")
print("✅ Saved: ix_23470_relative_to_53215.txt")
